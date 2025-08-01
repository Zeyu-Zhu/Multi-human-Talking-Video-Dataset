import argparse
import copy
import logging
import math
import os
import json
from os import path
import numpy as np
import os.path as osp
import random
import time
import warnings
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict, List, Optional, Tuple, Union

import diffusers
import mlflow
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs
from diffusers import AutoencoderKL, DDIMScheduler
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
from einops import rearrange
from omegaconf import OmegaConf
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPVisionModelWithProjection

from src.dataset.stage_two import HumanDanceVideoDataset
from src.models.mutual_self_attention import ReferenceAttentionControl
from src.models.pose_guider import PoseGuider
from src.models.unet_2d_condition import UNet2DConditionModel
from src.models.unet_3d_audio import UNet3DConditionModel
from src.pipelines.pipeline_pose2vid_audio import Pose2VideoPipeline
from src.utils.util import (
    delete_additional_ckpt,
    import_filename,
    read_frames,
    save_videos_grid,
    seed_everything,
)

warnings.filterwarnings("ignore")

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.10.0.dev0")
logger = get_logger(__name__, log_level="INFO")
from src.dataset.dance_video_audio import keypoints_to_bbox, scale_bbox, calculate_square_bbox, bbox_intersection

def activate_score_normalize(scores):
    scores = torch.Tensor(scores)
    for index in range(len(scores)):
        scores[index] = scores[index]/3
    return scores
    
def generate_masks(frame_pose_list):
    full_mask_list = []
    face_mask_list = []
    lip_mask_list = []
    attention_mask_list = []
    for pose_list_index in range(frame_pose_list.shape[0]):
        pose_list = frame_pose_list[pose_list_index, ...]
        per_person_full_mask = []
        per_person_face_mask = []
        per_person_lip_mask = []
        per_person_attention_mask = []
        for pose_index in range(pose_list.shape[0]):
            pose = pose_list[pose_index]
            ## full mask
            left_point, right_point = pose[6], pose[5]
            length = (right_point[0] - left_point[0])*1.2
            left_up_point = [left_point[0], max(left_point[1] - length, 0)]
            right_up_point = [right_point[0], max(right_point[1] - length, 0)]
            keypoint_list = np.array([left_point, right_point, left_up_point, right_up_point, pose[23], pose[31], pose[39], pose[53]])
            bbox = keypoints_to_bbox(keypoint_list)
            full_bbox = scale_bbox(bbox, 1.4, 1.6)
            ## face mask
            points = np.array([pose[23], pose[31], pose[39], pose[53]])
            face_bbox = keypoints_to_bbox(points)
            face_bbox = scale_bbox(face_bbox, 1.4, 1.6)
            ## lip mask
            top_point, down_point = pose[53], pose[31]         
            lip_bbox = calculate_square_bbox(top_point, down_point)
            lip_bbox = scale_bbox(lip_bbox, 1.6, 0.8)
            ## union
            face_bbox = bbox_intersection(face_bbox, full_bbox)
            if face_bbox is None: face_bbox = full_bbox
            lip_bbox = bbox_intersection(lip_bbox, face_bbox)
            if lip_bbox is None: lip_bbox = face_bbox
            
            per_person_full_mask.append(get_mask_from_bbox(full_bbox))
            per_person_face_mask.append(get_mask_from_bbox(face_bbox))
            per_person_lip_mask.append(get_mask_from_bbox(lip_bbox))
            ## attention mask
            attention_mask_bbox = scale_bbox(full_bbox, 1.2)
            attention_mask = get_mask_from_bbox(attention_mask_bbox)
            per_person_attention_mask.append(attention_mask)
        full_mask_list.append(per_person_full_mask)
        face_mask_list.append(per_person_face_mask)
        lip_mask_list.append(per_person_lip_mask)
        attention_mask_list.append(per_person_attention_mask)     
    return attention_mask_list, full_mask_list, face_mask_list, lip_mask_list
    
def count_keypoints_in_bbox(keypoints, bbox):
    xmin, ymin, xmax, ymax = bbox
    in_bbox_count = 0
    for idx, (x, y) in enumerate(keypoints):
        if xmin <= x <= xmax and ymin <= y <= ymax:
            in_bbox_count += 1
    return in_bbox_count
    
def get_human_info(speaker_info_path, pose_json_dir, start_index, clip_length):
    ## masks: need to align with the human(check out if keypoints lie in instance bbox)
    with open(speaker_info_path, 'r') as f: infos = json.load(f)
    pose_name_list = sorted([pose_name for pose_name in os.listdir(pose_json_dir) if pose_name.endswith('.json')])
    pose_name_list = pose_name_list[start_index:start_index+clip_length]
        
    frame_pose_list = []
    for pose_name in pose_name_list: ## along frame
        with open(path.join(pose_json_dir, pose_name), 'r') as f: pose_list = json.load(f)['instance_info']
        pose_in_order = []
        for info in infos: ## along person
            instance_bbox, within_bbox_count = info['instance_bbox'], []
            for pose in pose_list: ## within 2 pose
                within_bbox_count.append(count_keypoints_in_bbox(pose['keypoints'], instance_bbox))
            match_index = np.argmax(within_bbox_count)
            pose_in_order.append(pose_list[match_index]['keypoints'])
        frame_pose_list.append(pose_in_order)
    frame_pose_list = np.array(frame_pose_list).transpose(1, 0, 2, 3) # (huamn, frame, 133, 2)
    # get the mask: attention mask for each person, full mask, face mask, lip mask
    attention_mask, full_mask, face_mask, lip_mask = generate_masks(frame_pose_list)
    # attention_mask, full_mask, face_mask, lip_mask = torch.zeros(1), torch.zeros(1), torch.zeros(1), torch.zeros(1)
    mask = dict(attention_mask=attention_mask, 
                    full_mask=full_mask, 
                    face_mask=face_mask, 
                    lip_mask=lip_mask)
    ## activate scores
    activate_scores = []
    for info in infos:
        activate_score = info['speaking_score'][start_index:start_index+clip_length]
        activate_scores.append(activate_score)
    activate_scores = activate_score_normalize(activate_scores)    
    activate_scores = activate_scores.unsqueeze(dim=-1)
    return activate_scores, mask
    
def get_mask_from_bbox(bbox):
    attention_mask_2d = np.zeros((1080, 1920), dtype=np.uint8)
    x_1, y_1, x_2, y_2 = bbox
    x_1, y_1, x_2, y_2 = int(x_1), int(y_1), int(x_2), int(y_2)
    x_1 = max(0, x_1)
    y_1 = max(0, y_1)
    x_2 = min(1920, x_2)
    y_2 = max(1080, y_2)
    attention_mask_2d[y_1:y_2, x_1:x_2] = 255
    return Image.fromarray(attention_mask_2d)

class Net(nn.Module):
    def __init__(
        self,
        reference_unet: UNet2DConditionModel,
        denoising_unet: UNet3DConditionModel,
        pose_guider: PoseGuider,
        reference_control_writer,
        reference_control_reader,
        pose_adaptor: Optional[PoseGuider]=None,
    ):
        super().__init__()
        self.reference_unet = reference_unet
        self.denoising_unet = denoising_unet
        self.pose_guider = pose_guider
        self.reference_control_writer = reference_control_writer
        self.reference_control_reader = reference_control_reader
        self.pose_adaptor = pose_adaptor
        
    def forward(
        self,
        noisy_latents,
        timesteps,
        ref_image_latents,
        clip_image_embeds,
        pose_img,
        ref_pose_img,
        uncond_fwd: bool = False,
        instance_bboxes: list = None,
        
        audio_embedding=None,
        face_mask_list=None,
        activate_score_list=None,
    ):
        pose_cond_tensor = pose_img.to(device="cuda")
        #pose_fea = self.pose_guider(pose_cond_tensor)
        pose_fea = None
        for instance_bbox in instance_bboxes:
            if pose_fea is None: pose_fea = self.pose_guider(pose_cond_tensor*instance_bbox)
            else: pose_fea += self.pose_guider(pose_cond_tensor*instance_bbox)

        ref_pose_fea = None
        if ref_pose_img is not None and self.pose_adaptor is not None:
            for instance_bbox in instance_bboxes:
                if ref_pose_fea is None: ref_pose_fea = self.pose_adaptor(ref_pose_img*instance_bbox)
                else: ref_pose_fea += self.pose_adaptor(ref_pose_img*instance_bbox)
            ref_pose_fea = ref_pose_fea.squeeze(2)
            
        if not uncond_fwd:
            ref_timesteps = torch.zeros_like(timesteps)
            self.reference_unet(
                ref_image_latents,
                ref_timesteps,
                ref_pose_fea=ref_pose_fea,
                encoder_hidden_states=clip_image_embeds,
                return_dict=False,
            )
            self.reference_control_reader.update(self.reference_control_writer)

        model_pred = self.denoising_unet(
            noisy_latents,
            timesteps,
            pose_cond_fea=pose_fea,
            encoder_hidden_states=clip_image_embeds,
            
            audio_embedding=audio_embedding,
            face_mask_list=face_mask_list,
            activate_score_list=activate_score_list
        ).sample

        return model_pred


def compute_snr(noise_scheduler, timesteps):
    """
    Computes SNR as per
    https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L847-L849
    """
    alphas_cumprod = noise_scheduler.alphas_cumprod
    sqrt_alphas_cumprod = alphas_cumprod**0.5
    sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5

    # Expand the tensors.
    # Adapted from https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L1026
    sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device=timesteps.device)[
        timesteps
    ].float()
    while len(sqrt_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_alphas_cumprod = sqrt_alphas_cumprod[..., None]
    alpha = sqrt_alphas_cumprod.expand(timesteps.shape)

    sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(
        device=timesteps.device
    )[timesteps].float()
    while len(sqrt_one_minus_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[..., None]
    sigma = sqrt_one_minus_alphas_cumprod.expand(timesteps.shape)

    # Compute SNR.
    snr = (alpha / sigma) ** 2
    return snr


def log_validation(
    vae,
    image_enc,
    net,
    scheduler,
    accelerator,
    width,
    height,
    clip_length=24,
    generator=None,
):
    logger.info("Running validation... ")

    ori_net = accelerator.unwrap_model(net)
    reference_unet = ori_net.reference_unet
    denoising_unet = ori_net.denoising_unet
    pose_guider = ori_net.pose_guider
    pose_adaptor = ori_net.pose_adaptor

    if generator is None:
        generator = torch.manual_seed(42)
    tmp_denoising_unet = copy.deepcopy(denoising_unet)
    tmp_denoising_unet = tmp_denoising_unet.to(dtype=torch.float16)

    pipe = Pose2VideoPipeline(
        vae=vae,
        image_encoder=image_enc,
        reference_unet=reference_unet,
        denoising_unet=tmp_denoising_unet,
        pose_guider=pose_guider,
        scheduler=scheduler,
        pose_adaptor=pose_adaptor,
    )
    pipe = pipe.to(accelerator.device)

    # test_cases = [
    #     ("/users/zeyuzhu/dataset_project/Datasets/fallowshow/datasets/fallowshow_109_000702to001185/frames/output_0001.png",
    #      "/users/zeyuzhu/dataset_project/Datasets/fallowshow/datasets/fallowshow_109_000702to001185/pose_frame"),
    #     ("/users/zeyuzhu/dataset_project/Datasets/fallowshow/datasets/fallowshow_111_010652to010857/frames/output_0001.png",
    #      "/users/zeyuzhu/dataset_project/Datasets/fallowshow/datasets/fallowshow_111_010652to010857/pose_frame"),
    #     # ("/users/zeyuzhu/dataset_project/Datasets/latenightshow/datasets/latenightshow_192_002244to002393/frames/output_0001.png",
    #     # "/users/zeyuzhu/dataset_project/Datasets/latenightshow/datasets/latenightshow_192_002244to002393/sparse_pose_frame"),
    #     ("/users/zeyuzhu/dataset_project/Datasets/latenightshow/datasets/latenightshow_60_000334to000585/frames/output_0001.png",
    #     "/users/zeyuzhu/dataset_project/Datasets/latenightshow/datasets/latenightshow_60_000334to000585/pose_frame"),
    #     ("/users/zeyuzhu/dataset_project/Datasets/latenightshow/datasets/latenightshow_65_000720to000846/frames/output_0001.png",
    #     "/users/zeyuzhu/dataset_project/Datasets/latenightshow/datasets/latenightshow_65_000720to000846/pose_frame"),
    #     ("/users/zeyuzhu/dataset_project/Datasets/latenightshow/datasets/latenightshow_1_001559to001809/frames/output_0001.png",
    #     "/users/zeyuzhu/dataset_project/Datasets/latenightshow/datasets/latenightshow_1_001559to001809/pose_frame"),

        # (
        #    "/users/zeyuzhu/dataset_project/Datasets/fallowshow/datasets/fallowshow_0_000838to001076/frames/output_0001.png",
        #     "/users/zeyuzhu/dataset_project/Datasets/fallowshow/datasets/fallowshow_0_000838to001076/sparse_pose_frame/",
        # ),
        # (
        #    "/users/zeyuzhu/dataset_project/Datasets/fallowshow/datasets/fallowshow_1_009155to009310/frames/output_0001.png",
        #     "/users/zeyuzhu/dataset_project/Datasets/fallowshow/datasets/fallowshow_1_009155to009310/sparse_pose_frame",
        # ),
        # (
        #    "/users/zeyuzhu/dataset_project/Datasets/fallowshow/datasets/fallowshow_0_000838to001076/frames/output_0001.png",
        #     "/users/zeyuzhu/dataset_project/Datasets/fallowshow/datasets/fallowshow_1_009155to009310/sparse_pose_frame",
        # ),
        # (
        #    "/users/zeyuzhu/dataset_project/Datasets/latenightshow/datasets/latenightshow_192_010312to010495/frames/output_0001.png",
        #     "/users/zeyuzhu/dataset_project/Datasets/fallowshow/datasets/fallowshow_0_000838to001076/sparse_pose_frame",
        # ),
    #]
    test_index = [0, 300]
    test_data_config = []
    with open("./all_data_config/test_easy.json") as f: easy_test_data = json.load(f)
    with open("./all_data_config/test_hard.json") as f: hard_test_data = json.load(f)
    for index in test_index: test_data_config.append(easy_test_data[index])
    for index in test_index: test_data_config.append(hard_test_data[index])
    
    results = []
    for test_data in test_data_config:
        img_path = test_data["image_dir"]
        pose_path = test_data["pose_dir"]
        speaker_info_path = test_data["speak_info_path"]
        audio_embeeding_path = test_data["audio_embedding_path"]
        with open(speaker_info_path, 'r') as f: infos = json.load(f)
        
        start_frame, end_frame = infos[0]["clip_range"][0]-1, infos[0]["clip_range"][1]
        img_list = sorted([path.join(img_path, name) for name in os.listdir(img_path)])[start_frame: end_frame]
        pose_list = sorted([path.join(pose_path, name) for name in os.listdir(pose_path)])[start_frame: end_frame]
        
        pose_json_dir = img_path.replace('frames', 'pose')
        print(pose_json_dir)
        ref_image_path = img_list[0]
        ref_image_pil = Image.open(ref_image_path).convert("RGB")
        ref_pose_pil = Image.open(ref_image_path.replace('frames', 'pose_frame')).convert("RGB")

        instance_bboxes = []
        for info in infos: instance_bboxes.append(get_mask_from_bbox(info['instance_bbox']))
        for index in range(len(instance_bboxes)):
            instance_bboxes[index] = torch.from_numpy(np.array(instance_bboxes[index].resize((width, height)))) / 255.0
            instance_bboxes[index] = instance_bboxes[index].unsqueeze(dim=0).unsqueeze(dim=0).unsqueeze(dim=0)
        
        ref_name = test_data["image_dir"].split('/')[-2]
        pose_name = str(infos[0]["clip_range"][0]) + '_' + str(infos[0]["clip_range"][0])
        
        pose_images = []
        for pose_path in pose_list:
            pose_images.append(Image.open(pose_path))
        #pose_images = read_frames(pose_video_path)
        pose_transform = transforms.Compose([transforms.Resize((height, width)), transforms.ToTensor()])
        # audio embedding
        audio_embedding = torch.load(audio_embeeding_path)
    
        result = []
        for frame_index in tqdm(range(0, len(pose_images)-clip_length, clip_length)):
            pose_list = []
            pose_tensor_list = []
            for pose_image_pil in pose_images[frame_index:frame_index+clip_length]:
                pose_tensor_list.append(pose_transform(pose_image_pil))
                pose_list.append(pose_image_pil)

            audio_embedding_clip = audio_embedding[frame_index:frame_index+clip_length, :, :]
            pose_tensor = torch.stack(pose_tensor_list, dim=0)  # (f, c, h, w)
            pose_tensor = pose_tensor.transpose(0, 1)
            # activate score and face mask
            activate_scores, mask = get_human_info(speaker_info_path=speaker_info_path, pose_json_dir=pose_json_dir,
                                                   start_index=frame_index, clip_length=clip_length,)
            
            pipeline_output = pipe(
            ref_image_pil,
            pose_list,
            instance_bboxes,
            width,
            height,
            clip_length,
            20,
            3.5,
            generator=generator,
            ref_pose_image=ref_pose_pil,
            
            audio_embedding=audio_embedding_clip,
            face_mask_list=mask['face_mask'],
            activate_score_list=activate_scores,
            )
            
            video = pipeline_output.videos
            pose_tensor = pose_tensor.unsqueeze(0)
            video = torch.cat([video, pose_tensor], dim=0)
            result.append(video)
            
        result = torch.cat(result, dim=2)
        results.append({"name": f"{ref_name}_{pose_name}", "vid": result})
        # for pose_image_pil in pose_images[:clip_length]:
        #     pose_tensor_list.append(pose_transform(pose_image_pil))
        #     pose_list.append(pose_image_pil)

        # pose_tensor = torch.stack(pose_tensor_list, dim=0)  # (f, c, h, w)
        # pose_tensor = pose_tensor.transpose(0, 1)

        # pipeline_output = pipe(
        #     ref_image_pil,
        #     pose_list,
        #     width,
        #     height,
        #     clip_length,
        #     20,
        #     3.5,
        #     generator=generator,
        # )
        # video = pipeline_output.videos
        # print(video.shape)
        # # Concat it with pose tensor
        # pose_tensor = pose_tensor.unsqueeze(0)
        # video = torch.cat([video, pose_tensor], dim=0)

    del tmp_denoising_unet
    del pipe
    torch.cuda.empty_cache()

    return results


def main(cfg):
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.solver.gradient_accumulation_steps,
        mixed_precision=cfg.solver.mixed_precision,
        log_with="mlflow",
        project_dir="./mlruns",
        kwargs_handlers=[kwargs],
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if cfg.seed is not None:
        seed_everything(cfg.seed)

    exp_name = cfg.exp_name
    save_dir = f"{cfg.output_dir}/{exp_name}"
    if accelerator.is_main_process:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    inference_config_path = "./configs/inference/inference_v2.yaml"
    infer_config = OmegaConf.load(inference_config_path)

    if cfg.weight_dtype == "fp16":
        weight_dtype = torch.float16
    elif cfg.weight_dtype == "fp32":
        weight_dtype = torch.float32
    else:
        raise ValueError(
            f"Do not support weight dtype: {cfg.weight_dtype} during training"
        )

    sched_kwargs = OmegaConf.to_container(cfg.noise_scheduler_kwargs)
    if cfg.enable_zero_snr:
        sched_kwargs.update(
            rescale_betas_zero_snr=True,
            timestep_spacing="trailing",
            prediction_type="v_prediction",
        )
    val_noise_scheduler = DDIMScheduler(**sched_kwargs)
    sched_kwargs.update({"beta_schedule": "scaled_linear"})
    train_noise_scheduler = DDIMScheduler(**sched_kwargs)

    image_enc = CLIPVisionModelWithProjection.from_pretrained(cfg.image_encoder_path,).to(dtype=weight_dtype, device="cuda")
    vae = AutoencoderKL.from_pretrained(cfg.vae_model_path).to("cuda", dtype=weight_dtype)
    
    reference_unet = UNet2DConditionModel.from_pretrained(
        cfg.base_model_path,
        subfolder="unet",
    ).to(device="cuda", dtype=weight_dtype)

    denoising_unet = UNet3DConditionModel.from_pretrained_2d(
        cfg.base_model_path,
        cfg.mm_path,
        subfolder="unet",
        unet_additional_kwargs=OmegaConf.to_container(
            infer_config.unet_additional_kwargs
        ),
    ).to(device="cuda")
    
    pose_guider = PoseGuider(
        conditioning_embedding_channels=320, block_out_channels=(16, 32, 96, 256)
    ).to(device="cuda", dtype=weight_dtype)

    pose_adaptor = PoseGuider(conditioning_embedding_channels=320,).to(device="cuda", dtype=weight_dtype)
    # pose_adaptor = None
    
    stage1_ckpt_dir = cfg.stage1_ckpt_dir
    stage1_ckpt_step = cfg.stage1_ckpt_step
    denoising_unet.load_state_dict(
        torch.load(
            os.path.join(stage1_ckpt_dir, f"denoising_unet-{stage1_ckpt_step}.pth"),
            map_location="cpu",
        ),
        strict=False,
    )
    
    if cfg.motion_module_path is not None:
        print('loading: motion_module', cfg.motion_module_path)
        denoising_unet.load_state_dict(torch.load(cfg.motion_module_path, map_location="cpu",), strict=False,)
    #print(len(denoising_unet.audio_up_module), len(denoising_unet.audio_down_module))
    
    reference_unet.load_state_dict(
        torch.load(
            os.path.join(stage1_ckpt_dir, f"reference_unet-{stage1_ckpt_step}.pth"),
            map_location="cpu",
        ),
        strict=False,
    )
    pose_guider.load_state_dict(
        torch.load(
            os.path.join(stage1_ckpt_dir, f"pose_guider-{stage1_ckpt_step}.pth"),
            map_location="cpu",
        ),
        strict=False,
    )
    pose_adaptor.load_state_dict(
        torch.load(
            os.path.join(stage1_ckpt_dir, f"pose_adaptor-{stage1_ckpt_step}.pth"),
            map_location="cpu",
        ),
        strict=False,
    )

    # Freeze
    vae.requires_grad_(False)
    image_enc.requires_grad_(False)
    reference_unet.requires_grad_(False)
    denoising_unet.requires_grad_(False)
    pose_guider.requires_grad_(False)
    pose_adaptor.requires_grad_(False)
    
    # Set audio module learnable
    #denoising_unet.audio_up_module.requires_grad_(True)
    #denoising_unet.audio_down_module.requires_grad_(True)
    #denoising_unet.audio_proj.requires_grad_(True)
    
    # Set motion module learnable
    for name, module in denoising_unet.named_modules():
        if "motion_modules" in name:
            for params in module.parameters():
                params.requires_grad = True
        if "audio" in name:
            for params in module.parameters():
                params.requires_grad = True
                
    reference_control_writer = ReferenceAttentionControl(
        reference_unet,
        do_classifier_free_guidance=False,
        mode="write",
        fusion_blocks="full",
    )
    reference_control_reader = ReferenceAttentionControl(
        denoising_unet,
        do_classifier_free_guidance=False,
        mode="read",
        fusion_blocks="full",
    )

    net = Net(
        reference_unet,
        denoising_unet,
        pose_guider,
        reference_control_writer,
        reference_control_reader,
        pose_adaptor=pose_adaptor,
    )
            
    if cfg.solver.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            reference_unet.enable_xformers_memory_efficient_attention()
            denoising_unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError(
                "xformers is not available. Make sure it is installed correctly"
            )

    if cfg.solver.gradient_checkpointing:
        reference_unet.enable_gradient_checkpointing()
        denoising_unet.enable_gradient_checkpointing()

    if cfg.solver.scale_lr:
        learning_rate = (
            cfg.solver.learning_rate
            * cfg.solver.gradient_accumulation_steps
            * cfg.data.train_bs
            * accelerator.num_processes
        )
    else:
        learning_rate = cfg.solver.learning_rate

    # Initialize the optimizer
    if cfg.solver.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    trainable_params = list(filter(lambda p: p.requires_grad, net.parameters()))
    optimizer = optimizer_cls(
        trainable_params,
        lr=learning_rate,
        betas=(cfg.solver.adam_beta1, cfg.solver.adam_beta2),
        weight_decay=cfg.solver.adam_weight_decay,
        eps=cfg.solver.adam_epsilon,
    )

    # Scheduler
    lr_scheduler = get_scheduler(
        cfg.solver.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=cfg.solver.lr_warmup_steps
        * cfg.solver.gradient_accumulation_steps,
        num_training_steps=cfg.solver.max_train_steps
        * cfg.solver.gradient_accumulation_steps,
    )

    train_dataset = HumanDanceVideoDataset(
        width=cfg.data.train_width,
        height=cfg.data.train_height,
        n_sample_frames=cfg.data.n_sample_frames,
        sample_rate=cfg.data.sample_rate,
        img_scale=(1.0, 1.0),
        data_meta_paths=cfg.data.meta_paths,
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=cfg.data.train_bs, shuffle=True, num_workers=4
    )

    # Prepare everything with our `accelerator`.
    (
        net,
        optimizer,
        train_dataloader,
        lr_scheduler,
    ) = accelerator.prepare(
        net,
        optimizer,
        train_dataloader,
        lr_scheduler,
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / cfg.solver.gradient_accumulation_steps
    )
    # Afterwards we recalculate our number of training epochs
    num_train_epochs = math.ceil(
        cfg.solver.max_train_steps / num_update_steps_per_epoch
    )

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        run_time = datetime.now().strftime("%Y%m%d-%H%M")
        accelerator.init_trackers(
            exp_name,
            init_kwargs={"mlflow": {"run_name": run_time}},
        )
        # dump config file
        mlflow.log_dict(OmegaConf.to_container(cfg), "config.yaml")

    # Train!
    total_batch_size = (
        cfg.data.train_bs
        * accelerator.num_processes
        * cfg.solver.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {cfg.data.train_bs}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(
        f"  Gradient Accumulation steps = {cfg.solver.gradient_accumulation_steps}"
    )
    logger.info(f"  Total optimization steps = {cfg.solver.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if cfg.resume_from_checkpoint:
        if cfg.resume_from_checkpoint != "latest":
            resume_dir = cfg.resume_from_checkpoint
        else:
            resume_dir = save_dir
        # Get the most recent checkpoint
        dirs = os.listdir(resume_dir)
        dirs = [d for d in dirs if d.startswith("checkpoint")]
        dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
        path = dirs[-1]
        accelerator.load_state(os.path.join(resume_dir, path))
        accelerator.print(f"Resuming from checkpoint {path}")
        global_step = int(path.split("-")[1])

        first_epoch = global_step // num_update_steps_per_epoch
        resume_step = global_step % num_update_steps_per_epoch

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(
        range(global_step, cfg.solver.max_train_steps),
        disable=not accelerator.is_local_main_process,
    )
    progress_bar.set_description("Steps")

    for epoch in range(first_epoch, num_train_epochs):
        train_loss = 0.0
        t_data_start = time.time()
        for step, batch in enumerate(train_dataloader):
            t_data = time.time() - t_data_start
            with accelerator.accumulate(net):
                # Convert videos to latent space
                pixel_values_vid = batch["pixel_values_vid"].to(weight_dtype)
                with torch.no_grad():
                    video_length = pixel_values_vid.shape[1]
                    pixel_values_vid = rearrange(
                        pixel_values_vid, "b f c h w -> (b f) c h w"
                    )
                    latents = vae.encode(pixel_values_vid).latent_dist.sample()
                    latents = rearrange(
                        latents, "(b f) c h w -> b c f h w", f=video_length
                    )
                    latents = latents * 0.18215

                noise = torch.randn_like(latents)
                if cfg.noise_offset > 0:
                    noise += cfg.noise_offset * torch.randn(
                        (latents.shape[0], latents.shape[1], 1, 1, 1),
                        device=latents.device,
                    )
                bsz = latents.shape[0]
                # Sample a random timestep for each video
                timesteps = torch.randint(
                    0,
                    train_noise_scheduler.num_train_timesteps,
                    (bsz,),
                    device=latents.device,
                )
                timesteps = timesteps.long()

                pixel_values_pose = batch["pixel_values_pose"]  # (bs, f, c, H, W)
                pixel_values_pose = pixel_values_pose.transpose(1, 2)  # (bs, c, f, H, W)
                
                pixel_values_ref_pose = batch["pixel_values_ref_pose"]  # (bs, c, H, W)
                pixel_values_ref_pose = pixel_values_ref_pose.unsqueeze(dim=2) # (bs, c, 1, H, W)
    
                
                instance_bboxes = batch["instance_bboxes"]
                for index in range(len(instance_bboxes)): instance_bboxes[index] = instance_bboxes[index].unsqueeze(2)
                #print(instance_bboxes[0].shape) # [1, 1, 1, 384, 640]
                uncond_fwd = random.random() < cfg.uncond_ratio
                clip_image_list = []
                ref_image_list = []
                for batch_idx, (ref_img, clip_img) in enumerate(
                    zip(
                        batch["pixel_values_ref_img"],
                        batch["clip_ref_img"],
                    )
                ):
                    if uncond_fwd:
                        clip_image_list.append(torch.zeros_like(clip_img))
                    else:
                        clip_image_list.append(clip_img)
                    ref_image_list.append(ref_img)

                with torch.no_grad():
                    ref_img = torch.stack(ref_image_list, dim=0).to(
                        dtype=vae.dtype, device=vae.device
                    )
                    ref_image_latents = vae.encode(
                        ref_img
                    ).latent_dist.sample()  # (bs, d, 64, 64)
                    ref_image_latents = ref_image_latents * 0.18215

                    clip_img = torch.stack(clip_image_list, dim=0).to(
                        dtype=image_enc.dtype, device=image_enc.device
                    )
                    clip_img = clip_img.to(device="cuda", dtype=weight_dtype)
                    clip_image_embeds = image_enc(
                        clip_img.to("cuda", dtype=weight_dtype)
                    ).image_embeds
                    clip_image_embeds = clip_image_embeds.unsqueeze(1)  # (bs, 1, d)

                # add noise
                noisy_latents = train_noise_scheduler.add_noise(
                    latents, noise, timesteps
                )

                # Get the target for loss depending on the prediction type
                if train_noise_scheduler.prediction_type == "epsilon":
                    target = noise
                elif train_noise_scheduler.prediction_type == "v_prediction":
                    target = train_noise_scheduler.get_velocity(
                        latents, noise, timesteps
                    )
                else:
                    raise ValueError(
                        f"Unknown prediction type {train_noise_scheduler.prediction_type}"
                    )
                ## audio embedding
                audio_embedding = batch['audio_embedding']
                activate_score_list = batch['activate_scores']
                attention_mask_list = batch['attention_mask']
                face_mask_list = batch['face_mask']
                
                ## 
                # ---- Forward!!! -----
                model_pred = net(
                    noisy_latents,
                    timesteps,
                    ref_image_latents,
                    clip_image_embeds,
                    pixel_values_pose,
                    ref_pose_img=pixel_values_ref_pose,
                    uncond_fwd=uncond_fwd,
                    instance_bboxes=instance_bboxes,
                    
                    audio_embedding=audio_embedding,
                    face_mask_list=face_mask_list,
                    activate_score_list=activate_score_list,
                )

                if cfg.snr_gamma == 0:
                    loss = F.mse_loss(
                        model_pred.float(), target.float(), reduction="mean"
                    )
                else:
                    snr = compute_snr(train_noise_scheduler, timesteps)
                    if train_noise_scheduler.config.prediction_type == "v_prediction":
                        # Velocity objective requires that we add one to SNR values before we divide by them.
                        snr = snr + 1
                    mse_loss_weights = (
                        torch.stack(
                            [snr, cfg.snr_gamma * torch.ones_like(timesteps)], dim=1
                        ).min(dim=1)[0]
                        / snr
                    )
                    loss = F.mse_loss(
                        model_pred.float(), target.float(), reduction="none"
                    )
                    loss = (
                        loss.mean(dim=list(range(1, len(loss.shape))))
                        * mse_loss_weights
                    )
                    loss = loss.mean()

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(cfg.data.train_bs)).mean()
                train_loss += avg_loss.item() / cfg.solver.gradient_accumulation_steps
                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        trainable_params,
                        cfg.solver.max_grad_norm,
                    )
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                reference_control_reader.clear()
                reference_control_writer.clear()
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                if global_step % cfg.val.validation_steps == 0:
                    if accelerator.is_main_process:
                        generator = torch.Generator(device=accelerator.device)
                        generator.manual_seed(cfg.seed)

                        sample_dicts = log_validation(
                            vae=vae,
                            image_enc=image_enc,
                            net=net,
                            scheduler=val_noise_scheduler,
                            accelerator=accelerator,
                            width=cfg.data.train_width,
                            height=cfg.data.train_height,
                            clip_length=cfg.data.n_sample_frames,
                            generator=generator,
                        )

                        for sample_id, sample_dict in enumerate(sample_dicts):
                            sample_name = sample_dict["name"]
                            vid = sample_dict["vid"]
                            with TemporaryDirectory() as temp_dir:
                                out_file = Path(
                                    f"{temp_dir}/{global_step:06d}-{sample_name}.gif"
                                )
                                save_videos_grid(vid, out_file, n_rows=2, fps=25)
                                mlflow.log_artifact(out_file)

            logs = {
                "step_loss": loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
                "td": f"{t_data:.2f}s",
            }
            t_data_start = time.time()
            progress_bar.set_postfix(**logs)

            if global_step >= cfg.solver.max_train_steps:
                break
        # save model after each epoch
        if accelerator.is_main_process:
            save_path = os.path.join(save_dir, f"checkpoint-{global_step}")
            delete_additional_ckpt(save_dir, 1)
            accelerator.save_state(save_path)
            # save motion module only
            unwrap_net = accelerator.unwrap_model(net)
            save_checkpoint(
                unwrap_net.denoising_unet,
                save_dir,
                "denoising_unet",
                global_step,
                total_limit=3,
            )

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    accelerator.end_training()


def save_checkpoint(model, save_dir, prefix, ckpt_num, total_limit=None):
    save_path = osp.join(save_dir, f"{prefix}-{ckpt_num}.pth")

    if total_limit is not None:
        checkpoints = os.listdir(save_dir)
        checkpoints = [d for d in checkpoints if d.startswith(prefix)]
        checkpoints = sorted(
            checkpoints, key=lambda x: int(x.split("-")[1].split(".")[0])
        )

        if len(checkpoints) >= total_limit:
            num_to_remove = len(checkpoints) - total_limit + 1
            removing_checkpoints = checkpoints[0:num_to_remove]
            logger.info(
                f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
            )
            logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

            for removing_checkpoint in removing_checkpoints:
                removing_checkpoint = os.path.join(save_dir, removing_checkpoint)
                os.remove(removing_checkpoint)

    #mm_state_dict = OrderedDict()
    state_dict = model.state_dict()
    # for key in state_dict:
    #     if "motion_module" in key:
    #         mm_state_dict[key] = state_dict[key]
    torch.save(state_dict, save_path)


def decode_latents(vae, latents):
    video_length = latents.shape[2]
    latents = 1 / 0.18215 * latents
    latents = rearrange(latents, "b c f h w -> (b f) c h w")
    # video = self.vae.decode(latents).sample
    video = []
    for frame_idx in tqdm(range(latents.shape[0])):
        video.append(vae.decode(latents[frame_idx : frame_idx + 1]).sample)
    video = torch.cat(video)
    video = rearrange(video, "(b f) c h w -> b c f h w", f=video_length)
    video = (video / 2 + 0.5).clamp(0, 1)
    # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
    video = video.cpu().float().numpy()
    return video


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/training/stage2.yaml")
    args = parser.parse_args()

    if args.config[-5:] == ".yaml":
        config = OmegaConf.load(args.config)
    elif args.config[-3:] == ".py":
        config = import_filename(args.config).cfg
    else:
        raise ValueError("Do not support this format config file")
    main(config)
