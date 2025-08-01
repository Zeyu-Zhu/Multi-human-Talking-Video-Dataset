import argparse
import os
from datetime import datetime
from pathlib import Path
from typing import List

from os import path
import av
import numpy as np
import torch
import torchvision
from diffusers import AutoencoderKL, DDIMScheduler
from diffusers.pipelines.stable_diffusion import StableDiffusionPipeline
from einops import repeat
from omegaconf import OmegaConf
from PIL import Image
from torchvision import transforms
from transformers import CLIPVisionModelWithProjection

from configs.prompts.test_cases import TestCasesDict
from src.models.pose_guider import PoseGuider
from src.models.unet_2d_condition import UNet2DConditionModel
from src.models.unet_3d_audio import UNet3DConditionModel
from src.pipelines.pipeline_pose2vid_audio_long import Pose2VideoPipeline
from src.utils.util import get_fps, read_frames, save_videos_grid
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

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("-W", type=int, default=640)
    parser.add_argument("-H", type=int, default=384)
    parser.add_argument("-L", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cfg", type=float, default=3.5)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--fps", type=int)
    parser.add_argument("--motion_frame_num", type=int, default=10)
    args = parser.parse_args()

    return args

import json
from tqdm import tqdm

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

def main():
    src_fps = 25
    args = parse_args()

    config = OmegaConf.load(args.config)

    if config.weight_dtype == "fp16":
        weight_dtype = torch.float16
    else:
        weight_dtype = torch.float32

    vae = AutoencoderKL.from_pretrained(
        config.pretrained_vae_path,
    ).to("cuda", dtype=weight_dtype)

    reference_unet = UNet2DConditionModel.from_pretrained(
        config.pretrained_base_model_path,
        subfolder="unet",
    ).to(dtype=weight_dtype, device="cuda")

    inference_config_path = config.inference_config
    infer_config = OmegaConf.load(inference_config_path)
    denoising_unet = UNet3DConditionModel.from_pretrained_2d(
        config.pretrained_base_model_path,
        "",
        subfolder="unet",
        unet_additional_kwargs=infer_config.unet_additional_kwargs,
    ).to(dtype=weight_dtype, device="cuda")

    image_enc = CLIPVisionModelWithProjection.from_pretrained(
        config.image_encoder_path
    ).to(dtype=weight_dtype, device="cuda")
        
    pose_guider = PoseGuider(320, block_out_channels=(16, 32, 96, 256)).to(dtype=weight_dtype, device="cuda")
    pose_adaptor = PoseGuider(320, block_out_channels=(16, 32, 96, 256)).to(dtype=weight_dtype, device="cuda")
    #pose_adaptor = None
    
    sched_kwargs = OmegaConf.to_container(infer_config.noise_scheduler_kwargs)
    scheduler = DDIMScheduler(**sched_kwargs)

    generator = torch.manual_seed(args.seed)

    width, height = args.W, args.H

    # load pretrained weights
    denoising_unet.load_state_dict(
        torch.load(config.denoising_unet_path, map_location="cpu"),
        strict=False,
    )
    weight =  torch.load(config.denoising_unet_path, map_location="cpu")
        
    reference_unet.load_state_dict(
        torch.load(config.reference_unet_path, map_location="cpu"),
    )
    pose_guider.load_state_dict(
        torch.load(config.pose_guider_path, map_location="cpu"),
    )
    pose_adaptor.load_state_dict(
        torch.load(config.pose_guider_path, map_location="cpu"),
    )
    
    pipe = Pose2VideoPipeline(
        vae=vae,
        image_encoder=image_enc,
        reference_unet=reference_unet,
        denoising_unet=denoising_unet,
        pose_guider=pose_guider,
        pose_adaptor=pose_adaptor,
        scheduler=scheduler,
        
        sample_size=(height, width),
    )
    pipe = pipe.to("cuda", dtype=weight_dtype)

    date_str = datetime.now().strftime("%Y%m%d")
    time_str = datetime.now().strftime("%H%M")
    save_dir_name = f"{time_str}--seed_{args.seed}-{args.W}x{args.H}"

    save_dir = Path(f"result_proposed/mask_adaptor_multi_long/{date_str}/{save_dir_name}")
    save_dir.mkdir(exist_ok=True, parents=True)


    for ref_image_path in config["test_cases"].keys():
        # Each ref_image may correspond to multiple actions
        for pose_video_path in config["test_cases"][ref_image_path]:
            result = []
            motion_frame = None
            ref_name = path.dirname(path.dirname(ref_image_path))
            pose_name = Path(pose_video_path).stem.replace("_kps", "")

            ref_image_pil = Image.open(ref_image_path).convert("RGB")

            # pose_list = []
            # pose_tensor_list = []
            # pose_images = read_frames(pose_video_path)
            pose_path_list = [path.join(pose_video_path, name) for name in os.listdir(pose_video_path)]
            pose_path_list = sorted(pose_path_list)
            pose_images = []
            for pose_path in pose_path_list: pose_images.append(Image.open(pose_path))
            print(f"pose video has {len(pose_images)} frames, with {src_fps} fps")
            pose_transform = transforms.Compose([transforms.Resize((height, width)), transforms.ToTensor()])
            ## ref pose(first frame)
            ref_pose_pil = pose_images[0] 
            ## ref image(first frame)
            ref_image_tensor = pose_transform(ref_image_pil)  # (c, h, w)
            ref_image_tensor = ref_image_tensor.unsqueeze(1).unsqueeze(0)  # (1, c, 1, h, w)
            ref_image_tensor = repeat(ref_image_tensor, "b c f h w -> b c (repeat f) h w", repeat=args.L)
            ## instance_bbox
            pose_json_dir = path.dirname(ref_image_path).replace('frames', 'pose')
            speaker_info_path = path.join(path.dirname(path.dirname(ref_image_path)), 'speaker_info_with_instance_bbox.json')
            instance_bboxes = []
            with open(speaker_info_path, 'r') as f: info = json.load(f)
            for info in info: instance_bboxes.append(get_mask_from_bbox(info['instance_bbox']))
            for index in range(len(instance_bboxes)):
                instance_bboxes[index] = torch.from_numpy(np.array(instance_bboxes[index].resize((width, height)))) / 255.0
                instance_bboxes[index] = instance_bboxes[index].unsqueeze(dim=0).unsqueeze(dim=0).unsqueeze(dim=0)
            ## audio embedding
            audio_embeeding_path = path.join(path.dirname(path.dirname(ref_image_path)), 'audio_embeding.pt')
            audio_embedding = torch.load(audio_embeeding_path)
            
            clip_length = args.L
            each_step_frame_num = clip_length-args.motion_frame_num
            for frame_index in tqdm(range(0, len(pose_images), each_step_frame_num)):
                pose_list = []
                pose_tensor_list = []
                
                if frame_index > len(pose_images): break
                if frame_index+clip_length > len(pose_images): 
                    required_frame_num = len(pose_images) - frame_index
                    frame_index = len(pose_images) - clip_length
                else: required_frame_num = each_step_frame_num
                    
                for pose_image_pil in pose_images[frame_index:frame_index+clip_length]:
                    pose_tensor_list.append(pose_transform(pose_image_pil))
                    pose_list.append(pose_image_pil)
                ## tgt pose
                pose_tensor = torch.stack(pose_tensor_list, dim=0)  # (f, c, h, w)
                pose_tensor = pose_tensor.transpose(0, 1)
                pose_tensor = pose_tensor.unsqueeze(0)
                ## audio embedding
                audio_embedding_clip = audio_embedding[frame_index:frame_index+clip_length, :, :]
                ## face mask and activate score
                activate_scores, mask = get_human_info(speaker_info_path=speaker_info_path, pose_json_dir=pose_json_dir,
                                                       start_index=frame_index, clip_length=clip_length,)

                ## inference
                video = pipe(
                    ref_image_pil,
                    pose_list,
                    instance_bboxes,
                    width,
                    height,
                    args.L,
                    args.steps,
                    args.cfg,
                    generator=generator,
                    motion_frame_k=args.motion_frame_num,
                    motion_frames=motion_frame,
                    ref_pose_image=ref_pose_pil,
                    audio_embedding=audio_embedding_clip,
                    face_mask_list=mask['face_mask'],
                    activate_score_list=activate_scores,
                ).videos
                fixed_video = video[:, :, :required_frame_num, ...]
                if motion_frame is not None and len(fixed_video)>=args.motion_frame_num:
                    fixed_video[:, :, :args.motion_frame_num, ...] += motion_frame
                    fixed_video[:, :, :args.motion_frame_num, ...] /= 2 
                if args.motion_frame_num != 0: motion_frame = video[:, :, -args.motion_frame_num:, ...] 
                video = fixed_video
                result.append(video)              
            result = torch.cat(result, dim=2)
            save_videos_grid(
                result,
                f"{save_dir}/{ref_name}_{pose_name}_{args.H}x{args.W}_{int(args.cfg)}_{time_str}.mp4",
                n_rows=3,
                fps=src_fps if args.fps is None else args.fps,)


if __name__ == "__main__":
    main()
# python inference_long.py --config ./configs/prompts/animation_long.yaml -W 640 -H 384 -L 16
# CUDA_VISIBLE_DEVICES=5 python -m scripts.pose2vid --config ./configs/prompts/animation.yaml -W 640 -H 384 -L 25
# python -m pytorch_fid path/to/dataset1 path/to/dataset2
# ffmpeg -i ./output/20250103/motion_num_k=5/output_0001_sparse_pose_frame_384x640_3_1423.mp4 -qscale:v 2 -vf fps=25 ./output/20250103/motion_num_k=5/%04d.png

# animate_anyone 93.49692998042805
# train_from_start 83.67855073964878
# fine_tune 81.27001226308931

# motion_num_k=0 83.02624315725194
# motion_num_k=5 82.81869740101547

#python inference_audio_long.py --config ./configs/prompts/animation_audio.yaml -W 640 -H 384 -L 15
#ffmpeg -i /users/zeyuzhu/ControlSD/Moore-AnimateAnyone/result_proposed/mask_adaptor_multi_long/20250115/1241--seed_42-640x384/users/zeyuzhu/dataset_project/Datasets/latenightshow/datasets/latenightshow_1_002843to003190_sparse_pose_frame_384x640_3_1241.mp4 -i /users/zeyuzhu/dataset_project/Datasets/latenightshow/datasets/latenightshow_1_002843to003190/audio.wav -c:v copy -c:a aac -strict experimental latenightshow_1_002843to003190.mp4
