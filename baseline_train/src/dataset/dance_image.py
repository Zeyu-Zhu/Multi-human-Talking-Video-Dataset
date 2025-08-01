import os
import json
import random
import numpy as np
from os import path 

import torch
import torchvision.transforms as transforms
from decord import VideoReader
from PIL import Image
from torch.utils.data import Dataset
from transformers import CLIPImageProcessor


class HumanDanceDataset(Dataset):
    def __init__(
        self,
        img_size,
        img_scale=(1.0, 1.0),
        img_ratio=(0.9, 1.0),
        drop_ratio=0.1,
        data_meta_paths=["./data/fahsion_meta.json"],
        sample_margin=30,
    ):
        super().__init__()

        self.img_size = img_size
        self.img_scale = img_scale
        self.img_ratio = img_ratio
        self.sample_margin = sample_margin

        # -----
        # vid_meta format:
        # [{'video_path': , 'kps_path': , 'other':},
        #  {'video_path': , 'kps_path': , 'other':}]
        # -----
        vid_meta = []
        for data_meta_path in data_meta_paths:
            vid_meta.extend(json.load(open(data_meta_path, "r")))
        self.vid_meta = vid_meta

        self.clip_image_processor = CLIPImageProcessor()

        self.transform = transforms.Compose(
            [
                # transforms.RandomResizedCrop(
                #     self.img_size,
                #     scale=self.img_scale,
                #     ratio=self.img_ratio,
                #     interpolation=transforms.InterpolationMode.BILINEAR,
                # ),
                transforms.Resize(self.img_size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        self.cond_transform = transforms.Compose(
            [
                # transforms.RandomResizedCrop(
                #     self.img_size,
                #     scale=self.img_scale,
                #     ratio=self.img_ratio,
                #     interpolation=transforms.InterpolationMode.BILINEAR,
                # ),
                transforms.Resize(self.img_size),
                transforms.ToTensor(),
            ]
        )

        self.drop_ratio = drop_ratio

    def augmentation(self, image, transform, state=None):
        if state is not None:
            torch.set_rng_state(state)
        return transform(image)

    def get_mask_from_bbox(self, bbox):
        attention_mask_2d = np.zeros((1080, 1920), dtype=np.uint8)
        x_1, y_1, x_2, y_2 = bbox
        x_1, y_1, x_2, y_2 = int(x_1), int(y_1), int(x_2), int(y_2)
        x_1 = max(0, x_1)
        y_1 = max(0, y_1)
        x_2 = min(1920, x_2)
        y_2 = max(1080, y_2)
        attention_mask_2d[y_1:y_2, x_1:x_2] = 255
        return Image.fromarray(attention_mask_2d)
    
    def __getitem__(self, index):        
        while True:
            try:
                video_meta = self.vid_meta[index]
                img_path = video_meta["image_path"]
                kps_path = video_meta["pose_path"].replace('sparse_pose_frame', 'pose_frame')
                speaker_info_path = path.join(path.dirname(img_path), 'speaker_info_with_instance_bbox.json')
            
                instance_bboxes = []
                with open(speaker_info_path, 'r') as f: infos = json.load(f)
                for info in infos: 
                    instance_bboxes.append(self.get_mask_from_bbox(info['instance_bbox']))
            
                img_list = [path.join(img_path, name) for name in os.listdir(img_path)]
                pose_list = [path.join(kps_path, name) for name in os.listdir(kps_path)]
                if len(img_list) == len(pose_list): break
                else: index = random.randint(0, self.__len__())
            except:
                index = random.randint(0, self.__len__())
                
        video_length = len(img_list)
        margin = min(self.sample_margin, video_length)  
        ## ref image index
        ref_img_idx = random.randint(0, video_length - 1)
        ## tgt image index
        if ref_img_idx + margin < video_length: tgt_img_idx = random.randint(ref_img_idx + margin, video_length - 1)
        elif ref_img_idx - margin > 0: tgt_img_idx = random.randint(0, ref_img_idx - margin)
        else: tgt_img_idx = random.randint(0, video_length - 1)
        ## ref img
        ref_img = img_list[ref_img_idx]
        ref_img_pil = Image.open(ref_img)
        ## ref pose
        ref_pose = pose_list[ref_img_idx]
        ref_pose_pil = Image.open(ref_pose)
        ## tgt img
        tgt_img = img_list[tgt_img_idx]
        tgt_img_pil = Image.open(tgt_img)
        ## tgt pose
        tgt_pose = pose_list[tgt_img_idx]
        tgt_pose_pil = Image.open(tgt_pose)

        state = torch.get_rng_state()
        ref_bboxes = []
        for instance_bbox in instance_bboxes:
            ref_bboxes.append(self.augmentation(instance_bbox, self.cond_transform, state))
        ref_bboxes = [ref_bboxes]
        
        tgt_img = self.augmentation(tgt_img_pil, self.transform, state)
        ref_img_vae = self.augmentation(ref_img_pil, self.transform, state)
        ref_pose_img = self.augmentation(ref_pose_pil, self.cond_transform, state)
        tgt_pose_img = self.augmentation(tgt_pose_pil, self.cond_transform, state)
        clip_image = self.clip_image_processor(images=ref_img_pil, return_tensors="pt").pixel_values[0]

        sample = dict(
            # video_dir=video_path,
            instance_bboxes=ref_bboxes,
            img=tgt_img,
            ref_pose=ref_pose_img,
            tgt_pose=tgt_pose_img,
            ref_img=ref_img_vae,
            clip_images=clip_image,
        )

        return sample

    def __len__(self):
        return len(self.vid_meta)


if __name__ == '__main__':
    datasets = HumanDanceDataset(img_size=(1080, 1920), 
                                 data_meta_paths=["/users/zeyuzhu/ControlSD/Moore-AnimateAnyone/data_config/stage_1.json"])
    sample = datasets[0]
    for key in sample.keys():
        print(key, sample[key].shape)
    img = (sample['img']+1)/2
    
    from torchvision.transforms import ToPILImage
    to_pil = ToPILImage()  # torchvision 提供的转换器
    image = to_pil(img)
    image.save("output_image.png")