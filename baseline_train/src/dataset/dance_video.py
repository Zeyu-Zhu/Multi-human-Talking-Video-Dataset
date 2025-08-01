import json
import random
from typing import List

import os
from os import path
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from decord import VideoReader
from PIL import Image
from torch.utils.data import Dataset
from transformers import CLIPImageProcessor


class HumanDanceVideoDataset(Dataset):
    def __init__(
        self,
        sample_rate,
        n_sample_frames,
        width,
        height,
        img_scale=(1.0, 1.0),
        img_ratio=(0.9, 1.0),
        drop_ratio=0.1,
        data_meta_paths=["./data/fashion_meta.json"],
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_sample_frames = n_sample_frames
        self.width = width
        self.height = height
        self.img_scale = img_scale
        self.img_ratio = img_ratio
        self.sample_size = (height, width)
        
        vid_meta = []
        for data_meta_path in data_meta_paths:
            vid_meta.extend(json.load(open(data_meta_path, "r")))
        self.vid_meta = vid_meta

        self.clip_image_processor = CLIPImageProcessor()

        self.pixel_transform = transforms.Compose(
            [
                transforms.Resize(self.sample_size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        self.cond_transform = transforms.Compose(
            [
                transforms.Resize(self.sample_size),
                transforms.ToTensor(),
            ]
        )

        self.drop_ratio = drop_ratio

    def augmentation(self, images, transform, state=None):
        if state is not None:
            torch.set_rng_state(state)
        if isinstance(images, List):
            transformed_images = [transform(img) for img in images]
            ret_tensor = torch.stack(transformed_images, dim=0)  # (f, c, h, w)
        else:
            ret_tensor = transform(images)  # (c, h, w)
        return ret_tensor

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
                for info in infos: instance_bboxes.append(self.get_mask_from_bbox(info['instance_bbox']))
            
                img_list = sorted([path.join(img_path, name) for name in os.listdir(img_path)])
                pose_list = sorted([path.join(kps_path, name) for name in os.listdir(kps_path)])
                if len(img_list) == len(pose_list): break
                else: index = random.randint(0, self.__len__())
            except:
                index = random.randint(0, self.__len__())
                
        video_length = len(img_list)

        clip_length = min(video_length, (self.n_sample_frames - 1) * self.sample_rate + 1)
        start_idx = random.randint(0, video_length - clip_length)
        batch_index = np.linspace(start_idx, start_idx + clip_length - 1, self.n_sample_frames, dtype=int).tolist()

        # read frames and kps
        vid_pil_image_list = []
        pose_pil_image_list = []
        for index in batch_index:
            img = img_list[index]
            vid_pil_image_list.append(Image.open(img))
            img = pose_list[index]
            pose_pil_image_list.append(Image.open(img))

    
        state = torch.get_rng_state()
        ## ref image
        ref_img_idx = random.randint(0, video_length - 1)
        ref_img = Image.open(img_list[ref_img_idx])
        ## ref pose
        ref_pose_pil = Image.open(pose_list[ref_img_idx])
        pixel_values_ref_pose = self.augmentation(ref_pose_pil, self.cond_transform, state)
        
        # transform
        ref_bboxes = []
        for instance_bbox in instance_bboxes:
            ref_bboxes.append(self.augmentation(instance_bbox, self.cond_transform, state))
            
        pixel_values_vid = self.augmentation(vid_pil_image_list, self.pixel_transform, state)
        pixel_values_pose = self.augmentation(pose_pil_image_list, self.cond_transform, state)
        pixel_values_ref_img = self.augmentation(ref_img, self.pixel_transform, state)
        clip_ref_img = self.clip_image_processor(images=ref_img, return_tensors="pt").pixel_values[0]

        sample = dict(
            #video_dir=video_path,
            instance_bboxes=ref_bboxes,
            pixel_values_vid=pixel_values_vid,
            pixel_values_pose=pixel_values_pose,
            pixel_values_ref_pose=pixel_values_ref_pose,
            pixel_values_ref_img=pixel_values_ref_img,
            clip_ref_img=clip_ref_img,
        )
        
        return sample

    def __len__(self):
        return len(self.vid_meta)

if __name__ == '__main__':
    dataset = HumanDanceVideoDataset(width=640, height=384, 
                           sample_rate=4, n_sample_frames=16,
                           data_meta_paths=["/users/zeyuzhu/ControlSD/Moore-AnimateAnyone/data_config/stage_1.json"],)
    sample = dataset[0]
    video = (sample['pixel_values_vid'].transpose(1, 2).transpose(2, 3)+1)*255
    print(video.shape)
    import torchvision.io as io
    io.write_video('test.mp4', video, fps=5, video_codec="libx264")