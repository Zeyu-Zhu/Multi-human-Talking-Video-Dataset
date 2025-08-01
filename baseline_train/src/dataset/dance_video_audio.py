import json
import random
from typing import List
import torch.nn.functional as F

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


def bbox_intersection(bbox1, bbox2):
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2
    
    inter_x1 = max(x1_1, x1_2)
    inter_y1 = max(y1_1, y1_2)
    inter_x2 = min(x2_1, x2_2)
    inter_y2 = min(y2_1, y2_2)
    
    if inter_x1 < inter_x2 and inter_y1 < inter_y2:
        return int(inter_x1), int(inter_y1), int(inter_x2), int(inter_y2)
    else:
        return None
    
def keypoints_to_bbox(keypoints):
    keypoints = np.array(keypoints)
    x_min = np.min(keypoints[:, 0])
    x_max = np.max(keypoints[:, 0])
    y_min = np.min(keypoints[:, 1])
    y_max = np.max(keypoints[:, 1])
    return int(x_min), int(y_min), int(x_max), int(y_max)

def scale_bbox(bbox, scale_x, scale_y=None):
    if scale_y is None:
        scale_y = scale_x
    x_min, y_min, x_max, y_max = bbox
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    width = x_max - x_min
    height = y_max - y_min
    new_width = width * scale_x
    new_height = height * scale_y
    new_x_min = max(x_center - new_width / 2, 0)
    new_x_max = x_center + new_width / 2
    new_y_min = max(y_center - new_height / 2, 0)
    new_y_max = y_center + new_height / 2
    return int(new_x_min), int(new_y_min), int(new_x_max), int(new_y_max)

def calculate_square_bbox(top_midpoint, bottom_midpoint):
    x1, y1 = top_midpoint
    x2, y2 = bottom_midpoint
    height = y2 - y1
    width = height  
    x_min = x1 - width / 2
    y_min = y1
    x_max = x1 + width / 2
    y_max = y2
    return int(x_min), int(y_min), int(x_max), int(y_max)

def generate_attention_mask(height, width, bbox):
    xmin, ymin, xmax, ymax = bbox
    attention_mask_2d = np.zeros((height, width), dtype=int)
    attention_mask_2d[ymin:ymax+1, xmin:xmax+1] = 1
    attention_mask = attention_mask_2d
    # attention_mask = Image.fromarray(attention_mask_2d)
    return attention_mask


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
        
        self.mask_transform = [transforms.Compose([transforms.Resize((self.sample_size[0]//8, self.sample_size[1]//8)), transforms.ToTensor(),]),
                               transforms.Compose([transforms.Resize((self.sample_size[0]//16, self.sample_size[1]//16)), transforms.ToTensor(),]),
                               transforms.Compose([transforms.Resize((self.sample_size[0]//32, self.sample_size[1]//32)), transforms.ToTensor(),]),
                               transforms.Compose([transforms.Resize((self.sample_size[0]//64, self.sample_size[1]//64)), transforms.ToTensor(),]),
                               transforms.Compose([transforms.Resize((self.sample_size[0]//64, self.sample_size[1]//64)), transforms.ToTensor(),])]
        self.drop_ratio = drop_ratio

    def activate_score_normalize(self, scores):
        scores = torch.Tensor(scores)
        for index in range(len(scores)):
            scores[index] = scores[index]/3
        return scores
    
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
        y_2 = min(1080, y_2)
        attention_mask_2d[y_1:y_2, x_1:x_2] = 255
        return Image.fromarray(attention_mask_2d)
    
    def count_keypoints_in_bbox(self, pose, bbox):
        keypoints = pose['keypoints']
        scores = pose['keypoint_scores']
        xmin, ymin, xmax, ymax = bbox
        in_bbox_count = 0
        for idx, (x, y) in enumerate(keypoints):
            score = scores[idx]
            if score<0.4: continue
            if xmin <= x <= xmax and ymin <= y <= ymax:
                in_bbox_count += 1
        return in_bbox_count

    def generate_masks(self, frame_pose_list):
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
                
                per_person_full_mask.append(self.get_mask_from_bbox(full_bbox))
                per_person_face_mask.append(self.get_mask_from_bbox(face_bbox))
                per_person_lip_mask.append(self.get_mask_from_bbox(lip_bbox))
                ## attention mask
                attention_mask_bbox = scale_bbox(full_bbox, 1.2)
                attention_mask = self.get_mask_from_bbox(attention_mask_bbox)
                per_person_attention_mask.append(attention_mask)
            full_mask_list.append(per_person_full_mask)
            face_mask_list.append(per_person_face_mask)
            lip_mask_list.append(per_person_lip_mask)
            attention_mask_list.append(per_person_attention_mask)     
        return attention_mask_list, full_mask_list, face_mask_list, lip_mask_list
    
    def get_human_info(self, speaker_info_path, pose_json_dir, batch_index):
        clip_length = len(batch_index)
        ## masks: need to align with the human(check out if keypoints lie in instance bbox)
        with open(speaker_info_path, 'r') as f: infos = json.load(f)
        pose_name_list = sorted([pose_name for pose_name in os.listdir(pose_json_dir) if pose_name.endswith('.json')])
        pose_name_list = [pose_name_list[i] for i in batch_index]

        frame_pose_list = []
        for pose_name in pose_name_list: ## along frame
            with open(path.join(pose_json_dir, pose_name), 'r') as f: pose_list = json.load(f)['instance_info']
            pose_in_order = []
            for info in infos: ## along person
                instance_bbox, within_bbox_count = info['instance_bbox'], []
                for pose in pose_list: ## within 2 pose
                    within_bbox_count.append(self.count_keypoints_in_bbox(pose, instance_bbox))
                match_index = np.argmax(within_bbox_count)
                pose_in_order.append(pose_list[match_index]['keypoints'])
            frame_pose_list.append(pose_in_order)
        frame_pose_list = np.array(frame_pose_list).transpose(1, 0, 2, 3) # (huamn, frame, 133, 2)
        #get the mask: attention mask for each person, full mask, face mask, lip mask
        attention_mask, full_mask, face_mask, lip_mask = self.generate_masks(frame_pose_list)
        # attention_mask, full_mask, face_mask, lip_mask = torch.zeros(1), torch.zeros(1), torch.zeros(1), torch.zeros(1)
        mask = dict(attention_mask=attention_mask, 
                    full_mask=full_mask, 
                    face_mask=face_mask, 
                    lip_mask=lip_mask)
    
        ## activate scores(already aligned with instance_bbox)
        activate_scores = []

        for info in infos:
            activate_score = [info['speaking_score'][i] for i in batch_index] 
            activate_scores.append(activate_score)

        activate_scores = self.activate_score_normalize(activate_scores)    
        activate_scores = activate_scores.unsqueeze(dim=-1)
        return activate_scores, mask
    
    def __getitem__(self, index):
        while True:
            try:
                video_meta = self.vid_meta[index]
                img_path = video_meta["image_path"]
                kps_path = video_meta["pose_path"]
                audio_embedding_path = path.join(path.dirname(img_path), 'audio_embeding.pt')
                speaker_info_path = path.join(path.dirname(img_path), 'speaker_info_with_instance_bbox.json')
                ## audio embedding
                audio_embedding = torch.load(audio_embedding_path)
                ## instance bboxes
                instance_bboxes = []
                with open(speaker_info_path, 'r') as f: infos = json.load(f)
                for info in infos: 
                    instance_bboxes.append(self.get_mask_from_bbox(info['instance_bbox']))
                ## frames and poses
                img_list = sorted([path.join(img_path, name) for name in os.listdir(img_path)])
                pose_list = sorted([path.join(kps_path, name) for name in os.listdir(kps_path)])
                video_length = len(img_list)

                sample_rate = random.randint(1, self.sample_rate)
                clip_length = min(video_length, (self.n_sample_frames - 1) * sample_rate + 1)
                start_idx = random.randint(0, video_length - clip_length)
                batch_index = np.linspace(start_idx, start_idx + clip_length - 1, self.n_sample_frames, dtype=int).tolist()
                state = torch.get_rng_state()
                ## audio_embedding
                
                audio_embedding = audio_embedding[batch_index, :, :]
                
                ## tgt frames and pose
                vid_pil_image_list = []
                pose_pil_image_list = []
                for index in batch_index:
                    img = img_list[index]
                    vid_pil_image_list.append(Image.open(img))
                    img = pose_list[index]
                    pose_pil_image_list.append(Image.open(img))
                ## ref image
                ref_img_idx = random.randint(0, video_length - 1)
                ref_img = Image.open(img_list[ref_img_idx])
                ## ref pose
                ref_pose_pil = Image.open(pose_list[ref_img_idx])
                pixel_values_ref_pose = self.augmentation(ref_pose_pil, self.cond_transform, state)
                ## transform
                ref_bboxes = []
                for instance_bbox in instance_bboxes:
                    ref_bboxes.append(self.augmentation(instance_bbox, self.cond_transform, state))
                pixel_values_vid = self.augmentation(vid_pil_image_list, self.pixel_transform, state)
                pixel_values_pose = self.augmentation(pose_pil_image_list, self.cond_transform, state)
                pixel_values_ref_img = self.augmentation(ref_img, self.pixel_transform, state)
                clip_ref_img = self.clip_image_processor(images=ref_img, return_tensors="pt").pixel_values[0]
                ## human level variable: activate score, multi-level mask
                pose_json_dir = img_path.replace('frames', 'pose')
                activate_scores, mask = self.get_human_info(speaker_info_path, pose_json_dir, batch_index)
                # mask list
                face_mask_list = mask['face_mask']
                attention_mask_list = mask['attention_mask']
                person_face_mask_level_list, person_attention_mask_level_list = [], []
                for huamn_index in range(len(face_mask_list)):
                    face_mask_level_list = []
                    attention_mask_level_list = []
                    for mask_transform_level in self.mask_transform:
                        face_mask_level_list.append(self.augmentation(face_mask_list[huamn_index], mask_transform_level, state))
                        attention_mask_level_list.append(self.augmentation(attention_mask_list[huamn_index], mask_transform_level, state))
                    person_face_mask_level_list.append(face_mask_level_list)
                    person_attention_mask_level_list.append(attention_mask_level_list)
                if len(img_list) == len(pose_list): break
                else: index = random.randint(0, self.__len__())
            except:
                index = random.randint(0, self.__len__())
            
        sample = dict(
            #video_dir=video_path,
            instance_bboxes=ref_bboxes,
            pixel_values_vid=pixel_values_vid,
            pixel_values_pose=pixel_values_pose,
            pixel_values_ref_pose=pixel_values_ref_pose,
            pixel_values_ref_img=pixel_values_ref_img,
            clip_ref_img=clip_ref_img,
            
            audio_embedding=audio_embedding,
            activate_scores=activate_scores,
            face_mask=person_face_mask_level_list,
            attention_mask=person_attention_mask_level_list,
        )
        return sample

    def __len__(self):
        return len(self.vid_meta)

if __name__ == '__main__':
    import cv2
    dataset = HumanDanceVideoDataset(width=640, height=384, 
                           sample_rate=4, n_sample_frames=16,
                           data_meta_paths=["/users/zeyuzhu/ControlSD/Moore-AnimateAnyone/data_config/stage_1.json"],)
    sample = dataset[0]
    #video = (sample['mask']['face_mask'].transpose(1, 2).transpose(2, 3))*255
    face_mask_1 = sample['face_mask'][0][1]
    video = torch.Tensor(face_mask_1).transpose(2, 1).transpose(3, 2).repeat(1, 1, 1, 3)*255
    cv2.imwrite('test_1.png', video[1].numpy()*255)
    
    face_mask_2 = sample['face_mask'][1][1]
    video = torch.Tensor(face_mask_2).transpose(2, 1).transpose(3, 2).repeat(1, 1, 1, 3)*255
    cv2.imwrite('test_2.png', video[1].numpy()*255)
    
    import torchvision.io as io
    io.write_video('test.mp4', video, fps=5, video_codec="libx264")
    # audio_embedding: frame, bin, 768
    # activate_scores: num_person, frame, 1
    # mask:
    
