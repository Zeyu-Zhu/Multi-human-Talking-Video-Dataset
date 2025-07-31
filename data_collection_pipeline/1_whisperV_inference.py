import os
import sys
import pdb
import glob
import torch
import pickle
import warnings
import argparse
import subprocess
import multiprocessing
import numpy as np
from os import path
from tqdm import tqdm
from functools import partial
warnings.filterwarnings("ignore")
sys.path.append("whisperV/inference_folder")
from demoTalkNet import scene_detect_param, inference_video_param, track_shot_param, crop_video_param, evaluate_network_param, visualization_param


TalkSet_model_path = "whisperV/inference_folder/pretrain_TalkSet.model"


def seg_speaker(data, args):
    file_name, device = data[0], data[1]
    torch.cuda.set_device(device) 
    print(file_name)
    ## create save path
    # mkdir for each data
    raw_video_path = path.join(args.raw_data_path, file_name)
    save_root_path = path.join(args.save_root_path, file_name.split('.')[0])
    # if os.path.isdir(save_root_path):
    #     return
    # else:
    os.mkdir(save_root_path)
    # mkdir for each element
    avi_path = path.join(save_root_path, 'avi')
    crop_path = path.join(save_root_path, 'crop')
    frame_path = path.join(save_root_path, 'frame')
    result_path = path.join(save_root_path, 'result')
    os.mkdir(avi_path)
    os.mkdir(crop_path)
    os.mkdir(frame_path)
    os.mkdir(result_path)
    ## preprocess raw video and save files
    video_save_path, audio_save_path = path.join(avi_path, 'video.avi'),  path.join(avi_path, 'audio.wav')
    command_video = ("ffmpeg -y -i %s -qscale:v 2 -threads %d -async 1 -r 25 %s -loglevel panic" % (raw_video_path, 10, video_save_path))
    command_audio = ("ffmpeg -y -i %s -qscale:a 0 -ac 1 -vn -threads %d -ar 16000 %s -loglevel panic" % (video_save_path, 10, audio_save_path))
    command_frame = ("ffmpeg -y -i %s -qscale:v 2 -threads %d -f image2 %s -loglevel panic" % (video_save_path, 10, path.join(frame_path, '%06d.jpg')))
    subprocess.call(command_video, shell=True, stdout=None)
    subprocess.call(command_audio, shell=True, stdout=None)
    subprocess.call(command_frame, shell=True, stdout=None)
    ## detect and segment
    scene = scene_detect_param(video_save_path, result_path)
    faces = inference_video_param(video_save_path, result_path, frame_path)
    # face tracking
    all_tracks, vild_tracks = [], []
    for shot in scene:
        if shot[1].frame_num - shot[0].frame_num >= args.min_frame_shot:
            track = track_shot_param(args.min_failed_detect, args.min_frame_shot, args.min_face_size, faces[shot[0].frame_num:shot[1].frame_num])
            all_tracks.extend(track)
    for ii, track in tqdm(enumerate(all_tracks), total = len(all_tracks)):
        crop_track = crop_video_param(frame_path, audio_save_path, track, os.path.join(crop_path, '%05d'%ii))
        vild_tracks.append(crop_track)
    tracks_save_path = os.path.join(result_path, 'tracks.pckl')
    with open(tracks_save_path, 'wb') as fil:
        pickle.dump(vild_tracks, fil)
    # speak detect
    files = glob.glob("%s/*.avi"%crop_path)
    files.sort()
    scores = evaluate_network_param(files, TalkSet_model_path, crop_path)
    smooth_scores_list = []
    # smooth the score
    for tidx, track in enumerate(vild_tracks):
        score = scores[tidx]
        smooth_scores = []
        for fidx, frame in enumerate(track['track']['frame'].tolist()):
            s = score[max(fidx - 3, 0): min(fidx + 3, len(score) - 1)] # average smoothing
            s = np.mean(s)
            smooth_scores.append(s)
        smooth_scores_list.append(smooth_scores)
    score_save_path = os.path.join(result_path, 'smooth_scores.pckl')
    with open(score_save_path, 'wb') as fil:
        pickle.dump(smooth_scores_list, fil)
    visualization_param(vild_tracks, scores, frame_path, avi_path)
 
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "speaker segmentation")
    parser.add_argument('--min_face_size', type=int, default=16, help='min face size in pixels')
    parser.add_argument('--min_frame_shot', type=int, default=12, help='min frame num for one shot')
    parser.add_argument('--min_failed_detect', type=int, default=12, help='num of missed detections allowed before tracking is stopped')
    parser.add_argument('--raw_data_path', type=str, default="", help='raw video data path')
    parser.add_argument('--save_root_path', type=str, default="", help='result save root path')
    args = parser.parse_args()

    file_name_list = [int(f.split('.')[0]) for f in os.listdir(args.raw_data_path)]
    file_name_list = sorted(file_name_list)
    file_name_list = [str(f)+'.mp4' for f in file_name_list]
    GPU_List = [0, 1, 2, 3, 4, 5, 6, 7]
    device_list = ['cuda:'+str(GPU_List[index%len(GPU_List)]) for index in range(len(file_name_list))]
    all_data = [[file_name_list[index], device_list[index]] for index in range(len(file_name_list))]
    pdb.set_trace()
    
    num_processes = 7
    ctx = multiprocessing.get_context('spawn')
    with ctx.Pool(processes=num_processes) as pool:
        func = partial(seg_speaker, args=args)
        results = list(tqdm(pool.imap(func, all_data), total=len(all_data)))