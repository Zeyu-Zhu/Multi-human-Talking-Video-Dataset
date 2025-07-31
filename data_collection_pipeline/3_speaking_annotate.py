## label the speaking periode for each speaker(defined by his/her head track)
import os
import pdb
import json
import pickle
import argparse
import subprocess
import numpy as np
import multiprocessing
from functools import partial
from tqdm import tqdm
from os import path
from scipy.signal import medfilt

URL_PATH = 'mapping.json' # video index to url link
with open(URL_PATH, 'r') as f: url_dic = json.load(f)
DATA_SOURCE = 'test'
FPS = 25

def process(data, args):
    video_name = data
    origin_video_path = path.join(args.seg_result_root, video_name, 'avi', 'video.avi')
    tracks_result_path = path.join(args.seg_result_root, video_name, 'result', 'tracks.pckl')
    smooth_scores_result_path = path.join(args.seg_result_root, video_name, 'result', 'smooth_scores.pckl')
    with open(tracks_result_path, 'rb') as f: tracks_data = pickle.load(f)
    with open(smooth_scores_result_path, 'rb') as f: smooth_scores_data = pickle.load(f)
    with open(path.join(args.vaild_video_root, video_name+'.json'), 'r') as f: vaild_clip = json.load(f)
    for clip in vaild_clip:
        period, speaker_list = clip['period'], clip['track_index']
        ## mkdir for the data
        start_frame, end_frame = str(period[0]).zfill(6) , str(period[1]).zfill(6)
        clip_name = "{}_{}_{}to{}".format(DATA_SOURCE, video_name, start_frame, end_frame)
        save_path = path.join(args.datasets_root, clip_name)
        if path.exists(save_path): 
            pass
        else: os.mkdir(save_path)
        ## get speak-listen period for each person
        speaker_info_list = []
        speaker_info_merged_list = []
        for speaker_index in speaker_list:
            speaker_info_dic = {}
            if_speak_score = smooth_scores_data[speaker_index]
            speaker_track = tracks_data[speaker_index]['track']
            speak_period, listen_period, bboxes, if_speak_scores = get_speak_period(args.ks, 0, if_speak_score, speaker_track, period)
            speaker_info_dic['face_track'] = bboxes.tolist()
            speaker_info_dic['speak'] = speak_period
            speaker_info_dic['listen'] = listen_period
            speaker_info_dic['period_num'] = len(speak_period)+len(listen_period)
            speaker_info_dic['speaking_score'] = if_speak_scores
            
            speaker_info_list.append(speaker_info_dic)
            
            speaker_info_dic = {}
            speak_period_merged, listen_period_merged, bboxes, if_speak_scores = get_speak_period(args.ks, args.min_frame_s_l, if_speak_score, speaker_track, period)
            speaker_info_dic['face_track'] = bboxes.tolist()
            speaker_info_dic['speak'] = speak_period_merged
            speaker_info_dic['listen'] = listen_period_merged
            speaker_info_dic['period_num'] = len(speak_period_merged)+len(listen_period_merged)
            speaker_info_dic['speaking_score'] = if_speak_scores
            speaker_info_merged_list.append(speaker_info_dic)
            
        ## record the meta info for the data
        start_time, end_time = period[0]/FPS, period[1]/FPS
        meta = {'source': DATA_SOURCE, 'video': video_name, 'period': [start_time, end_time], 'num_speaker': len(speaker_list), 'url': url_dic[video_name+'.mp4'][-1]}
        ## save the result: video_clip(.mp4), speaker_info(.json), meta(.json), frames(dir/.png)
        
        frame_path = path.join(save_path, 'frames')
        if path.exists(frame_path): 
            pass
        else: os.mkdir(frame_path)
        
        command_video = ['ffmpeg', '-y', '-i',origin_video_path, '-ss', str(start_time), '-to', str(end_time), '-c', 'copy', path.join(save_path, 'video.mp4')]
        command_frame = ['ffmpeg', '-y', '-i', path.join(save_path, 'video.mp4'), path.join(frame_path, 'output_%04d.png')]
        with open(path.join(save_path, 'speaker_info.json'), 'w') as f: json.dump(speaker_info_list, f, indent=4)
        with open(path.join(save_path, 'speaker_info_merged.json'), 'w') as f: 
            print(save_path, speaker_info_merged_list[0].keys())
            json.dump(speaker_info_merged_list, f, indent=4)
        with open(path.join(save_path, 'meta.json'), 'w') as f: json.dump(meta, f, indent=4)
        subprocess.run(command_video)
        subprocess.run(command_frame)

def get_speak_period(ks, min_frame_s_l, scores, tracks, period):
    frames = tracks['frame'].tolist()
    start_index, end_index = frames.index(period[0]), frames.index(period[1])
    ## smooth the scores: denoise
    filitered_scores = medfilt(scores, ks)
    ## find the break point
    break_point = [start_index]
    for index in range(start_index+1, end_index):
        if filitered_scores[index]*filitered_scores[index-1] <= 0:
            break_point.append(index)
    break_point.append(end_index)
    
    del_list = []
    for index in range(1, len(break_point)):
        if break_point[index] - break_point[index-1] < min_frame_s_l:
            if index == end_index: 
                del_list.append(break_point[index-1]) 
            else:
                del_list.append(break_point[index]) 
    for point in del_list:
        break_point.remove(point)
        
    ## determain speak period
    speak_period = []
    liste_period = []
    for index in range(len(break_point)-1):
        start = break_point[index]
        end = break_point[index+1]
        avg_value = np.average(scores[start:end])
        # the frame index is in the local video
        start_frame_clip = frames[start] - period[0]
        end_frame_clip = frames[end] - period[0]
        if avg_value > 0: speak_period.append([start_frame_clip, end_frame_clip])
        else: liste_period.append([start_frame_clip, end_frame_clip])
    bboxes = tracks['bbox'][start_index:end_index]
    if_speak_scores = scores[start_index:end_index]
    return speak_period, liste_period, bboxes, if_speak_scores

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seg_result_root', type=str, default='', help='')
    parser.add_argument('--vaild_video_root', type=str, default='', help='')
    parser.add_argument('--datasets_root', type=str, default='', help='')
    parser.add_argument('--ks', type=int, default=5, help='the kernal size for the median filter of the scores')
    parser.add_argument('--min_frame_s_l', type=int, default=25, help='min frame num for speak/listen')
    args = parser.parse_args()
    
    file_name_list = [str(index) for index in range(0, 25)]
    for index in range(0, 25):
        file_name_list.append(str(index))

    all_data = file_name_list
    num_processes = 10
    ctx = multiprocessing.get_context('spawn')
    with ctx.Pool(processes=num_processes) as pool:
        func = partial(process, args=args)
        results = list(tqdm(pool.imap(func, all_data), total=len(all_data)))