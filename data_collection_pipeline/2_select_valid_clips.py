### select vaild video clips from the raw videos
import os
import pdb
import json
import tqdm
import pickle
import argparse
import itertools
import multiprocessing
from functools import partial
import numpy as np
from os import path

## input: track_1, track_2
## for each track: start_frame, end_frame
def get_overlaped_period(track_1, track_2):
    start_frame_1, end_frame_1 = track_1[0], track_1[1]
    start_frame_2, end_frame_2 = track_2[0], track_2[1]
    if start_frame_1 >= end_frame_2 or start_frame_2 >= end_frame_1: return None, None
    start_frame = max(start_frame_1, start_frame_2)
    end_frame = min(end_frame_1, end_frame_2)
    return start_frame, end_frame

## input: track_list, num_people
## return: the required video clip period
## Each track only shows in one scence
def get_vaild_period(track_list, area_list, args):
    vaild_period_list = []
    elements = list(range(len(track_list)))
    combinations = itertools.combinations(elements, args.num_people)
    
    for combination in combinations:
        area_human = []
        if_face_too_small = False
        for human_index in combination:
            if area_list[human_index] >= args.min_face_area:
                area_human.append(area_list[human_index])
            else: 
                if_face_too_small = True
                break
        if if_face_too_small: continue
        
        start_frame, end_frame = track_list[combination[0]][0], track_list[combination[0]][1]
        for index in range(1, args.num_people):
            start_frame, end_frame = get_overlaped_period([start_frame, end_frame], track_list[combination[index]])
            if start_frame is None: break
        if start_frame is None: continue
        else: 
            if int(end_frame) - int(start_frame) > args.min_frame:
                area_radio = np.min(area_human)/np.max(area_human)
                if area_radio > args.min_face_area_radio:
                    vaild_period_list.append({'period': [int(start_frame), int(end_frame)], 'track_index': combination, 'area_radio': area_radio})
                
    ## delect the overlap period: more than num people shows
    result = []
    period_dic = {}
    for index in range(len(vaild_period_list)):
        name = str(vaild_period_list[index]['period'][0])+'-'+str(vaild_period_list[index]['period'][1])
        if name in period_dic:
            period_dic[name]['num'] += 1
        else:
            period_dic[name] = {}
            period_dic[name]['num'], period_dic[name]['index'] = 1, index
    for key in period_dic.keys():
        if period_dic[key]['num'] == 1: 
            result.append(vaild_period_list[period_dic[key]['index']])
    return result

def select_video_clip(data, args):
    ## load the result
    file_name = str(data) 
    tracks_result_path = path.join(args.seg_result_root_path, file_name, 'result', 'tracks.pckl')
    # actually the tracks are all within one shot(according to 1_seg_speaker.py)
    with open(tracks_result_path, 'rb') as f: tracks_data = pickle.load(f)    
    ## select video clip
    track_list = []
    motion_list = []
    area_list = []
    for index in range(len(tracks_data)):
        track = tracks_data[index]['track']['frame']
        bbox = tracks_data[index]['track']['bbox']
        # track cneter motion
        area = np.average(np.abs((bbox[..., 0]-bbox[..., 2])*(bbox[..., 1]-bbox[..., 3])))
        
        track_list.append([np.min(track), np.max(track)])
        area_list.append(area)
    vaild_video_period = get_vaild_period(track_list, area_list, args)
    frame_all_num = 0
    for period in vaild_video_period:
        frame_all_num += period['period'][1] - period['period'][0]
    with open(path.join(args.select_video_clip_save_path, file_name+'.json'), 'w') as json_file:
        json.dump(vaild_video_period, json_file)
    return len(vaild_video_period), frame_all_num

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--min_face_area', type=int, default=64*64, help='')
    parser.add_argument('--min_face_area_radio', type=int, default=0.5, help='')
    parser.add_argument('--min_frame', type=int, default=25*5, help='the min frame number for a vaild video clip')
    parser.add_argument('--num_people', type=int, default=2, help='the number of people in the conversation')
    parser.add_argument('--seg_result_root_path', type=str, default='', help='')
    parser.add_argument('--select_video_clip_save_path', type=str, default='', help='')
    args = parser.parse_args()

    file_name_list = [index for index in range(0, 25)]
    num_data = 0
    all_frame = 0
    for idx, data in enumerate(file_name_list):
        num, frame_num = select_video_clip(data, args)
        num_data += num
        all_frame += frame_num
    print(num_data, "total length:", all_frame/25)