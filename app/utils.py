########################################################################################################################
# ------------------------------------------- of AUTHOR: Omm AKA Antonio Colapso---------------------------------------#
########################################################################################################################
import time
import tensorflow as tf
from typing import List
import cv2
import os 
import bisect
import functools
import math
import os
import random
import re
import sys
import threading
from collections import Counter, defaultdict, deque
from copy import deepcopy
from functools import cmp_to_key, lru_cache, reduce
from heapq import heapify, heappop, heappush, heappushpop, nlargest, nsmallest
from io import BytesIO, IOBase
from itertools import accumulate, combinations, permutations
from operator import add, iand, ior, itemgetter, mul, xor
from string import ascii_lowercase, ascii_uppercase
from typing import *
import collections
import heapq
import itertools


# Disable all GPUS
tf.config.set_visible_devices([], 'GPU')





vocab = [x for x in "abcdefghijklmnopqrstuvwxyz'?!123456789 "]
char_to_num = tf.keras.layers.StringLookup(vocabulary=vocab, oov_token="")
# Mapping integers back to original characters
num_to_char = tf.keras.layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True
)

def load_video(path:str) -> List[float]: 
    #print(path)
    cap = cv2.VideoCapture(path)
    frames = []
    for _ in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))): 
        ret, frame = cap.read()
        frame = tf.image.rgb_to_grayscale(frame)
        frames.append(frame[190:236,80:220,:])
    cap.release()
    
    mean = tf.math.reduce_mean(frames)
    std = tf.math.reduce_std(tf.cast(frames, tf.float32))
    return tf.cast((frames - mean), tf.float32) / std
    
def load_alignments(path:str) -> List[str]: 
    #print(path)
    with open(path, 'r') as f: 
        lines = f.readlines() 
    tokens = []
    for line in lines:
        line = line.split()
        if line[2] != 'sil': 
            tokens = [*tokens,' ',line[2]]
    return char_to_num(tf.reshape(tf.strings.unicode_split(tokens, input_encoding='UTF-8'), (-1)))[1:]
def load_data(path: str): 
    path = bytes.decode(path.numpy())
    file_name = os.path.splitext(os.path.basename(path))[0]
    
    # Define the base directories
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.abspath(os.path.join(BASE_DIR, '..', 'data', 's1'))
    alignment_dir = os.path.abspath(os.path.join(BASE_DIR, '..', 'data', 'alignments', 's1'))

    # Construct the full paths
    video_path = os.path.join(data_dir, f'{file_name}.mpg')
    alignment_path = os.path.join(alignment_dir, f'{file_name}.align')

    # Check if the files exist
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file {video_path} does not exist.")
    if not os.path.exists(alignment_path):
        raise FileNotFoundError(f"Alignment file {alignment_path} does not exist.")
    
    frames = load_video(video_path)
    alignments = load_alignments(alignment_path)
    
    return frames, alignments

# def load_data(path: str): 
#     path = bytes.decode(path.numpy())
#     file_name = path.split('/')[-1].split('.')[0]
#     # File name splitting for windows
#     file_name = path.split('\\')[-1].split('.')[0]
#     video_path = os.path.join('..','data','s1',f'{file_name}.mpg')
#     alignment_path = os.path.join('..','data','alignments','s1',f'{file_name}.align')
#     frames = load_video(video_path) 
#     alignments = load_alignments(alignment_path)
    
#     return frames, alignments
