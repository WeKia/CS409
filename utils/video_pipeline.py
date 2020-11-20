import cv2
import torch
import numpy as np
import mmcv
import os

from pytube import YouTube

def get_youtube(url, resolutions='720p', use_cache=False):
    """Get Youtube Video from url
    :param str url:
        Youtube url to download make sure url is available

    :return frames:
        Frames of video """
    
    folder = '/home/ubuntu/project/tmp/'

    yt = YouTube(url)

    # Highest resolution is too big!
    #stream = yt.streams.filter(file_extension='mp4').order_by('resolution').desc().first()
    
    videos = []

    for res in resolutions:
        
        if (not os.path.exists(folder + f'tmp_{res}.mp4')) or not use_cache:
            stream = yt.streams.filter(file_extension='mp4', res=res).first()
            stream.download(output_path=folder ,filename=f'tmp_{res}')

        video = mmcv.VideoReader(folder + f'tmp_{res}.mp4')
        frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in video]
        
        videos.append(frames)

    del yt
    
    return videos

def get_videos_from_file(path):

    if not os.path.exists(path):
        raise "Video not found!"

    video = mmcv.VideoReader(path)

    frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in video]

    del video
    
    return frames