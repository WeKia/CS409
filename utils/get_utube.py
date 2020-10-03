import cv2
import torch
import numpy as np
import mmcv

from PIL import Image, ImageDraw
from pytube import YouTube
from detectors import DSFD

def get_youtube(url):
    """Get Youtube Video from url
    :param str url:
        Youtube url to download make sure url is available

    :return frames:
        Frames of video """

    yt = YouTube(url)

    # Highest resolution is too big!
    #stream = yt.streams.filter(file_extension='mp4').order_by('resolution').desc().first()

    stream = yt.streams.filter(file_extension='mp4', res="720p").first()
    #stream = yt.streams.get_highest_resolution()
    stream.download(filename='tmp')

    video = mmcv.VideoReader('tmp.mp4')
    frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in video]

    return frames

if __name__ == '__main__':
    frames = get_youtube("https://www.youtube.com/watch?v=6RLLOEzdxsM")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    Net = DSFD(device=device)

    frames_tracked = []
    for i, frame in enumerate(frames):
        print('\rTracking frame: {}'.format(i + 1), end='')

        # Detect faces
        boxes = Net.detect_faces(frame, scales=[0.5, 1.0])

        # Draw faces
        frame_draw = frame.copy()
        draw = ImageDraw.Draw(frame_draw)
        if boxes is not None:
            for box in boxes:
                draw.rectangle(box[:-1].tolist(), outline=(255, 0, 0), width=6)

        # Add to frame list
        frames_tracked.append(frame_draw.resize((640, 360), Image.BILINEAR))

    dim = frames_tracked[0].size
    fourcc = cv2.VideoWriter_fourcc(*'FMP4')
    video_tracked = cv2.VideoWriter('video_tracked.mp4', fourcc, 25.0, dim)
    for frame in frames_tracked:
        video_tracked.write(cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR))
    video_tracked.release()