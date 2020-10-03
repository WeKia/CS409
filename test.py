import warnings
warnings.simplefilter("ignore", UserWarning)
import cv2
import torch
import numpy as np

from PIL import Image, ImageDraw
from utils.get_utube import get_youtube
from detectors import DSFD

def main():
    frames = get_youtube("https://www.youtube.com/watch?v=6RLLOEzdxsM")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    Net = DSFD(device=device)

    frames_tracked = []
    for i, frame in enumerate(frames[:2000]):
        print('\rTracking frame: {}'.format(i + 1), end='')

        # Detect faces
        boxes = Net.detect_faces(frame, conf_th=0.9,scales=[0.5, 1.0])

        # Draw faces
        frame_draw = Image.fromarray(frame.copy())
        draw = ImageDraw.Draw(frame_draw)
        if boxes is not None:
            for box in boxes:
                draw.rectangle(box[:-1].tolist(), outline=(255, 0, 0), width=6)

        # Add to frame list
        frames_tracked.append(frame_draw)

    dim = frames_tracked[0].size
    fourcc = cv2.VideoWriter_fourcc(*'FMP4')
    video_tracked = cv2.VideoWriter('video_tracked.mp4', fourcc, 25.0, dim)
    for frame in frames_tracked:
        video_tracked.write(cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR))
    video_tracked.release()


if __name__ == "__main__":
    main()