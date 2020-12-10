import argparse
import torch
import warnings
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", FutureWarning)

import glob
import time
import math
import pandas as pd
import numpy as np
import cv2

#from memory_profiler import profile
from facenet_pytorch import MTCNN, InceptionResnetV1
from tqdm import tqdm
from deepsort.deep_sort import DeepSort
from InsightFace.model import Backbone
from utils.detect_align import detect_align_face, test_transform
from utils.video_pipeline import get_videos_from_file
from utils.ops import xyxy_to_xywh, make_csv

class detected_object:
    def __init__(self, id):
        self.id = id
        self.frames = []
        self.boxes = []
        self.faces = []

    def update(self, frame_num, box, embed):
        self.frames.append(frame_num)
        self.boxes.append(box)
        self.faces.append(embed)

    def merge(self, obj):
        self.frames += obj.frames
        self.boxes += obj.boxes
        self.faces += obj.faces

device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video1', type=str)
    parser.add_argument('--video2', type=str)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--webm', action='store_true')
    parser.add_argument('--compare_threshold', default=1.0, type=float)

    # Followings are formats for int, float, Use them if argument type is int or float
    #parser.add_argument('--intformat', type=int, default=32)
    #parser.add_argument('--floatformat', type=float, default=10 * 60)
    args = parser.parse_args()
    return args

def embed_img(face, embedder):
    trans_img = test_transform(face).to('cuda').unsqueeze(0)

    with torch.no_grad():
        embed = embedder(trans_img)

    return embed

@profile
def detecting(videos):

    detector = MTCNN(image_size=112, device=device, post_process=False, thresholds=[0.7, 0.8, 0.9])
    arcface = Backbone(50, 0.6, 'ir_se').to('cuda')

    arcface.load_state_dict(torch.load('/home/ubuntu/project/InsightFace/weights/ir_50se.pth'))
    arcface.eval()

    sort_weight = '/home/ubuntu/project/deepsort/deep/checkpoint/ckpt.t7'

    person_paths = glob.glob("/home/ubuntu/project/tmp/faces/*.png")
    person = None

    for path in person_paths:
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        _, faces = detect_align_face(img, detector)

        assert (len(faces) == 1)

        embed = embed_img(faces[0], arcface)

        if person is None:
            person = embed
        else:
            person = torch.cat([person, embed])

    frame_num = 0
    objects = []

    for video in videos:
        for frame in tqdm(video, desc="Frame Processed :"):
            img = frame.copy()

            boxes, faces = detect_align_face(img, detector)

            if len(boxes) > 0:

                boxes = boxes[:, 0:4]

                for box, face in zip(boxes, faces):
                    id = 0 # Defalut = None

                    new_obj = True

                    embed = embed_img(face, arcface)

                    norm = (embed - person).norm(dim=1)

                    norm, idx = torch.min(norm).item(), torch.argmin(norm).item()

                    if norm <= args.compare_threshold:
                        id = idx + 1

                    for obj in objects:
                        if obj.id == id:
                            obj.update(frame_num, box, embed)
                            new_obj = False
                            break

                    if new_obj:
                        obj = detected_object(id)
                        obj.update(frame_num, box, embed)
                        objects.append(obj)
            frame_num += 1
    
    return objects


def main(args):

    # Get frames from video
    video1_frames = get_videos_from_file(args.video1)

    if args.test:
        video2_frames = []
        #video1_frames = video1_frames[:400]
    else:
        video2_frames = get_videos_from_file(args.video2)

    videos = [video1_frames, video2_frames]

    del video1_frames

    objects = detecting(videos)

    detected = np.array(objects)

    return detected, videos

def get_color(c, x, max_val):
        colors = np.array([[1, 0, 1], [0, 0, 1], [0, 1, 1], [0, 1, 0], [1, 1, 0], [1, 0, 0]], dtype=np.float32)

        ratio = float(x) / max_val * 5
        i = int(math.floor(ratio))
        j = int(math.ceil(ratio))
        ratio = ratio - i
        r = (1 - ratio) * colors[i][c] + ratio * colors[j][c]
        return int(r * 255)

def test(args):
    objects, videos = main(args)

    H, W, _ = videos[0][0].shape
    
    dim = (W, H)

    frames = videos[0].copy()

    for obj in objects:
        if obj.id == 0:
            continue

        for i in range(len(obj.faces)):            
            frame_num = obj.frames[i]
            box = obj.boxes[i]

            frame = frames[frame_num]

            x1 = int(box[0])
            y1 = int(box[1])
            x2 = int(box[2])
            y2 = int(box[3])
            
            identity = obj.id
            
            offset = identity * 123457 % 80
            red = get_color(2, offset, 80)
            green = get_color(1, offset, 80)
            blue = get_color(0, offset, 80)
            
            rgb = (red, green, blue)
            
            frame = cv2.putText(frame, f'person_{identity}', (x1 - 10, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, rgb, 2)
            frame = cv2.rectangle(frame, (x1, y1), (x2, y2), rgb, 1)

            frames[frame_num] = frame
        
    make_csv(objects, '/home/ubuntu/project/tmp/baseline.csv')

    fourcc = cv2.VideoWriter_fourcc(*'XVID')   
    video_tracked = cv2.VideoWriter(f'/home/ubuntu/project/tmp/video_tracked_720p.mp4', fourcc, 30.0, dim)

    for frame in frames:
        video_tracked.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        
    video_tracked.release()

    if args.webm:
        fourcc = cv2.VideoWriter_fourcc(*'VP90')   
        webm = cv2.VideoWriter('/home/ubuntu/project/tmp/video_tracked_720p.webm', fourcc, 30.0, dim)

        for frame in frames:
            webm.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        
        webm.release()

if __name__ == "__main__":
    init_time = time.time()

    args = parse_arguments()

    if args.test:
        test(args)
    else:
        main(args)

    print(f"total elapsed time : {time.time()-init_time}s")