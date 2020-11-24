import argparse
import torch
import warnings
warnings.simplefilter("ignore", UserWarning)

import glob
import time
import math
import numpy as np
import cv2

#from memory_profiler import profile
from facenet_pytorch import MTCNN, InceptionResnetV1
from sklearn.cluster import DBSCAN
from tqdm import tqdm
from deepsort.deep_sort import DeepSort
from detectors import DSFD
from utils.video_pipeline import get_videos_from_file
from utils.ops import xyxy_to_xywh

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
    parser.add_argument('--compare_threshold', default=0.9, type=float)

    # Followings are formats for int, float, Use them if argument type is int or float
    #parser.add_argument('--intformat', type=int, default=32)
    #parser.add_argument('--floatformat', type=float, default=10 * 60)
    args = parser.parse_args()
    return args

def embed_img(img, box, model, img_size=160):
    x1, y1, x2, y2 = box[0:4]
    x1 = int(x1)
    y1 = int(y1)
    x2 = int(x2)
    y2 = int(y2)

    img_cropped = img[y1:y2, x1:x2]
    img_cropped = cv2.resize(img_cropped, (img_size, img_size), interpolation=cv2.INTER_AREA)
    img_cropped = (img_cropped - 127.5) /128.0
    img_cropped = torch.tensor(img_cropped, dtype=torch.float32)

    # change (heigth, Width, channel) to (channel, heigth, width)
    img_cropped = img_cropped.permute(2, 0, 1)
    img_cropped = img_cropped.unsqueeze(dim=0).to(device)

    embed = model(img_cropped).detach().cpu()[0]
    
    del img_cropped

    return embed

@profile
def detecting(videos):

    detector = DSFD(device=device, PATH_WEIGHT = '/home/ubuntu/project/detectors/dsfd/weights/dsfd_vgg_0.880.pth')
    facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

    sort_weight = '/home/ubuntu/project/deepsort/deep/checkpoint/ckpt.t7'

    person_paths = glob.glob("/home/ubuntu/project/tmp/faces/*.png")
    person = []

    for path in person_paths:
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        boxes = detector.detect_faces(img, scales=[1.0])

        assert (len(boxes) == 1)

        box = boxes[0, 0:4]
        embed = embed_img(img, box, facenet)
        person.append(embed)

    frame_num = 0
    objects = []

    for video in videos:
        for frame in tqdm(video, desc="Frame Processed :"):
            img = frame.copy()

            boxes = detector.detect_faces(img, scales=[0.7])

            if len(boxes) > 0:

                boxes = boxes[:, 0:4]

                for box in boxes:
                    id = 0 # Defalut = None

                    new_obj = True

                    embed = embed_img(img, box, facenet)

                    min_norm = float("inf")

                    for i, p in enumerate(person):
                        norm = np.linalg.norm(embed - p)

                        if norm <= args.compare_threshold:
                            if norm < min_norm:
                                min_norm = norm
                                id = i + 1

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
        #video1_frames = video1_frames[:100]
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
            
            frame = cv2.putText(frame, f'person_{identity}', (x1 - 10, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,0.5, rgb, 2)
            frame = cv2.rectangle(frame, (x1, y1), (x2, y2), rgb, 1)

            frames[frame_num] = frame

    fourcc = cv2.VideoWriter_fourcc(*'XVID')   
    video_tracked = cv2.VideoWriter(f'/home/ubuntu/project/tmp/video_tracked_720p.mp4', fourcc, 20.0, dim)

    for frame in frames:
        video_tracked.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        
    video_tracked.release()

    if args.webm:
        fourcc = cv2.VideoWriter_fourcc(*'VP90')   
        webm = cv2.VideoWriter('/home/ubuntu/project/tmp/video_tracked_720p.webm', fourcc, 20.0, dim)

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