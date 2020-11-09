import argparse
import torch
import warnings
warnings.simplefilter("ignore", UserWarning)

import math
import numpy as np
import cv2

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
        self.faces = []

    def update(self, frame_num, box, embed):
        self.faces.append((frame_num, box, embed))

device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video1', type=str)
    parser.add_argument('--video2', type=str)
    parser.add_argument('--test', action='store_true')

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
def main(args):

    # Get frames from video
    video1_frames = get_videos_from_file(args.video1)

    if args.test:
        video2_frames = []
    else:
        video2_frames = get_videos_from_file(args.video2)

    videos = [video1_frames[:500], video2_frames]

    detector = DSFD(device=device, PATH_WEIGHT = './detectors/dsfd/weights/dsfd_vgg_0.880.pth')
    facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

    sort_weight = './deepsort/deep/checkpoint/ckpt.t7'

    frame_num = 0
    objects = []

    for video in videos:

        tracker = DeepSort(model_path=sort_weight, nms_max_overlap=0.5, use_cuda=True, max_age=1)

        for frame in tqdm(video, desc="Frame Processed :"):
            img = frame.copy()

            boxes = detector.detect_faces(img, scales=[0.5, 1.0])

            if len(boxes) > 0:

                boxes, scores = boxes[:, 0:4], boxes[:, 4]

                #transform xyxy boxes to xywh
                boxes = xyxy_to_xywh(boxes)

                outputs = tracker.update(boxes, scores, img)
            
            else: outputs = []

            for box in outputs:
                id = box[-1]
                new_obj = True

                embed = embed_img(img, box, facenet)

                for obj in objects:
                    if obj.id == id:
                        obj.update(frame_num, box[:4], embed)
                        new_obj = False
                        break

                if new_obj:
                    obj = detected_object(id)
                    obj.update(frame_num, box[:4], embed)
                    objects.append(obj)

            frame_num += 1

    means = []

    objects = np.array(objects)

    for obj in objects:
        arr = []

        for face in obj.faces:
            arr.append(np.array(face[2]))
            
        mean = np.stack(arr).mean(axis=0)

        means.append(mean)

        dist = []

        for face in obj.faces:
            dist.append(np.linalg.norm(np.array(face[2]) - mean))

        dist = np.array(dist)

        print(f"min : {dist.min()}")
        print(f"max : {dist.max()}")
        print(f"mean : {dist.mean()}")

    print(len(means))
    print(len(objects))

    clustering = DBSCAN(eps=0.3, min_samples=1).fit(means)

    lab = clustering.labels_

    detected = []

    for i, u in enumerate(np.unique(lab)):
        same_objs = objects[lab==u]

        first = same_objs[0]
        first.id = i

        for obj in same_objs[1:]:
            first.faces += obj.faces

        detected.append(first)

    return detected

def get_color(c, x, max_val):
        colors = np.array([[1, 0, 1], [0, 0, 1], [0, 1, 1], [0, 1, 0], [1, 1, 0], [1, 0, 0]], dtype=np.float32)

        ratio = float(x) / max_val * 5
        i = int(math.floor(ratio))
        j = int(math.ceil(ratio))
        ratio = ratio - i
        r = (1 - ratio) * colors[i][c] + ratio * colors[j][c]
        return int(r * 255)

def test(args):
    video1_frames = get_videos_from_file(args.video1)

    videos = [video1_frames[:500]]

    objects = main(args)

    H, W, _ = videos[0][0].shape
    
    dim = (W, H)

    frames = videos[0].copy()

    for obj in objects:

        for info in obj.faces:            
            frame_num, box, _ = info

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
    video_tracked = cv2.VideoWriter(f'./tmp/video_tracked_720p.mp4', fourcc, 20.0, dim)
    for frame in frames:
        video_tracked.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    video_tracked.release()

if __name__ == "__main__":
    args = parse_arguments()

    if args.test:
        test(args)
    else:
        main(args)