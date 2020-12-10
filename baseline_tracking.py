import argparse
import torch
import warnings
warnings.simplefilter("ignore", UserWarning)

import time
import math
import numpy as np
import cv2

#from memory_profiler import profile
from facenet_pytorch import MTCNN, InceptionResnetV1
from sklearn.cluster import DBSCAN
from InsightFace.model import Backbone
from tqdm import tqdm
from deepsort.deep_sort import DeepSort
from detectors import DSFD
from utils.detect_align import detect_align_face, test_transform
from utils.video_pipeline import get_videos_from_file
from utils.ops import xyxy_to_xywh, make_csv
from utils.scene_dectector import scene_detect

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
    parser.add_argument('--embbeder', type=str, default='Arcface')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--comparing', action='store_true')
    parser.add_argument('--compare_threshold', default=0.5, type=float)
    parser.add_argument('--webm', action='store_true')

    # Followings are formats for int, float, Use them if argument type is int or float
    #parser.add_argument('--intformat', type=int, default=32)
    #parser.add_argument('--floatformat', type=float, default=10 * 60)
    args = parser.parse_args()
    return args

def embed_img_facenet(img, box, model, img_size=160):
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

def embed_img_arcface(face, embedder):
    trans_img = test_transform(face).to('cuda').unsqueeze(0)

    with torch.no_grad():
        embed = embedder(trans_img)

    return embed

#@profile
def scene_detecting(videos):
    return scene_detect(videos)

#@profile
def tracking(videos, args):

    # We can do it more simply but not now
    if args.embbeder == "facenet":
        detector = DSFD(device=device, PATH_WEIGHT = '/home/ubuntu/project/detectors/dsfd/weights/dsfd_vgg_0.880.pth')
        facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    else:
        detector = MTCNN(image_size=112, device=device, post_process=False, thresholds=[0.7, 0.8, 0.9])
        arcface = Backbone(50, 0.6, 'ir_se').to('cuda')

        arcface.load_state_dict(torch.load('/home/ubuntu/project/InsightFace/weights/ir_50se.pth'))
        arcface.eval()

    sort_weight = '/home/ubuntu/project/deepsort/deep/checkpoint/ckpt.t7'

    tracker = DeepSort(model_path=sort_weight, n_init=0, nms_max_overlap=0.5, use_cuda=True, max_age=1)

    objects = []

    for video in videos:

        tracker.reset()
        
        scene_list = scene_detecting(video)
        
        frame_num = 0

        for frame in tqdm(video, desc="Frame Processed :"):

            if frame_num >= scene_list[0]:
                tracker.reset()
                scene_list.pop(0)

            img = frame.copy()

            if args.embbeder == "facenet":
                boxes = detector.detect_faces(img, scales=[0.7])

                if len(boxes) > 0:

                    boxes, scores = boxes[:, 0:4], boxes[:, 4]

                    #transform xyxy boxes to xywh
                    boxes = xyxy_to_xywh(boxes)

                    outputs = tracker.update(boxes, scores, img)
                else: outputs = []
            else:
                boxes, faces, scores = detect_align_face(img, detector)

                outputs = tracker.update(boxes, scores, img)

                assert (len(outputs) == len(faces))

            for box in outputs:
                id = box[-1]
                new_obj = True

                if args.embbeder == "facenet":
                    embed = embed_img_facenet(img, box, facenet)
                else:
                    embed = embed_img_arcface(face, arcface)

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
    
    return objects


#@profile
def clustering(objects):

    means = []

    for obj in objects:
        arr = []

        for face in obj.faces:
            arr.append(np.array(face))
            
        mean = np.stack(arr).mean(axis=0)

        means.append(mean)

        dist = []

        for face in obj.faces:
            dist.append(np.linalg.norm(np.array(face) - mean))

        dist = np.array(dist)

    clustering = DBSCAN(eps=0.7, min_samples=1).fit(means)

    lab = clustering.labels_

    detected = []

    for i, u in enumerate(np.unique(lab)):
        same_objs = objects[lab==u]

        first = same_objs[0]
        first.id = i

        for obj in same_objs[1:]:
            first.merge(obj)

        detected.append(first)

    return detected


#@profile
def comparing(objects):
    
    detected = []
    objects = list(objects)

    print(f"object length : {len(objects)}")

    while len(objects) > 0:
        popped, objects = objects[0], objects[1:]

        arr = []

        for face in popped.faces:
            arr.append(np.array(face))
            
        popped_mean_face = np.stack(arr).mean(axis=0)

        popped.id = len(detected)
        detected.append(popped)

        remove_list = []

        for obj in objects:

            if len(set(obj.frames) & set(popped.frames)) > 0:
                continue

            arr = []

            for face in obj.faces:
                arr.append(np.array(face))
                
            mean = np.stack(arr).mean(axis=0)

            if (np.linalg.norm(mean - popped_mean_face) <= args.compare_threshold):
                remove_list.append(obj)
                popped.merge(obj)
            
        for obj in remove_list:
            objects.remove(obj)

    return detected

def main(args):

    # Get frames from video
    video1_frames = get_videos_from_file(args.video1)

    if args.test:
        video2_frames = []
        video1_frames = video1_frames[:1500]
    else:
        video2_frames = get_videos_from_file(args.video2)

    videos = [video1_frames, video2_frames]

    del video1_frames

    objects = tracking(videos, args)

    objects = np.array(objects)

    if args.comparing:
        detected = comparing(objects)
    else:
        detected = clustering(objects)

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

    make_csv(objects, '/home/ubuntu/project/tmp/advanced.csv')

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