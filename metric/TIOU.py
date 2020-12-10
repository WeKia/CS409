import os
import sys

sys.path.append('/home/ubuntu/project')

import torch
import argparse
import glob
import pandas as pd
import numpy as np
import cv2


from facenet_pytorch import MTCNN
from InsightFace.model import Backbone
from utils.detect_align import warp_and_crop_face

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class group:
    def __init__(self, name):
        self.name = name
        self.faces = []
        self.frames = []

    def add_face(self, face):
        self.faces.append(face)

    def add_frames(self, start, end):
        self.frames.append((start, end))

def embed_img(face, embedder):
    trans_img = test_transform(face).to('cuda').unsqueeze(0)

    with torch.no_grad():
        embed = embedder(trans_img)

    return embed

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--GT', type=str)
    parser.add_argument('--prediction', type=str)
    parser.add_argument('--compare_threshold', default=1.0, type=float)

    args = parser.parse_args()
    return args

def load_data(path):

    """
    detector = MTCNN(image_size=112, device=device, post_process=False, thresholds=[0.7, 0.8, 0.9])
    arcface = Backbone(50, 0.6, 'ir_se').to('cuda')

    arcface.load_state_dict(torch.load('/home/ubuntu/project/InsightFace/weights/ir_50se.pth'))
    arcface.eval()
    """

    if not os.path.exists(path):
        raise Exception(f'Given path {path} doesn\'t exist!')

    if not os.path.isdir(path):
        raise Exception(f'Given path {path} is not directory!')

    csv_path = glob.glob(path + '/*.csv')[0]

    csv = pd.read_csv(csv_path)

    objs = np.unique(csv['Obj'].values)

    outputs = []

    for obj in objs:
        obj_path = path + '/' + obj

        obj_group = group(obj)

        """
        if not os.path.exists(obj_path):
            raise Exception(f'Obj {obj} is not in path {obj_path}!')

        if not os.path.exists(obj_path):
            raise Exception(f'Obj {obj} path {obj_path} is not directory!')

        
        obj_faces = glob.glob(obj_path + '/*')

        for face in obj_faces:
            img = cv2.imread(face)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            _, faces = warp_and_crop_face(img, detector)

            assert (len(faces) == 1)

            embed = embed_img(faces[0], arcface)

            obj_group.add_face(embed)
        """

        for start, end in zip(csv[csv['Obj'] == obj]['Start'], csv[csv['Obj'] == obj]['End']):
            obj_group.add_frames(start, end)
        
        outputs.append(obj_group)
    
    return outputs

def getbyname(objs, name):
    for obj in objs:
        if obj.name == name:
            return obj

def calculate_tiou(obj1, obj2):
    frames = [0 for _ in range(30 * 150)] # set 30(fps) * 150(2min 30sec) frames to 0

    for obj in [obj1, obj2]:
        if obj is None:
            return 0

        for f in obj.frames:
            start, end = f

            for i in range(start, end+1):
                frames[i] = frames[i] + 1

    frames = np.array(frames)

    tiou =  len(frames[frames>=2]) / len(frames[frames>=1])
    
    return tiou
        
def main(args):
    GT = load_data(args.GT)
    Pred = load_data(args.prediction)

    pairs = []

    # AS we have just one data
    # We make pairs by setting same object with same name
    
    GT_objs = [dat.name for dat in GT]
    pred_objs = [dat.name for dat in Pred]

    names = np.unique(GT_objs + pred_objs)

    TIOU = []

    for obj_name in names:
        gt = getbyname(GT, obj_name)
        pred = getbyname(Pred, obj_name)

        tiou = calculate_tiou(gt, pred)

        TIOU.append(tiou)

    print(TIOU)
    print(np.mean(TIOU))

if __name__ == '__main__':
    args = parse_arguments()
    main(args)