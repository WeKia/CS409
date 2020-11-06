import numpy as np
import cv2
import torch

from detectors import DSFD
from facenet_pytorch import MTCNN, InceptionResnetV1


def compare_faces(face1, face2):
    