
import cv2

from InsightFace.mtcnn_pytorch.src.align_trans import warp_and_crop_face, get_reference_facial_points
from torchvision import transforms as trans

reference = get_reference_facial_points(default_square= True)
test_transform = trans.Compose([
                    trans.ToTensor(),
                    trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

def detect_align_face(img, model, scale=0.7):

    img_resized = cv2.resize(img, dsize=(0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

    boxes, probs, landmarks = model.detect(img_resized, landmarks=True)

    if boxes is None:
        return [], []

    faces = []
    for landmark in landmarks:
        face = warp_and_crop_face(img_resized, landmark, reference, crop_size=(112,112))
        faces.append(face)

    boxes = boxes / scale

    return boxes, faces, probs