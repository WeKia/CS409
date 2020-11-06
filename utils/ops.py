import numpy as np

def xyxy_to_xywh(boxes):
    xywh_boxes = []
    
    for box in boxes:
        x1 = int(box[0])
        y1 = int(box[1])
        x2 = int(box[2])
        y2 = int(box[3])
                
        w = abs(x2 - x1)
        h = abs(y2 - y1)
        x = x1 + w/2
        y = y1 + h/2
        
        xywh_boxes.append([x, y, w, h])
    
    return np.array(xywh_boxes)