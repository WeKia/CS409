import numpy as np
import pandas as pd

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

def make_csv(objects, path):
    data = pd.DataFrame([], columns=['Obj', 'Start', 'End'])

    for obj in objects:
        start = last = obj.frames[0]
        for frame_num in obj.frames[1:]:
            if last != frame_num - 1:
                data = data.append({'Obj' :'Object'+ str(obj.id),
                                    'Start' : start,
                                    'End' : last}, ignore_index=True)
                start = frame_num

            last = frame_num
        
        if start != obj.frames[-1]:
            data = data.append({'Obj' :'Object'+ str(obj.id),
                                    'Start' : start,
                                    'End' : frame_num}, ignore_index=True)

    data.to_csv(path)