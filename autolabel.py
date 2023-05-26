
import os
import sys
import json
import glob
import numpy as np
from PIL import Image

import torch
import pandas as pd


def create_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)
    return path

def voc_to_yolo_bbox(bbox, w, h):
    # xmin, ymin, xmax, ymax
    x_center = ((bbox[2] + bbox[0]) / 2) / w
    y_center = ((bbox[3] + bbox[1]) / 2) / h
    width = (bbox[2] - bbox[0]) / w
    height = (bbox[3] - bbox[1]) / h
    return [x_center, y_center, width, height]


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
img_paths = glob.iglob("dataset/*/data/images/*.npy")

model_imagenet = torch.hub.load('yolov5', 'custom', path='./yolo-finetune.pt', source='local', autoshape=True)
model_imagenet.conf = 0.05
model_imagenet.iou = 0.5
model_imagenet.agnostic = True

model_coco = torch.hub.load('yolov5', 'custom', path='./yolov5l.pt', source='local', autoshape=True)
model_coco.conf = 0.5
model_coco.iou = 0.5
model_coco.agnostic = True

for img_path in img_paths:
    # print(img_path)
    
    # save the bbox position in yolo format
    xywh_dir = create_dir(os.path.dirname(img_path).replace("images", "xywh"))
    xywh_file = os.path.basename(img_path)
    
    img_npy = np.squeeze(np.transpose(np.load(img_path), (1, 2, 0))).astype(np.uint8)
    img = Image.fromarray(img_npy)
    W, H = img.size
    results_imagenet = model_imagenet(img, size=256)
    results_coco = model_coco(img, size=256)
    
    # merge and bbox and pad
    results_imagenet = results_imagenet.pandas().xyxy[0]
    results_coco = results_coco.pandas().xyxy[0]
    results_coco = results_coco[results_coco["name"] == "person"]
    results = pd.concat([results_imagenet, results_coco], ignore_index=True)
    
    if results.empty:
        box_yolo = [0.5, 0.5, 1, 1]
    else:
        box = [results["xmin"].min(), results["ymin"].min(),
               results["xmax"].max(), results["ymax"].max()]
        box_yolo = voc_to_yolo_bbox(box, W, H)
    np.save(os.path.join(xywh_dir, xywh_file), box_yolo)
        
        
    
    

    
    