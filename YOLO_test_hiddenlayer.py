import sys
sys.path.append("./YOLO/")

import torch
import cv2
import matplotlib.pyplot as plt
from utils import *
from darknet import Darknet
from PIL import Image

cfg_file = './YOLO/cfg/yolov3.cfg'
weight_file = './YOLO/weights/yolov3.weights'
namesfile = './YOLO/data/coco.names'

m = Darknet(cfg_file)
m.load_weights(weight_file)
class_names = load_class_names(namesfile)
plt.rcParams['figure.figsize'] = [24.0, 14.0]
img = np.array(Image.open('./Flickr8k/Flicker8k_Dataset/2513260012_03d33305cf.jpg'))
original_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
resized_image = cv2.resize(original_image, (m.width, m.height))
nms_thresh = 0.6
iou_thresh = 0.4
hidden = get_hidden_layer(m, resized_image, iou_thresh, nms_thresh)
print(hidden)