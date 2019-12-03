import sys
import datetime
sys.path.append("./YOLO/")
import pickle
import torch
import cv2
import matplotlib.pyplot as plt
from utils import *
from darknet import Darknet
from PIL import Image
import numpy as np

cfg_file = './YOLO/cfg/yolov3.cfg'
weight_file = './YOLO/weights/yolov3.weights'
namesfile = './YOLO/data/coco.names'


m = Darknet(cfg_file)
m.load_weights(weight_file)
class_names = load_class_names(namesfile)
plt.rcParams['figure.figsize'] = [24.0, 14.0]
img = np.array(Image.open('./Flickr8k/Flicker8k_Dataset/2513260012_03d33305cf.jpg'))
img_to_hidden = {}

train_images_file = './Flickr8k/Flickr8k_text/Flickr_8k.trainImages.txt'
test_images_file = './Flickr8k/Flickr8k_text/Flickr_8k.testImages.txt'
val_images_file = './Flickr8k/Flickr8k_text/Flickr_8k.devImages.txt'


train_img = list(open(train_images_file, 'r').read().strip().split('\n'))
test_img = list(open(test_images_file, 'r').read().strip().split('\n'))
val_img = list(open(val_images_file, 'r').read().strip().split('\n'))
filenames = ["test", "val", "train"]
img_sets = [test_img, val_img, train_img]
img_dicts = [{}, {}, {}]

log = open("logfile.txt", "w")

for i in range(3):
    img_set = img_sets[i]
    log.write("------Beginning "+filenames[i]+"--------\n")
    log.flush()
    hiddenlayer_file = open(filenames[i]+"_hiddenweights.txt", 'w')
    total = len(img_set)
    for idx, image_name in enumerate(img_set):
        img = np.array(Image.open("./Flickr8k/Flicker8k_Dataset/" + image_name))
        original_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        resized_image = cv2.resize(original_image, (m.width, m.height))
        nms_thresh = 0.6
        iou_thresh = 0.4
        hidden = get_hidden_layer(m, resized_image, iou_thresh, nms_thresh).view(689520, -1).squeeze(1)
        hidden = hidden.detach().cpu().numpy()
        #img_dicts[i][image_name] = hidden

        hiddenlayer_file.write(image_name + " " + " ".join([str(x) for x in hidden]) + '\n')
        hiddenlayer_file.flush()
        if idx % 100 == 0:
            log.write(str(datetime.datetime.now()) + "\t" + str(idx / total) + "\n")
            log.flush()

    hiddenlayer_file.close()

    #with open(filenames[i]+"_hidden.pickle", 'wb') as handle:
    #    pickle.dump(img_dicts[i], handle, protocol=pickle.HIGHEST_PROTOCOL)
