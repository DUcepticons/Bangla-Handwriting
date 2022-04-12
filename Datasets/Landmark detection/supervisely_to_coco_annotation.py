# -*- coding: utf-8 -*-
"""
Created on Sat Apr  9 02:06:36 2022

@author: Raiya
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import os
import json
import pandas as pd

import glob

json_files_location="D:\Github Projects\Bangla-Handwriting\Datasets\Landmark detection\A dataset landmark\A dataset landmark\\a\\ann"





'''
#print(data)
keypoints=data['people'][0]['pose_keypoints_2d']

keypoints_dict={}

for i in range(0,25):
    
    #print(joint_names[i],"- X:",keypoints[i*3],"- Y:",keypoints[i*3+1],"- P:",keypoints[i*3+2])
    keypoints_dict[joint_names[i]]={"X":keypoints[i*3],"Y":keypoints[i*3+1]}
'''

train_output = {}
train_images = []
train_annotations = []

val_output = {}
val_images = []
val_annotations = []

test_output = {}
test_images = []
test_annotations = []

for idx, json_file in enumerate(os.listdir(json_files_location)):
    file_path = os.path.join(json_files_location,json_file)
    f = open(file_path,)
    data = json.load(f)
    image = {}
    image["file_name"] = json_file[:-5]
    image["height"] = data['size']['height']
    image["width"] = data['size']['width']
    image["id"] = idx
    
    if idx <= 940:
        train_images.append(image)
    elif idx >940 and idx <=990:
        val_images.append(image)
    else:
        test_images.append(image)
    
    annotation = {}
    annotation["image_id"] = idx
    annotation["iscrowd"] = 0
    annotation["bbox"] = [0,0,data['size']['width'],data['size']['height']]
    annotation["category_id"]=1
    annotation["id"]= 1000+idx
    annotation["num_keypoints"] = len(data["objects"])
    
    #Declaring a zero array then filling it out
    keypoints = [0] * (3 * 12)
    sup_keypoints = data["objects"]
    for sup_keypoint in sup_keypoints:
        number = int(sup_keypoint["classTitle"][2:])
        keypoints[number*3] = sup_keypoint["points"]["exterior"][0][0]
        keypoints[number*3+1] = sup_keypoint["points"]["exterior"][0][1]
        keypoints[number*3+2] = 2
        

        
    annotation["keypoints"] = keypoints

    if idx <= 940:
        train_annotations.append(annotation)
    elif idx >940 and idx <=990:
        val_annotations.append(annotation)
    else:
        test_annotations.append(annotation)
        
    

train_output["images"]= train_images 
train_output["annotations"]= train_annotations    
with open("train_annotations.json", "w") as write_file:
    json.dump(train_output, write_file, indent=4)
    
val_output["images"]= val_images 
val_output["annotations"]= val_annotations    
with open("val_annotations.json", "w") as write_file:
    json.dump(val_output, write_file, indent=4)
    
test_output["images"]= test_images 
test_output["annotations"]= test_annotations    
with open("test_annotations.json", "w") as write_file:
    json.dump(test_output, write_file, indent=4)