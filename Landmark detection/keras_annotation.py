
import numpy as np
import matplotlib.pyplot as plt
import math
import os
import json
import pandas as pd

import glob

json_files_location="D:/Github Projects/Bangla-Handwriting/Landmark detection/annotated_dataset/B dataset landmark/b/ann"


images = {}


for idx, json_file in enumerate(os.listdir(json_files_location)):
    file_path = os.path.join(json_files_location,json_file)
    f = open(file_path,)
    data = json.load(f)
    image = {}

    image["img_height"] = data['size']['height']
    image["img_width"] = data['size']['width']
    image["img_bbox"] = [0,0,data['size']['width'],data['size']['height']]
    image['is_multiple_dogs'] = False
    image['img_path'] = json_file[:-5]
    
    
    keypoints = [([0]*3) for i in range(15)]
    
    sup_keypoints = data["objects"]
    for sup_keypoint in sup_keypoints:
        #Declaring a zero array then filling it out
     
        number = int(sup_keypoint["classTitle"])
        keypoints[number][0] = sup_keypoint["points"]["exterior"][0][0]
        keypoints[number][1] = sup_keypoint["points"]["exterior"][0][1]
        keypoints[number][2] = 1
      
        

        
    image["joints"] = keypoints

    images [json_file[:-5]] = image
    
with open("b_annotations.json", "w") as write_file:
    json.dump(images, write_file, indent=4)