import cv2 
import os 
import csv
import numpy as np 
from random import shuffle 
from tqdm import tqdm 
import tensorflow as tf
from tensorflow.keras import layers,Sequential,optimizers,applications, Model, applications
from tensorflow.keras.models import model_from_json, load_model
from matplotlib import pyplot as plt
from imgaug.augmentables.kps import KeypointsOnImage
import pandas as pd
import math
 
IMG_SIZE = 224

KEYPOINT_DEF = (
    "D:/Github Projects/Bangla-Handwriting/Datasets/Landmark detection/e_keypoint_definitions.csv"
)

# Load the metdata definition file and preview it.
keypoint_def = pd.read_csv(KEYPOINT_DEF)
keypoint_def.head()

# Extract the colours and labels.
colours = keypoint_def["Hex colour"].values.tolist()
colours = ["#" + colour for colour in colours]
labels = keypoint_def["Name"].values.tolist()

def visualize_keypoints(images, keypoints):
    fig, axes = plt.subplots(nrows=len(images)+1, ncols=2, figsize=(16, 12))
    [ax.axis("off") for ax in np.ravel(axes)]

    for (ax_orig, ax_all), image, current_keypoint in zip(axes, images, keypoints):
        ax_orig.imshow(image)
        ax_all.imshow(image)

        # If the keypoints were formed by `imgaug` then the coordinates need
        # to be iterated differently.
        if isinstance(current_keypoint, KeypointsOnImage):
            for idx, kp in enumerate(current_keypoint.keypoints):
                ax_all.scatter(
                    [kp.x], [kp.y], c=colours[idx], marker="x", s=50, linewidths=5
                )
        else:
            current_keypoint = np.array(current_keypoint)
            # Since the last entry is the visibility flag, we discard it.
            current_keypoint = current_keypoint[:, :2]
            for idx, (x, y) in enumerate(current_keypoint):
                ax_all.scatter([x], [y], c=colours[idx], marker="x", s=50, linewidths=5)

    plt.tight_layout(pad=2.0)
    plt.show()

def slope_pos_reward(x1, y1, x2, y2):
    if(x2 - x1 != 0):
        return (y2-y1)/(x2-x1)
    else:
        return 0

def get_area_penalty(x1, y1, x2, y2, x3, y3):
    area = 0.5 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
    return int(area)

path= "D:/Github Projects/Bangla-Handwriting/Datasets/Landmark detection/annotated_dataset/E dataset landmark"
img = "e-10.jpg"

# load model
model = load_model("e_keypoint_predictt.hdf5")

print("Loaded model from disk")


image_path = os.path.join(path, img) 
# loading the image from the path and then converting them into 
# greyscale for easier covnet prob 
img = cv2.imread(image_path, cv2.IMREAD_COLOR) 
  
# resizing the image for processing them in the covnet 
img = cv2.resize(img, (224, 224)) 
img = img.reshape(-1,224,224,3)         
prediction = model.predict(img).reshape(-1, 12, 2) * IMG_SIZE

print(prediction)

visualize_keypoints(img, prediction)


#Scoring mechanism

# Points 0,1 (if available),2,3 should be in same horizontal line - matra
matra_penalty = 0
matra_average_y = 0
if prediction[0][1][0] <= 10:
    matra_average_y = (prediction[0][0][1] + prediction[0][2][1] + prediction[0][3][1])/3
    matra_penalty = matra_penalty + abs(prediction[0][0][1] - matra_average_y)
    matra_penalty = matra_penalty + abs(prediction[0][2][1] - matra_average_y)
    matra_penalty = matra_penalty + abs(prediction[0][3][1] - matra_average_y)

else:
    matra_average_y = (prediction[0][0][1] + prediction[0][1][1] + prediction[0][2][1] + prediction[0][3][1])/4
    
    for i in range(0,4):
        matra_penalty = matra_penalty + abs(prediction[0][i][1] - matra_average_y)

# Check 1 and 4 are in same vertical line
_1_4_difference_penalty = abs(prediction[0][1][0] - prediction[0][4][0])   

# Distance between 1 & 4 and 4 and 7 is similar
_1_4_distance = math.sqrt( (prediction[0][1][0] - prediction[0][4][0])**2 + (prediction[0][1][1] - prediction[0][4][1])**2 )
_4_7_distance = math.sqrt( (prediction[0][4][0] - prediction[0][7][0])**2 + (prediction[0][4][1] - prediction[0][7][1])**2 ) 
_1_4_7_distance_ratio_penalty = abs(1 - (_1_4_distance/_4_7_distance))

# Check if 8 is right of point 9
_8_right_9_reward = prediction[0][8][0] - prediction[0][9][0]

# Check if 9 is right of point 0
_9_right_0_reward = prediction[0][9][0] - prediction[0][0][0]

# Check if 4 is below point 9
_4_below_9_reward = prediction[0][4][1] - prediction[0][9][1]

# Check if 9 is below point 0
_9_below_0_reward = prediction[0][9][1] - prediction[0][11][1]  

# Scoring 
e_score = - matra_penalty - _1_4_difference_penalty - _1_4_7_distance_ratio_penalty + _8_right_9_reward + _9_right_0_reward + _4_below_9_reward + _9_below_0_reward

print(e_score)








