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
    "D:/Github Projects/Bangla-Handwriting/Datasets/Landmark detection/b_keypoint_definitions.csv"
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

    
path= "D:/Github Projects/Bangla-Handwriting/Datasets/Landmark detection/annotated_dataset/B dataset landmark"
img = "b-10.jpg"

# load model
model = load_model("b_keypoint_predictt.hdf5")

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

# Points 0,1 (if available),2,3,4 should be in same horizontal line - matra
matra_penalty = 0
if prediction[0][1][0] <= 10:
    matra_average_y = (prediction[0][0][1] + prediction[0][2][1] + prediction[0][3][1] + prediction[0][4][1])/4
    matra_penalty = matra_penalty + abs(prediction[0][0][1] - matra_average_y)
    matra_penalty = matra_penalty + abs(prediction[0][2][1] - matra_average_y)
    matra_penalty = matra_penalty + abs(prediction[0][3][1] - matra_average_y)
    matra_penalty = matra_penalty + abs(prediction[0][4][1] - matra_average_y)
else:
    matra_average_y = (prediction[0][0][1] + prediction[0][1][1] + prediction[0][2][1] + prediction[0][3][1] + prediction[0][4][1])/5
    
    for i in range(0,5):
        matra_penalty = matra_penalty + abs(prediction[0][i][1] - matra_average_y)
    
# Points 3,11,12 should be in same vertical line - akar
akar_average_x = (prediction[0][3][0] + prediction[0][11][0] + prediction[0][12][0])/3
akar_penalty = 0
akar_penalty = akar_penalty + abs(prediction[0][3][0] - akar_average_x)
akar_penalty = akar_penalty + abs(prediction[0][11][0] - akar_average_x)
akar_penalty = akar_penalty + abs(prediction[0][12][0] - akar_average_x)

# Check if 11 is below point 8
_11_below_8_reward = prediction[0][11][1] - prediction[0][8][1]

# Check if 11 is below point 8
_14_below_8_reward = prediction[0][14][1] - prediction[0][8][1]

# Check if 6 is below point 11
_6_below_11_reward = prediction[0][6][1] - prediction[0][11][1]
    
# Check 0 and 9 are in same vertical line
_0_9_difference_penalty = abs(prediction[0][0][0] - prediction[0][9][0])    

# Check 1 (if available), 4 and 7 are in same vertical line
_1_5_8_penalty = 0
if prediction[0][1][0] <= 10:
    _1_5_8_difference_penalty = abs(prediction[0][4][0] - prediction[0][7][0])    
else:
    _1_5_8_average_x = (prediction[0][1][0] + prediction[0][5][0] + prediction[0][8][0])/3
    _1_5_8_penalty = _1_5_8_penalty + abs(prediction[0][1][0] - _1_5_8_average_x)
    _1_5_8_penalty = _1_5_8_penalty + abs(prediction[0][5][0] - _1_5_8_average_x)
    _1_5_8_penalty = _1_5_8_penalty + abs(prediction[0][8][0] - _1_5_8_average_x)   
    
# Distance between 6 and 7 < Distance between 7 and 8
_7_8_distance = math.sqrt( (prediction[0][8][0] - prediction[0][7][0])**2 + (prediction[0][8][1] - prediction[0][9][1])**2 )
_8_9_distance = math.sqrt( (prediction[0][9][0] - prediction[0][8][0])**2 + (prediction[0][9][1] - prediction[0][8][1])**2 )

_7_8_9_distance_reward = (_8_9_distance - _7_8_distance)

# Horizontal Distance between 0 and 2 > 2 x Horizontal Distance between 2 and 4
_0_2_difference = abs(prediction[0][0][0] - prediction[0][2][0])
_2_4_difference = abs(prediction[0][2][0] - prediction[0][4][0])
_0_2_two_times_2_4_length_reward = _0_2_difference - (2 * _2_4_difference)

# Scoring 
b_score = - matra_penalty - akar_penalty + _11_below_8_reward + _14_below_8_reward + _6_below_11_reward - _0_9_difference_penalty - _1_5_8_penalty + _7_8_9_distance_reward + _0_2_two_times_2_4_length_reward

print(b_score)








