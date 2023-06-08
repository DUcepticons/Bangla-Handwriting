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
    "D:/Github Projects/Bangla-Handwriting/Landmark detection/c_keypoint_definitions.csv"
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

    
# load model
model = load_model("c_keypoint_predict.h5")

print("Loaded model from disk")

letter_path = "D:/Github Projects/Bangla-Handwriting/Landmark detection/scored_dataset/c/"

output_file_path = "c-features-quality.csv" 

with open(output_file_path, mode='w', newline='') as file:
    # Create a CSV writer object
    writer = csv.writer(file)

    for quality_folder in os.listdir(letter_path):
        print ("Currently in quality folder: ",quality_folder)
        quality_path = os.path.join(letter_path, quality_folder)
        for img_file in os.listdir(quality_path): 


            img_path = os.path.join(quality_path, img_file)
        
    
            img = cv2.imread(img_path, cv2.IMREAD_COLOR) 
            
            # resizing the image for processing them in the covnet 
            img = cv2.resize(img, (224, 224)) 
            img = img.reshape(-1,224,224,3)         
            prediction = model.predict(img).reshape(-1, 12, 2) * IMG_SIZE

            #print(prediction)

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

            # Check if 7 is below point 6
            _7_below_6_reward = prediction[0][7][1] - prediction[0][6][1]

            # Check if matra is below point 9
            _matra_below_9_reward = matra_average_y - prediction[0][9][1]

            # Check if matra is below point 10
            _matra_below_10_reward = matra_average_y - prediction[0][10][1]

            # Check if matra is below point 11
            _matra_below_11_reward = matra_average_y - prediction[0][11][1]

            # Check if 10 is below point 11
            _10_below_11_reward = prediction[0][10][1] - prediction[0][11][1]

            # Check if 9 is below point 11
            _9_below_11_reward = prediction[0][9][1] - prediction[0][11][1]

            # Check if 10 is right of point 11
            _10_right_11_reward = prediction[0][10][0] - prediction[0][11][0]

            # Check if 9 is right of point 10
            _9_right_10_reward = prediction[0][9][0] - prediction[0][10][0]

            # Check if 2 is right of point 1
            _2_right_1_reward = prediction[0][2][0] - prediction[0][1][0]

            # Check if 5 is below point 4
            _5_below_4_reward = prediction[0][5][1] - prediction[0][4][1]

            # Check 4 and 5 are in same horizontal line
            _4_5_difference_penalty = abs(prediction[0][4][1] - prediction[0][5][1])    

            writer.writerow([matra_penalty, _7_below_6_reward, _matra_below_9_reward, _matra_below_10_reward, _matra_below_11_reward, _10_below_11_reward, _9_below_11_reward, _10_right_11_reward, _9_right_10_reward, _2_right_1_reward, _5_below_4_reward, _4_5_difference_penalty, quality_folder])           

            # Scoring 
            c_score = - matra_penalty + _7_below_6_reward + _matra_below_9_reward + _matra_below_10_reward + _matra_below_11_reward + _10_below_11_reward + _9_below_11_reward + _10_right_11_reward + _9_right_10_reward + _2_right_1_reward + _5_below_4_reward - _4_5_difference_penalty

            print(c_score)

file.close()   








