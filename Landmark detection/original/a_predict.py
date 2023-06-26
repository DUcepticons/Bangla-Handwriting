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
from angle_function import angle 

IMG_SIZE = 224

KEYPOINT_DEF = (
    "D:/Github Projects/Bangla-Handwriting/Landmark detection/a_keypoint_definitions.csv"
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
model = load_model("a_keypoint_predict.h5")

print("Loaded model from disk")

letter_path = "D:/Github Projects/Bangla-Handwriting/Landmark detection/scored_dataset/a/"

output_file_path = "a-features-quality.csv" 

with open(output_file_path, mode='w', newline='') as file:
    # Create a CSV writer object
    writer = csv.writer(file)

    writer.writerow(["width", "height", "keypoint_0_x", "keypoint_0_y", "keypoint_1_x", "keypoint_1_y", "keypoint_2_x", "keypoint_2_y", "keypoint_3_x", "keypoint_3_y", "keypoint_4_x", "keypoint_4_y",  "keypoint_5_x", "keypoint_5_y", "keypoint_6_x", "keypoint_6_y" "keypoint_7_x", "keypoint_7_y", "keypoint_8_x", "keypoint_8_y", "keypoint_9_x", "keypoint_9_y", "keypoint_10_x", "keypoint_10_y", "keypoint_11_x", "keypoint_11_y", "feature_0", "feature_1", "feature_2", "feature_3", "feature_4", "feature_5", "feature_6", "feature_7", "feature_8", "quality"])

    for quality_folder in os.listdir(letter_path):
        print ("Currently in quality folder: ",quality_folder)
        quality_path = os.path.join(letter_path, quality_folder)
        for img_file in os.listdir(quality_path):     
            img_path = os.path.join(quality_path, img_file)
        
    
            img = cv2.imread(img_path, cv2.IMREAD_COLOR) 
            img_height = img.shape[0]
            img_width = img.shape[1]
              
            # resizing the image for processing them in the covnet 
            img = cv2.resize(img, (224, 224)) 
            img = img.reshape(-1,224,224,3)         
            prediction = model.predict(img).reshape(-1, 12, 2) * IMG_SIZE
            
            print(prediction)
            
            visualize_keypoints(img, prediction)
            
            
            #Scoring mechanism
            
            # 0-2 and 3-10 perpendicular
            _0_2_perpend_3_10_penalty = abs(90 - angle([prediction[0][0][0], prediction[0][0][1], prediction[0][2][0],prediction[0][2][1]],  [prediction[0][3][0], prediction[0][3][1], prediction[0][10][0],prediction[0][10][1]]))
            
            
            # Points 0,1 (if available),2,3 should be in same horizontal line - matra
            matra_penalty = 0
            if prediction[0][1][0] <= 10:
                matra_average_y = (prediction[0][0][1] + prediction[0][2][1] + prediction[0][3][1])/3
                matra_penalty = matra_penalty + abs(prediction[0][0][1] - matra_average_y)
                matra_penalty = matra_penalty + abs(prediction[0][2][1] - matra_average_y)
                matra_penalty = matra_penalty + abs(prediction[0][3][1] - matra_average_y)
            else:
                matra_average_y = (prediction[0][0][1] + prediction[0][1][1] + prediction[0][2][1] + prediction[0][3][1])/4
                
                for i in range(0,4):
                    matra_penalty = matra_penalty + abs(prediction[0][i][1] - matra_average_y)
                
            # Points 3,11,10 should be in same vertical line - akar
            akar_average_x = (prediction[0][3][0] + prediction[0][11][0] + prediction[0][10][0])/3
            akar_penalty = 0
            akar_penalty = akar_penalty + abs(prediction[0][3][0] - akar_average_x)
            akar_penalty = akar_penalty + abs(prediction[0][11][0] - akar_average_x)
            akar_penalty = akar_penalty + abs(prediction[0][10][0] - akar_average_x)
            
            # Check if 10 is below point 7
            _10_below_7_reward = prediction[0][10][1] - prediction[0][7][1]
            
            # Check if 6 is below point 11
            _6_below_11_reward = prediction[0][6][1] - prediction[0][11][1]
                
            # Check 0 and 8 are in same vertical line
            _0_8_difference_penalty = abs(prediction[0][0][0] - prediction[0][8][0])    
            
            # Check 1 (if available), 4 and 7 are in same vertical line
            _1_4_7_difference_penalty = 0
            if prediction[0][1][0] <= 10:
                _1_4_7_difference_penalty = abs(prediction[0][4][0] - prediction[0][7][0])    
            else:
                _1_4_7_average_x = (prediction[0][1][0] + prediction[0][4][0] + prediction[0][7][0])/3
                _1_4_7_difference_penalty = _1_4_7_difference_penalty + abs(prediction[0][1][0] - _1_4_7_average_x)
                _1_4_7_difference_penalty = _1_4_7_difference_penalty + abs(prediction[0][4][0] - _1_4_7_average_x)
                _1_4_7_difference_penalty = _1_4_7_difference_penalty + abs(prediction[0][7][0] - _1_4_7_average_x)   
                
            # Distance between 6 and 7 < Distance between 7 and 8
            #_5_6_distance = math.sqrt( (prediction[0][6][0] - prediction[0][5][0])**2 + (prediction[0][6][1] - prediction[0][5][1])**2 )
            _6_7_distance = math.sqrt( (prediction[0][7][0] - prediction[0][6][0])**2 + (prediction[0][7][1] - prediction[0][6][1])**2 )
            _7_8_distance = math.sqrt( (prediction[0][8][0] - prediction[0][7][0])**2 + (prediction[0][8][1] - prediction[0][7][1])**2 )
            
            _6_7_8_distance_reward = (_7_8_distance - _6_7_distance)
            
            # Horizontal Distance between 0 and 2 > 3 x Horizontal Distance between 2 and 3 
            _0_2_difference = abs(prediction[0][0][0] - prediction[0][2][0])
            _2_3_difference = abs(prediction[0][2][0] - prediction[0][3][0])
            _0_2_four_times_2_3_length_reward = _0_2_difference - (3 * _2_3_difference)
            
            writer.writerow([img_width, img_height, prediction[0][0][0], prediction[0][0][1], prediction[0][1][0], prediction[0][1][1], prediction[0][2][0], prediction[0][2][1], prediction[0][3][0], prediction[0][3][1], prediction[0][4][0], prediction[0][4][1], prediction[0][5][0], prediction[0][5][1], prediction[0][6][0], prediction[0][6][1], prediction[0][7][0], prediction[0][7][1], prediction[0][8][0], prediction[0][8][1], prediction[0][9][0], prediction[0][9][1], prediction[0][10][0], prediction[0][10][1], prediction[0][11][0], prediction[0][11][1], _0_2_perpend_3_10_penalty, matra_penalty, akar_penalty, _10_below_7_reward, _6_below_11_reward, _0_8_difference_penalty, _1_4_7_difference_penalty, _6_7_8_distance_reward, _0_2_four_times_2_3_length_reward, quality_folder])
            
            
            # Scoring 
            a_score = - _0_2_perpend_3_10_penalty - matra_penalty - akar_penalty + _10_below_7_reward + _6_below_11_reward - _0_8_difference_penalty - _1_4_7_difference_penalty + _6_7_8_distance_reward + _0_2_four_times_2_3_length_reward
            
            print(a_score)
        
file.close()   








