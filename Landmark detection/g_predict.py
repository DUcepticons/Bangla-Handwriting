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
    "D:/Github Projects/Bangla-Handwriting/Landmark detection/g_keypoint_definitions.csv"
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

# load model
model = load_model("g_keypoint_predict.h5")

print("Loaded model from disk")

letter_path = "D:/Github Projects/Bangla-Handwriting/Landmark detection/scored_dataset/g/"

output_file_path = "g-features-quality.csv" 

with open(output_file_path, mode='w', newline='') as file:
    # Create a CSV writer object
    writer = csv.writer(file)

    writer.writerow(["width", "height", "keypoint_0_x", "keypoint_0_y", "keypoint_1_x", "keypoint_1_y", "keypoint_2_x", "keypoint_2_y", "keypoint_3_x", "keypoint_3_y", "keypoint_4_x", "keypoint_4_y",  "keypoint_5_x", "keypoint_5_y", "keypoint_6_x", "keypoint_6_y", "keypoint_7_x", "keypoint_7_y", "keypoint_8_x", "keypoint_8_y", "keypoint_9_x", "keypoint_9_y", "keypoint_10_x", "keypoint_10_y", "keypoint_11_x", "keypoint_11_y", "keypoint_12_x", "keypoint_12_y", "feature_0", "feature_1", "feature_2", "feature_3", "feature_4", "feature_5", "feature_6", "feature_7", "feature_8", "feature_9", "quality"])

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
            prediction = model.predict(img).reshape(-1, 13, 2) * IMG_SIZE

            print(prediction)

            visualize_keypoints(img, prediction)


            #Scoring mechanism
            # 10-9 and 11-12 perpendicular
            _10_9_perpend_11_12_penalty = abs(90 - angle([prediction[0][10][0], prediction[0][10][1], prediction[0][9][0],prediction[0][9][1]], [prediction[0][11][0], prediction[0][11][1], prediction[0][12][0],prediction[0][12][1]]))


            # Check 0 and 2 are in same horizontal line
            _0_2_difference_penalty = abs(prediction[0][0][1] - prediction[0][2][1])   

            # Distance between 0 & 2 and 2 and 3 is similar
            _0_2_distance = math.sqrt( (prediction[0][0][0] - prediction[0][2][0])**2 + (prediction[0][0][1] - prediction[0][2][1])**2 )
            _2_3_distance = math.sqrt( (prediction[0][2][0] - prediction[0][3][0])**2 + (prediction[0][2][1] - prediction[0][3][1])**2 ) 
            _0_2_3_distance_ratio_penalty = abs(1 - (_0_2_distance/_2_3_distance))

            # 5-7 and 11-9 perpendicular
            _5_7_perpend_11_9_penalty = abs(90 - angle([prediction[0][5][0], prediction[0][5][1], prediction[0][7][0],prediction[0][7][1]], [prediction[0][11][0], prediction[0][11][1], prediction[0][9][0],prediction[0][9][1]]))

            # Check if 3 is below point 5
            _3_below_5_reward = prediction[0][3][1] - prediction[0][5][1]


            # 5-6 and 6-7 lines close to 67 degrees
            _5_6_67_deg_6_7_penalty = abs(67 - angle([prediction[0][5][0], prediction[0][5][1], prediction[0][6][0],prediction[0][6][1]], [prediction[0][6][0], prediction[0][6][1], prediction[0][7][0],prediction[0][7][1]]))

            # Check if 1 is below point 0-2 line
            _0_2_average = (prediction[0][0][1]+prediction[0][2][1])/2
            _1_below_0_2_reward = prediction[0][1][1] - _0_2_average

            # Check if 1 is between 0 and 2
            _1_right_0_reward = prediction[0][1][0] - prediction[0][0][0]
            _2_right_1_reward = prediction[0][2][0] - prediction[0][1][0]

            # Check if 6 is right of point 0
            _6_right_0_reward = prediction[0][6][0] - prediction[0][0][0]

            writer.writerow([img_width, img_height, prediction[0][0][0], prediction[0][0][1], prediction[0][1][0], prediction[0][1][1], prediction[0][2][0], prediction[0][2][1], prediction[0][3][0], prediction[0][3][1], prediction[0][4][0], prediction[0][4][1], prediction[0][5][0], prediction[0][5][1], prediction[0][6][0], prediction[0][6][1], prediction[0][7][0], prediction[0][7][1], prediction[0][8][0], prediction[0][8][1], prediction[0][9][0], prediction[0][9][1], prediction[0][10][0], prediction[0][10][1], prediction[0][11][0], prediction[0][11][1], prediction[0][12][0], prediction[0][12][1], _10_9_perpend_11_12_penalty, _0_2_difference_penalty, _0_2_3_distance_ratio_penalty, _5_7_perpend_11_9_penalty, _3_below_5_reward, _5_6_67_deg_6_7_penalty, _1_below_0_2_reward, _1_right_0_reward, _2_right_1_reward, _6_right_0_reward , quality_folder])

            # Scoring 
            g_score = - _10_9_perpend_11_12_penalty - _0_2_difference_penalty - _0_2_3_distance_ratio_penalty  - _5_7_perpend_11_9_penalty + _3_below_5_reward - _5_6_67_deg_6_7_penalty + _1_below_0_2_reward + _1_right_0_reward + _2_right_1_reward + _6_right_0_reward 

            print(g_score)

file.close()   






