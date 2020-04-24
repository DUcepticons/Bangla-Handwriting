import cv2 
import os 
import csv
import numpy as np 
from random import shuffle 
from tqdm import tqdm 
import tensorflow as tf
from tensorflow.keras import layers,Sequential,optimizers

path= 'D:/Github Projects/Bangla-Handwriting/Generated-Dataset-with-Label/Train/a-1-39.jpg'

# loading the image from the path and then converting them into 
# greyscale for easier covnet prob 
img = cv2.imread(path, cv2.IMREAD_GRAYSCALE) 
  
# resizing the image for processing them in the covnet 
img = cv2.resize(img, (350, 350)) 
         
check = np.array(img).reshape(-1,350,350,1)
   
model = tf.keras.models.load_model("bangla_model.hdf5")
prediction = model.predict(check)

print(prediction)
