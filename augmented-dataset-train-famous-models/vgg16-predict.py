# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 20:23:54 2020

@author: Raiyaan Abdullah
"""
import cv2 
import os 
import csv
import numpy as np 
from random import shuffle 
from tqdm import tqdm 
import tensorflow as tf
from tensorflow.keras import layers,Sequential,optimizers,applications, Model, applications
from tensorflow.keras.models import model_from_json, load_model

letter_array=['a','b','c','d','e','f','g','h','i','j','k']
quality_array=['0.6','0.7','0.8','0.9']

#main code 
    
image_path= 'D:/Github Projects/Bangla-Handwriting/custom-data/i.jpg'

# load model
model = load_model("vgg16_model.h5")

print("Loaded model from disk")




# loading the image from the path and then converting them into 
# greyscale for easier covnet prob 
img = cv2.imread(image_path, cv2.IMREAD_COLOR) 
  
# resizing the image for processing them in the covnet 
img = cv2.resize(img, (224, 224)) 
img = img.reshape(-1,224,224,3)         
img = applications.vgg16.preprocess_input(img)
   

prediction = model.predict(img)
index = (np.argmax(prediction))
letter = letter_array[int(index / 4)]
quality = quality_array[int(index % 4)]


print(letter,' ',quality)
