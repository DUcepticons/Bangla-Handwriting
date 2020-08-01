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



#quality detection functions

def quality_switch(quality_index):
    switcher = {
        0: sixty,
        1: seventy,
        2: eighty,
        3: ninety
    }
    func = switcher.get(quality_index, "Cannot judge quality")
    func()

def sixty():
    print ( "Quality 60%")
def seventy():
    print ( "Quality 70%")
def eighty():
    print ( "Quality 80%")
def ninety():
    print ( "Quality 90%")

quality_model= None

def quality_assessment():
    quality_pred = quality_model.predict(img)
    quality_index=(np.argmax(quality_pred))
    quality_switch(quality_index)
    #print(quality_pred)


#letter detection functions

def a():
    print("অ")
    global quality_model 
    quality_model = tf.keras.models.load_model("letter-models/vgg16/a_vgg16_model.h5")
    quality_assessment()
    
def b():
    print("আ")
    global quality_model 
    quality_model = tf.keras.models.load_model("letter-models/vgg16/b_vgg16_model.h5")
    quality_assessment()
    
def c():
    print("ই")
    global quality_model 
    quality_model = tf.keras.models.load_model("letter-models/vgg16/c_vgg16_model.h5")
    quality_assessment()
    
def d():
    print("ঈ")
    global quality_model 
    quality_model = tf.keras.models.load_model("letter-models/vgg16/d_vgg16_model.h5")
    quality_assessment()
    
def e():
    print("উ")
    global quality_model 
    quality_model = tf.keras.models.load_model("letter-models/vgg16/e_vgg16_model.h5")
    quality_assessment()
    
def f():
    print("ঊ")
    global quality_model 
    quality_model = tf.keras.models.load_model("letter-models/vgg16/f_vgg16_model.h5")
    quality_assessment()
    
def g():
    print("ঋ")
    global quality_model 
    quality_model = tf.keras.models.load_model("letter-models/vgg16/g_vgg16_model.h5")
    quality_assessment()
    
def h():
    print("এ")
    global quality_model 
    quality_model = tf.keras.models.load_model("letter-models/vgg16/h_vgg16_model.h5")
    quality_assessment()
    
def i():
    print("ঐ")
    global quality_model 
    quality_model = tf.keras.models.load_model("letter-models/vgg16/i_vgg16_model.h5")
    quality_assessment()
    
def j():
    print("ও")
    global quality_model 
    quality_model = tf.keras.models.load_model("letter-models/vgg16/j_vgg16_model.h5")
    quality_assessment()
    
def k():
    print("ঔ")
    global quality_model 
    quality_model = tf.keras.models.load_model("letter-models/vgg16/k_vgg16_model.h5")
    quality_assessment()    


def switch(letter_index):
    switcher = {
        0: a,
        1: b,
        2: c,
        3: d,
        4: e,
        5: f,
        6: g,
        7: h,
        8: i,
        9: j,
        10: k
    }
    func = switcher.get(letter_index, lambda: print("Character not recognized"))
    func()


#main code 
    
path= 'D:/Github Projects/Bangla-Handwriting/Datasets/Categorized-Dataset-with-Label/d/0.8/'

# load model
model = load_model("vgg16_model_letter.h5")

print("Loaded model from disk")


for img in os.listdir(path):
    image_path = os.path.join(path, img) 
    # loading the image from the path and then converting them into 
    # greyscale for easier covnet prob 
    img = cv2.imread(image_path, cv2.IMREAD_COLOR) 
      
    # resizing the image for processing them in the covnet 
    img = cv2.resize(img, (224, 224)) 
    img = img.reshape(-1,224,224,3)         
    img = applications.vgg16.preprocess_input(img)
       
    
    
    prediction = model.predict(img)
    letter_index=(np.argmax(prediction))
    
    switch(letter_index)
    #print(prediction)
    