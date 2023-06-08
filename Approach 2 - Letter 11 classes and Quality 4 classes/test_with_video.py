# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 06:06:39 2020

@author: Riad
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
    quality_model = tf.keras.models.load_model("models/a_vgg16_model.h5")
    quality_assessment()
    
def b():
    print("আ")
    global quality_model 
    quality_model = tf.keras.models.load_model("models/b_vgg16_model.h5")
    quality_assessment()
    
def c():
    print("ই")
    global quality_model 
    quality_model = tf.keras.models.load_model("models/c_vgg16_model.h5")
    quality_assessment()
    
def d():
    print("ঈ")
    global quality_model 
    quality_model = tf.keras.models.load_model("models/d_vgg16_model.h5")
    quality_assessment()
    
def e():
    print("উ")
    global quality_model 
    quality_model = tf.keras.models.load_model("models/e_vgg16_model.h5")
    quality_assessment()
    
def f():
    print("ঊ")
    global quality_model 
    quality_model = tf.keras.models.load_model("models/f_vgg16_model.h5")
    quality_assessment()
    
def g():
    print("ঋ")
    global quality_model 
    quality_model = tf.keras.models.load_model("models/g_vgg16_model.h5")
    quality_assessment()
    
def h():
    print("এ")
    global quality_model 
    quality_model = tf.keras.models.load_model("models/h_vgg16_model.h5")
    quality_assessment()
    
def i():
    print("ঐ")
    global quality_model 
    quality_model = tf.keras.models.load_model("models/i_vgg16_model.h5")
    quality_assessment()
    
def j():
    print("ও")
    global quality_model 
    quality_model = tf.keras.models.load_model("models/j_vgg16_model.h5")
    quality_assessment()
    
def k():
    print("ঔ")
    global quality_model 
    quality_model = tf.keras.models.load_model("models/k_vgg16_model.h5")
    quality_assessment()    

def x():
    print("No letter has been found")
  

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
        10: k,
        11: x
    }
    func = switcher.get(letter_index, lambda: print("Character not recognized"))
    func()


#main code 
    
path= "C:\\Users\\Riad\\Documents\\GitHub\\Bangla-Handwriting\\Datasets\\Categorized-Dataset-with-Label\\a\\0.9\\"

# load model
model = load_model("models/vgg16_model_letter.h5")

print("Loaded model from disk")

'''
url = "http://192.168.0.100:4747" # Your url might be different, check the app
cam = cv2.VideoCapture(url+"/video")
'''
cam = cv2.VideoCapture(0)
while(True):
    _,frame=cam.read()
    img = cv2.resize(frame, (224, 224)) 
    img = img.reshape(-1,224,224,3)
    #frame=cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
    img = applications.vgg16.preprocess_input(img)
       
    
    
    prediction = model.predict(img)
    letter_index=(np.argmax(prediction))
    
    switch(letter_index)
    cv2.imshow("Video",frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cam.release()
cv2.destroyAllWindows()

    