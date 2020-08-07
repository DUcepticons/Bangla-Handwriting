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

def quality_assessment(quality_mod):
    quality_pred = quality_mod.predict(img)
    quality_index=(np.argmax(quality_pred))
    quality_switch(quality_index)
    #print(quality_pred)


#letter detection functions

def a():
    print("অ")
    global quality_model 
    quality_assessment(quality_model_a)
    
def b():
    print("আ")
    global quality_model 
    quality_assessment(quality_model_b)
    
def c():
    print("ই")
    global quality_model 
    quality_assessment(quality_model_c)
    
def d():
    print("ঈ")
    global quality_model 
    quality_assessment(quality_model_d)
    
def e():
    print("উ")
    global quality_model 
    quality_assessment(quality_model_e)
    
def f():
    print("ঊ")
    global quality_model 
    quality_assessment(quality_model_f)
    
def g():
    print("ঋ")
    global quality_model 
    quality_assessment(quality_model_g)
    
def h():
    print("এ")
    global quality_model 
    quality_assessment(quality_model_h)
    
def i():
    print("ঐ")
    global quality_model 
    quality_assessment(quality_model_i)
    
def j():
    print("ও")
    global quality_model 
    quality_assessment(quality_model_j)
    
def k():
    print("ঔ")
    global quality_model 
    quality_assessment(quality_model_k)    
def x():
    print("No latter found")


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
model = load_model("vgg16_model_letter.h5")
print("Letter model loaded")
global quality_model_a,quality_model_b,quality_model_c,quality_model_d,quality_model_e,quality_model_f,quality_model_g,quality_model_h,quality_model_i,quality_model_j,quality_model_k
quality_model_a = tf.keras.models.load_model("a_vgg16_model.h5")
quality_model_b = tf.keras.models.load_model("b_vgg16_model.h5")
quality_model_c = tf.keras.models.load_model("c_vgg16_model.h5")
quality_model_d = tf.keras.models.load_model("d_vgg16_model.h5")
quality_model_e = tf.keras.models.load_model("e_vgg16_model.h5")
quality_model_f = tf.keras.models.load_model("f_vgg16_model.h5")
quality_model_g = tf.keras.models.load_model("g_vgg16_model.h5")
quality_model_h = tf.keras.models.load_model("h_vgg16_model.h5")
quality_model_i = tf.keras.models.load_model("i_vgg16_model.h5")
quality_model_j = tf.keras.models.load_model("j_vgg16_model.h5")
quality_model_k = tf.keras.models.load_model("k_vgg16_model.h5")

print("All Loaded model from disk")
url = "http://192.168.43.72:4747" # Your url might be different, check the app
cam = cv2.VideoCapture(url+"/video")

#cam = cv2.VideoCapture(0)
start_y=90
start_x=170
height=300
width=300
while(True):
    _,frame=cam.read()
    img = frame[start_y:start_y+height, start_x:start_x+width]
    img_show = cv2.resize(frame, (224, 224)) 
    img=img_show
    img = img.reshape(-1,224,224,3)
    #frame=cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
    img = applications.vgg16.preprocess_input(img)
       
    
    
    prediction = model.predict(img)
    letter_index=(np.argmax(prediction))
    
    switch(letter_index)
    cv2.imshow("Video",img_show)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cam.release()
cv2.destroyAllWindows()

    