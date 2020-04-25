import cv2 
import os 
import csv
import numpy as np 
from random import shuffle 
from tqdm import tqdm 
import tensorflow as tf
from tensorflow.keras import layers,Sequential,optimizers


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
    quality_pred = quality_model.predict(check)
    quality_index=(np.argmax(quality_pred))
    quality_switch(quality_index)
    print(quality_pred)


#letter detection functions

def a():
    print("অ")
    global quality_model 
    quality_model = tf.keras.models.load_model("letter-models/a.hdf5")
    quality_assessment()
    
def b():
    print("আ")
    global quality_model 
    quality_model = tf.keras.models.load_model("letter-models/b.hdf5")
    quality_assessment()
    
def c():
    print("ই")
    global quality_model 
    quality_model = tf.keras.models.load_model("letter-models/c.hdf5")
    quality_assessment()
    
def d():
    print("ঈ")
    global quality_model 
    quality_model = tf.keras.models.load_model("letter-models/d.hdf5")
    quality_assessment()
    
def e():
    print("উ")
    global quality_model 
    quality_model = tf.keras.models.load_model("letter-models/e.hdf5")
    quality_assessment()
    
def f():
    print("ঊ")
    global quality_model 
    quality_model = tf.keras.models.load_model("f.hdf5")
    quality_assessment()
    
def g():
    print("ঋ")
    global quality_model 
    quality_model = tf.keras.models.load_model("letter-models/g.hdf5")
    quality_assessment()
    
def h():
    print("এ")
    global quality_model 
    quality_model = tf.keras.models.load_model("letter-models/h.hdf5")
    quality_assessment()
    
def i():
    print("ঐ")
    global quality_model 
    quality_model = tf.keras.models.load_model("letter-models/i.hdf5")
    quality_assessment()
    
def j():
    print("ও")
    global quality_model 
    quality_model = tf.keras.models.load_model("letter-models/j.hdf5")
    quality_assessment()
    
def k():
    print("ঔ")
    global quality_model 
    quality_model = tf.keras.models.load_model("letter-models/k.hdf5")
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
    
path= 'Generated-Dataset/Train/g-1-39.jpg'

# loading the image from the path and then converting them into 
# greyscale for easier covnet prob 
img = cv2.imread(path, cv2.IMREAD_GRAYSCALE) 
  
# resizing the image for processing them in the covnet 
img = cv2.resize(img, (350, 350)) 
         
check = np.array(img).reshape(-1,350,350,1)
   
model = tf.keras.models.load_model("bangla_model.hdf5")
prediction = model.predict(check)
letter_index=(np.argmax(prediction))

switch(letter_index)
print(prediction)





