# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 14:41:00 2020

@author: Raiyaan Abdullah
"""
import cv2 
import os 
import numpy as np 
from random import shuffle 
from tqdm import tqdm 
import tensorflow as tf
from tensorflow.keras import layers,Sequential,optimizers,applications, Model, applications, models

num_classes=44
batch_size = 4

data= np.load('augmented_data_mini.npy', allow_pickle=True)

print(np.shape(data))
'''Running the training and the testing in the dataset for our model'''

img_data = np.array([i[0] for i in data]).reshape(-1,224,224,3)
lbl_data = np.array([i[1] for i in data]).reshape(-1,44)

tr_img_data = img_data[:6200,:,:,:]
tr_lbl_data = lbl_data[:6200,:]

tst_img_data = img_data[6200:,:,:,:]
tst_lbl_data = lbl_data[6200:,:]

model= models.load_model('model.hdf5')
test_loss, test_acc = model.evaluate(tst_img_data,  tst_lbl_data, verbose=1)