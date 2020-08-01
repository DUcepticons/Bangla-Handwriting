# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 22:59:36 2020

@author: Riad
"""

import cv2 
import os 
import numpy as np 
from random import shuffle 
from tqdm import tqdm 
import tensorflow as tf
from tensorflow.keras import layers,Sequential,optimizers
'''Setting up the env'''
  
TRAIN_DIR = 'D:\Github Projects\Bangla-Handwriting\Datasets\Categorized-Dataset-with-Label\j'
TEST_DIR = 'Generated-Dataset/Test'

LR = 1e-3
  

  
'''Labelling the dataset'''
def label_img(subfolder): 

    # DIY One hot encoder 
    if   subfolder == '0.6': return [1, 0, 0, 0] 
    elif subfolder == '0.7': return [0, 1, 0, 0] 
    elif subfolder == '0.8': return [0, 0, 1, 0] 
    elif subfolder == '0.9': return [0, 0, 0, 1] 




  
'''Creating the training data'''
def create_train_data(): 
    # Creating an empty list where we should store the training data 
    # after a little preprocessing of the data 
    training_data = [] 
  
    # tqdm is only used for interactive loading 
    # loading the training data 
    for subfolder in os.listdir(TRAIN_DIR): 
        subfolder_path = os.path.join(TRAIN_DIR, subfolder) 
        for img in tqdm(os.listdir(subfolder_path)):   
            # labeling the images 
            label = label_img(subfolder) 

            path = os.path.join(subfolder_path, img) 
            # loading the image from the path and then converting them into 
            # greyscale for easier covnet prob 
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE) 
      
            # resizing the image for processing them in the covnet 
            img = cv2.resize(img, (350, 350)) 
      
            # final step-forming the training data list with numpy array of the images 
            training_data.append([np.array(img), np.array(label)]) 

  
    # shuffling of the training data to preserve the random state of our data 
    shuffle(training_data) 
  
    # saving our trained data for further uses if required 
    #np.save('train_data.npy', training_data) 
    return training_data 
  

  
'''Running the training and the testing in the dataset for our model'''
train_data = create_train_data() 
#test_data = process_test_data()
tr_img_data = np.array([i[0] for i in train_data]).reshape(-1,350,350,1)
tr_lbl_data = np.array([i[1] for i in train_data])


'''x_train=train_data[0]
y_train=train_data[1]
x_test=test_data[0]
y_test=test_data[1]'''

#print(x_train)

model =Sequential()
model.add(layers.InputLayer(input_shape=[350,350,1]))
model.add(layers.Conv2D(32, kernel_size=5, activation='relu'))
model.add(layers.MaxPooling2D(pool_size=5))
model.add(layers.Conv2D(64, kernel_size=5, activation='relu'))
model.add(layers.MaxPooling2D(pool_size=5))
model.add(layers.Conv2D(128, kernel_size=5, activation='relu'))
model.add(layers.MaxPooling2D(pool_size=5))
#model.add(layers.Dropout(0.25))
model.add(layers.Flatten())
model.add(layers.Dense(512,activation='relu'))
#model.add(layers.Dropout(0.5))
model.add(layers.Dense(256,activation='relu'))
model.add(layers.Dense(4,activation='softmax'))
optimizer=optimizers.Adam(lr=1e-3)
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(tr_img_data, tr_lbl_data, epochs=30)




model.summary()
model.save('model_test.h5')
print("Saved model to disk")
