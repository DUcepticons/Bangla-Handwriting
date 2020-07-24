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
  
TRAIN_DIR = 'Generated-Dataset/Train'
TEST_DIR = 'Generated-Dataset/Test'

LR = 1e-3
  
  
  
'''Labelling the dataset'''
def label_img(img): 
    word_label = img.split('-')[0] 
    # DIY One hot encoder 
    if   word_label == 'a': return [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] 
    elif word_label == 'b': return [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0] 
    elif word_label == 'c': return [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0] 
    elif word_label == 'd': return [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0] 
    elif word_label == 'e': return [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0] 
    elif word_label == 'f': return [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0] 
    elif word_label == 'g': return [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0] 
    elif word_label == 'h': return [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0] 
    elif word_label == 'i': return [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0] 
    elif word_label == 'j': return [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0] 
    elif word_label == 'k': return [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1] 
    
  
'''Creating the training data'''
def create_train_data(): 
    # Creating an empty list where we should store the training data 
    # after a little preprocessing of the data 
    training_data = [] 
  
    # tqdm is only used for interactive loading 
    # loading the training data 
    for img in tqdm(os.listdir(TRAIN_DIR)): 
  
        # labeling the images 
        label = label_img(img) 
  
        path = os.path.join(TRAIN_DIR, img) 
  
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
  
'''Processing the given test data'''
# Almost same as processing the training data but 
# we dont have to label it. 


def process_test_data(): 
    test_num_counter=0
    testing_data = [] 
    for img in tqdm(os.listdir(TEST_DIR)): 
        path = os.path.join(TEST_DIR, img) 
       
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE) 
        img = cv2.resize(img, (350, 350)) 
        testing_data.append([np.array(img), test_num_counter]) 
        test_num_counter += 1
          
    shuffle(testing_data) 
    #np.save('test_data.npy', testing_data) 
    return testing_data 
  
'''Running the training and the testing in the dataset for our model'''
train_data = create_train_data() 
test_data = process_test_data()
tr_img_data = np.array([i[0] for i in train_data]).reshape(-1,350,350,1)
tr_lbl_data = np.array([i[1] for i in train_data])

tst_img_data = np.array([i[0] for i in test_data]).reshape(-1,350,350,1)
tst_lbl_data = np.array([i[1] for i in test_data])
'''x_train=train_data[0]
y_train=train_data[1]
x_test=test_data[0]
y_test=test_data[1]'''
#test_labels = to_categorical(tst_lbl_data, num_classes=11)
#print(x_train)
#print(x_test)
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
model.add(layers.Dense(11,activation='softmax'))
optimizer=optimizers.Adam(lr=1e-3)
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(tr_img_data, tr_lbl_data, epochs=40)


#test_loss, test_acc = model.evaluate(tst_img_data,  tst_lbl_data, verbose=2)

model.summary()
model.save('bangla_model.hdf5')
print("Saved model to disk")
