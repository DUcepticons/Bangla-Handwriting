# Python program to create 
# Image Classifier using CNN 
  
# Importing the required libraries 
import cv2 
import os 
import numpy as np 
from random import shuffle 
from tqdm import tqdm 
  
'''Setting up the env'''
  
LOCATION = 'D:\Github Projects\Bangla-Handwriting\Datasets\Bangla Handwriting Dataset - Augmented Mini'

LR = 1e-3
  
  
  
'''Labelling the dataset'''
def label_img(word_label): 

    if   word_label == 'a': return [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] 
    elif word_label == 'b': return [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] 
    elif word_label == 'c': return [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0] 
    elif word_label == 'd': return [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0] 
    elif word_label == 'e': return [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0] 
    elif word_label == 'f': return [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0] 
    elif word_label == 'g': return [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0] 
    elif word_label == 'h': return [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0] 
    elif word_label == 'i': return [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0] 
    elif word_label == 'j': return [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0] 
    elif word_label == 'k': return [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0] 
    elif word_label == 'x': return [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1] 
    
data=[]
  
'''Creating the training data'''
def create_data(): 
    # Creating an empty list where we should store the training data 
    # after a little preprocessing of the data 
     
    
    # tqdm is only used for interactive loading 
    # loading the training data 
    for letterfolder in os.listdir(LOCATION): 
        
        letter_path = os.path.join(LOCATION, letterfolder) 
        for qualityfolder in os.listdir(letter_path):
            quality_path = os.path.join(letter_path, qualityfolder) 

            for img in tqdm(os.listdir(quality_path)):   
                # labeling the images 
                label = label_img(letterfolder) 
                
                path = os.path.join(quality_path, img) 
                
                # loading the image from the path and then converting them into 
                # greyscale for easier covnet prob 
                img = cv2.imread(path, cv2.IMREAD_COLOR) 
          
                # resizing the image for processing them in the covnet 
                img = cv2.resize(img, (224, 224)) 
                
                print(path,' ',np.transpose(label))          
                data.append([np.array(img), np.array(label)]) 



  
    # shuffling of the training data to preserve the random state of our data 
    shuffle(data) 
  
    # saving our trained data for further uses if required 
    np.save('data.npy', data) 
    return data 

create_data()


