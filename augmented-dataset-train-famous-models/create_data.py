import cv2 
import os 
import numpy as np 
from random import shuffle, random
from tqdm import tqdm 
import tensorflow as tf
from tensorflow.keras import layers,Sequential,optimizers
'''Setting up the env'''
  
LOCATION = '../Bangla Handwriting Dataset - Augmented Mini'


LR = 1e-3

letter_array=['a','b','c','d','e','f','g','h','i','j','k']
quality_array=['0.6','0.7','0.8','0.9']

'''Labelling the dataset'''
def label_img(letterfolder, qualityfolder): 
    label=np.zeros((44,1))
    letter_index= letter_array.index(letterfolder)
    quality_index= quality_array.index(qualityfolder)
    
    index= (letter_index*4)+quality_index
    
    label[index,0]=1
    
    return label

data = []
    
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
                label = label_img(letterfolder, qualityfolder) 
                
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