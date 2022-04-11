# Python program to create 
# Image Classifier using CNN 
  
# Importing the required libraries 
import cv2 
import os 
import numpy as np 
from random import shuffle 
from tqdm import tqdm 
from sklearn.utils import shuffle

'''
We are changing our approach significantly. Now there will be one folder containing all the images. There will an array containing all file names, and another array containing all labels. These will be used in the CNN with a custom generator. We will not create a single npy dataset like before.
'''
  
'''Setting up the env'''
  
LOCATION = 'D:\Github Projects\Bangla-Handwriting\Datasets\Bangla_Handwriting_Dataset_Augmented_96K'

LR = 1e-3
  
  
  
'''Labelling the dataset'''
def label_img(word_label): 

    if   word_label == '0.6': return [1, 0, 0, 0] 
    elif word_label == '0.7': return [0, 1, 0, 0] 
    elif word_label == '0.8': return [0, 0, 1, 0] 
    elif word_label == '0.9': return [0, 0, 0, 1] 

filenames=[]
labels=[]
  
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
                label = label_img(qualityfolder) 
                filenames.append(img)
                labels.append(label)
                
                '''
                path = os.path.join(quality_path, img) 
                
                # loading the image from the path and then converting them into 
                # greyscale for easier covnet prob 
                img = cv2.imread(path, cv2.IMREAD_COLOR) 
          
                # resizing the image for processing them in the covnet 
                img = cv2.resize(img, (224, 224)) 
                
                print(path,' ',np.transpose(label))          
                data.append([np.array(img), np.array(label)]) 

                '''


    # shuffling of the training data to preserve the random state of our data 
    filenames_shuffled, labels_shuffled = shuffle(filenames, labels) 

    # saving our trained data for further uses if required 
    np.save('quality_data_96k_filenames.npy', filenames_shuffled) 
    np.save('quality_data_96k_labels.npy', labels_shuffled) 

create_data()


