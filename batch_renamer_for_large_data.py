# Python program to create 
# Image Classifier using CNN 
  
# Importing the required libraries 
import cv2 
import os 
import numpy as np 
from random import shuffle 
from tqdm import tqdm 
  
'''Setting up the env'''

LOCATION = 'D:\Github Projects\Bangla-Handwriting\Datasets\Bangla Handwriting Dataset - Augmented 96K'

'''Creating the training data'''
def rename_data(): 
    # Creating an empty list where we should store the training data 
    # after a little preprocessing of the data 
     
    
    # tqdm is only used for interactive loading 
    # loading the training data 
    for letterfolder in os.listdir(LOCATION): 
        
        letter_path = os.path.join(LOCATION, letterfolder) 
        

        for qualityfolder in os.listdir(letter_path):
            quality_path = os.path.join(letter_path, qualityfolder) 
            
            counter=1
            for img in tqdm(os.listdir(quality_path)):   
                
                new_location= quality_path+'\\'+str(letterfolder)+'-'+str(qualityfolder)+'-'+str(counter)+'.jpg'
                counter=counter+1
                os.rename( os.path.join(quality_path,img), new_location)
                
rename_data()
                
                