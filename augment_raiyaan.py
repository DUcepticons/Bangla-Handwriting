# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 03:23:27 2020

@author: Raiyaan Abdullah
"""

import Augmentor
import cv2 
import os 
import numpy as np 
from random import shuffle 
from tqdm import tqdm 

LOCATION = 'D:\Github Projects\Bangla-Handwriting\Datasets\Bangla Handwriting Dataset - Augmented 96K'

# Creating an empty list where we should store the training data 
# after a little preprocessing of the data 
data = [] 

# tqdm is only used for interactive loading 
# loading the training data 
for letterfolder in os.listdir(LOCATION): 
    
    letter_path = os.path.join(LOCATION, letterfolder) 
    for qualityfolder in os.listdir(letter_path):
        quality_path = os.path.join(letter_path, qualityfolder) 


        p = Augmentor.Pipeline(source_directory=quality_path, output_directory='')
        p.rotate(probability=0.8, max_left_rotation=10, max_right_rotation=10)
        p.zoom(probability=0.2, min_factor=1.1, max_factor=1.5)
        p.random_brightness(probability=0.9,min_factor=0.8,max_factor=1.3)
        p.random_contrast(probability=0.9,min_factor=0.5,max_factor=1.5)
        
        p.sample(2000)
