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

LOCATION = 'generate'

# Creating an empty list where we should store the training data 
# after a little preprocessing of the data 
data = [] 




p = Augmentor.Pipeline(source_directory=LOCATION, output_directory='')
#p.rotate(probability=1, max_left_rotation=22, max_right_rotation=25)
#p.zoom(probability=1, min_factor=1.6, max_factor=1.8)
#p.random_brightness(probability=1,min_factor=1.6,max_factor=2)
p.random_contrast(probability=0.9,min_factor=1.8,max_factor=2)

p.sample(1)
