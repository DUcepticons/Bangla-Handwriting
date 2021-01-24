# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 13:36:41 2020

@author: Raiyaan Abdullah
"""

import matplotlib.pyplot as plt
import pandas as pd
import os

directory = "letter" 


for file in os.listdir(directory):
    df = pd.read_csv(os.path.join(directory,file))
    if file=="resnet50_letter.csv":
        plt.figure(figsize=(12,6)) 

    elif file=="vgg16_letter.csv":
        plt.figure(figsize=(10,6)) 
        
    else:
        plt.figure(figsize=(15,10))         
        
    print(df.columns) 
    
    x= df["epoch"]
    y= df["training loss"]
    z= df["val. loss"]
    # same plotting code as above!

    plt.plot(x,y,label="Training Loss")
    plt.plot(x,z,label="Validation Loss")
    plt.legend(loc='upper right')
        
    if file=="resnet50_letter.csv":        
        plt.xticks(range(len(x)+1))
    else:
        plt.xticks(range(1,len(x)+1))        
    plt.xlabel('Epoch')
    
    plt.savefig("../../images/graphs/letter/"+file[:-11]+'_loss_letter.png')

