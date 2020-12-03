# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 13:36:41 2020

@author: Raiyaan Abdullah
"""

import matplotlib.pyplot as plt
import pandas as pd
import os

directory = "vgg16" 


for file in os.listdir(directory):
    df = pd.read_csv(os.path.join(directory,file))
    plt.figure(figsize=(10,6)) 
    
    print(df.columns) 
    
    x= df["epoch"]
    y= df["training acc."]
    z= df["val. acc."]
    # same plotting code as above!
    plt.plot(x,y,label="Training Accuracy")
    plt.plot(x,z,label="Validation Accuracy")
    plt.legend(loc='lower right')
    
    
    plt.xticks(range(1,len(x)+1))
    
    plt.xlabel('Epoch')
    
    plt.savefig("../../images/graphs/vgg16/"+file[0]+'_accuracy_quality.png')

