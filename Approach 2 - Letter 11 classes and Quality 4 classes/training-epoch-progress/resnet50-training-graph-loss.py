# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 13:36:41 2020

@author: Raiyaan Abdullah
"""

import matplotlib.pyplot as plt
import pandas as pd
import os

directory = "resnet50" 


for file in os.listdir(directory):
    df = pd.read_csv(os.path.join(directory,file))
    plt.figure(figsize=(12,6)) 
    
    print(df.columns) 
    
    x= df["epoch"]
    y= df["training loss"]
    z= df["val. loss"]
    # same plotting code as above!
    plt.plot(x,y,label="Training Loss")
    plt.plot(x,z,label="Validation Loss")
    plt.legend(loc='upper right')
    
    
    plt.xticks(range(len(x)+1))
    
    plt.xlabel('Epoch')
    
    plt.savefig("../../images/graphs/resnet50/"+file[0]+'_loss_quality.png')

