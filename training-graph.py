# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 13:36:41 2020

@author: Raiyaan Abdullah
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

df = pd.read_csv("D:/Github Projects/Bangla-Handwriting/images/graphs/vgg16-letter.csv")

sns.set()

print(df.columns) 
x= df["epoch"]
y= df["training acc."]
z= df["val. acc."]
# same plotting code as above!
plt.plot(x,y,label="Training Accuracy")
plt.plot(x,z,label="Validation Accuracy")
plt.legend(loc='lower right')

plt.xlabel('Epoch')


