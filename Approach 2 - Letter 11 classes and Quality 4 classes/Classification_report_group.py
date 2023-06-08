# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 02:06:18 2020

@author: akash
"""


# confusion matrix in sklearn
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import pandas as pd
import os

LOCATION='D:/Github Projects/Bangla-Handwriting/Letter & Quality separate/reports/'

for cnnfolder in  os.listdir(LOCATION): 
    cnn_type_path = os.path.join(LOCATION, cnnfolder) 
    
    for file in os.listdir(cnn_type_path):
        file_path= os.path.join(cnn_type_path, file) 
        
        df = pd.read_csv(file_path)
        
        actualRaw = df["Input"].tolist()
        actual = [str(i) for i in actualRaw]
        
        predictedRaw = df["Output"].tolist()
        predicted = [str(i) for i in predictedRaw]
        
        labelsRaw = df["Labels"].dropna().tolist()
        labels = [str(i) for i in labelsRaw]
        
        print(file)
        # confusion matrix
        matrix = confusion_matrix(actual,predicted, labels)
        print('Confusion matrix : \n',matrix)
        
        
        # classification report for precision, recall f1-score and accuracy
        report = classification_report(actual,predicted,labels)
        print('Classification report : \n',report)

