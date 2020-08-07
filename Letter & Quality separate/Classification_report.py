# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 02:06:18 2020

@author: akash
"""


# confusion matrix in sklearn
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import pandas as pd

df = pd.read_csv("a_resnet50_report.csv")

actualRaw = df["Input"].tolist()
actual = [str(i) for i in actualRaw]

predictedRaw = df["Output"].tolist()
predicted = [str(i) for i in predictedRaw]

labelsRaw = df["Labels"].dropna().tolist()
labels = [str(i) for i in labelsRaw]

# confusion matrix
matrix = confusion_matrix(actual,predicted, labels)
print('Confusion matrix : \n',matrix)


# classification report for precision, recall f1-score and accuracy
report = classification_report(actual,predicted,labels)
print('Classification report : \n',report)

'''
reportData = report.split()
start = report.split().index(labels[0])
end = report.split().index('accuracy')
PRF1 = reportData[start:end]
print(PRF1)
'''