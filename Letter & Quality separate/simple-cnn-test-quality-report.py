import cv2 
import os 
import numpy as np 
from random import shuffle 
from tqdm import tqdm 
import tensorflow as tf
from tensorflow.keras import layers,Sequential,optimizers,applications, Model, applications 
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import model_from_json, load_model
import csv

num_classes=4
batch_size = 8 #more means better faster convergence but takes more resources
train_data_num = 1900 #change it accordingly

quality_array=[0.6,0.7,0.8,0.9]

LOCATION='D:\Github Projects\Bangla-Handwriting\Letter & Quality separate\quality-data'

for quality_data in os.listdir(LOCATION): 
    
    
    
    data= np.load(os.path.join(LOCATION,quality_data), allow_pickle=True)
    
    print(np.shape(data))
    '''Running the training and the testing in the dataset for our model'''
    
    img_data = np.array([i[0] for i in data]).reshape(-1,224,224,3)
    lbl_data = np.array([i[1] for i in data]).reshape(-1,num_classes)
    
    
    tst_img_data = img_data[train_data_num:,:,:,:]
    tst_lbl_data = lbl_data[train_data_num:,:]
    
    
    model = load_model('letter-models/simple/'+quality_data[0]+'_simple_model.h5')

    print('Testing on unseen data:')
    x_test = tst_img_data
    y_test = tst_lbl_data

    with open('reports/simple/'+quality_data[0]+'_simple_report.csv',mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',')
        csv_writer.writerow(['Input', 'Output','Labels'])
    
        prediction = model.predict(x_test)    
        for i in range(len(y_test)):
            
            quality_label=quality_array[np.argmax(y_test[i])]
            quality_predict=quality_array[np.argmax(prediction[i])]
        
            if i<num_classes:
                csv_writer.writerow([quality_label, quality_predict,quality_array[i]])
            else:
                csv_writer.writerow([quality_label, quality_predict])        
 

    del model
    

