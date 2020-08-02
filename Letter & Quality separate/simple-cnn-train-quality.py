import cv2 
import os 
import numpy as np 
from random import shuffle 
from tqdm import tqdm 
import tensorflow as tf
from tensorflow.keras import layers,Sequential,optimizers,applications, Model, applications
from tensorflow.keras.preprocessing import image

np.random.seed(1)
tf.random.set_seed(1)

num_classes=4
batch_size = 8 #more means better faster convergence but takes more resources
train_data_num = 1900 #change it accordingly

LOCATION='D:\Github Projects\Bangla-Handwriting\Letter & Quality separate\quality-data'

for quality_data in os.listdir(LOCATION): 
    data= np.load(os.path.join(LOCATION,quality_data), allow_pickle=True)
    
    print(np.shape(data))
    '''Running the training and the testing in the dataset for our model'''
    
    img_data = np.array([i[0] for i in data]).reshape(-1,224,224,3)
    lbl_data = np.array([i[1] for i in data]).reshape(-1,num_classes)
    
    tr_img_data = img_data[:train_data_num,:,:,:]
    tr_lbl_data = lbl_data[:train_data_num,:]
    
    tst_img_data = img_data[train_data_num:,:,:,:]
    tst_lbl_data = lbl_data[train_data_num:,:]
    
    
    model =Sequential()
    model.add(layers.InputLayer(input_shape=[224,224,3]))
    model.add(layers.Conv2D(32, kernel_size=5, activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=4))
    model.add(layers.Conv2D(64, kernel_size=5, activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=4))
    model.add(layers.Conv2D(128, kernel_size=5, activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=5))
    #model.add(layers.Dropout(0.25))
    model.add(layers.Flatten())
    model.add(layers.Dense(512,activation='relu'))
    #model.add(layers.Dropout(0.5))
    model.add(layers.Dense(256,activation='relu'))
    model.add(layers.Dense(64,activation='relu'))
    model.add(layers.Dense(num_classes,activation='softmax'))
    #model.summary()
    
    optimizer=optimizers.Adam(lr=1e-4)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        
    x_train = tr_img_data
    y_train = tr_lbl_data
    

    print(quality_data[0])
    print(model.evaluate(x_train, y_train, batch_size=batch_size, verbose=1))
    model.fit(x_train, y_train, epochs=20 , batch_size=batch_size, shuffle=False, 
              validation_split=0.1)
    
 


  
    
    print('Testing on unseen data:')
    x_test = tst_img_data
    y_test = tst_lbl_data
    test_loss, test_acc = model.evaluate(x_test,  y_test, verbose=1)

    
    
    model.save(quality_data[0]+'_simple_model.h5')
    del model
    
    print("Saved model to disk")
