import cv2 
import os 
import numpy as np 
from random import shuffle 
from tqdm import tqdm 
import tensorflow as tf
from tensorflow.keras import layers,Sequential,optimizers,applications, Model

data= np.load('data.npy', allow_pickle=True)

print(np.shape(data))
'''Running the training and the testing in the dataset for our model'''

img_data = np.array([i[0] for i in data]).reshape(-1,350,350,1)
lbl_data = np.array([i[1] for i in data]).reshape(-1,44)

tr_img_data = img_data[:20000,:,:,:]
tr_lbl_data = lbl_data[:20000,:]

tst_img_data = img_data[20000:,:,:,:]
tst_lbl_data = lbl_data[20000:,:]

model =Sequential()
model.add(layers.InputLayer(input_shape=[350,350,1]))
model.add(layers.Conv2D(32, kernel_size=5, activation='relu'))
model.add(layers.MaxPooling2D(pool_size=5))
model.add(layers.Conv2D(64, kernel_size=5, activation='relu'))
model.add(layers.MaxPooling2D(pool_size=5))
model.add(layers.Conv2D(128, kernel_size=5, activation='relu'))
model.add(layers.MaxPooling2D(pool_size=5))
#model.add(layers.Dropout(0.25))
model.add(layers.Flatten())
model.add(layers.Dense(512,activation='relu'))
#model.add(layers.Dropout(0.5))
model.add(layers.Dense(256,activation='relu'))
model.add(layers.Dense(44,activation='softmax'))
optimizer=optimizers.Adam(lr=1e-3)
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(tr_img_data, tr_lbl_data, epochs=30)

test_loss, test_acc = model.evaluate(tst_img_data,  tst_lbl_data, verbose=2)

model.summary()
model.save('model.hdf5')

print("Saved model to disk")

