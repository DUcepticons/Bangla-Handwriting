import cv2 
import os 
import numpy as np 
from random import shuffle 
from tqdm import tqdm 
import tensorflow as tf
from tensorflow.keras import layers,Sequential,optimizers,applications, Model, applications 
from tensorflow.keras.preprocessing import image

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        

        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")

    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

num_classes=12
batch_size = 16 #more means better faster convergence but takes more resources
train_data_num = 6000 #change it accordingly


data= np.load('data_letter.npy', allow_pickle=True)

print(np.shape(data))
'''Running the training and the testing in the dataset for our model'''

img_data = np.array([i[0] for i in data]).reshape(-1,224,224,3)
lbl_data = np.array([i[1] for i in data]).reshape(-1,num_classes)

tr_img_data = img_data[:train_data_num,:,:,:]
tr_lbl_data = lbl_data[:train_data_num,:]

tst_img_data = img_data[train_data_num:,:,:,:]
tst_lbl_data = lbl_data[train_data_num:,:]


#Code taken from: https://github.com/keras-team/keras/issues/9214
base_model = applications.ResNet50V2(weights='imagenet', include_top=False)
x = base_model.output
x = layers.GlobalMaxPooling2D()(x)
x = layers.Dense(256, activation='relu')(x)
predictions = layers.Dense(num_classes, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)
#model.summary()

for layer in base_model.layers:
    layer.trainable = False


optimizer=optimizers.Adam(learning_rate=1e-5)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

  
x_train = applications.resnet.preprocess_input(tr_img_data)
y_train = tr_lbl_data

'''
model.add(layers.InputLayer(input_shape=[224,224,3]))
model.add(layers.Conv2D(32, kernel_size=5, activation='relu'))
model.add(layers.MaxPooling2D(pool_size=3))
model.add(layers.Conv2D(64, kernel_size=5, activation='relu'))
model.add(layers.MaxPooling2D(pool_size=5))
model.add(layers.Conv2D(128, kernel_size=5, activation='relu'))
model.add(layers.MaxPooling2D(pool_size=5))
#model.add(layers.Dropout(0.25))
model.add(layers.Flatten())
model.add(layers.Dense(512,activation='relu'))
#model.add(layers.Dropout(0.5))
model.add(layers.Dense(256,activation='relu'))
'''

with tf.device('/gpu:0'):
   x_train = tf.convert_to_tensor(x_train, np.float32)
   y_train = tf.convert_to_tensor(y_train, np.float32)

#print(model.evaluate(x_train, y_train, batch_size=batch_size, verbose=1))
model.fit(x_train, y_train, epochs=2 , batch_size=batch_size, shuffle=False, 
          validation_split=0.1)


#unfreezing all layers and retraining with low learning rate
for layer in model.layers:
    layer.trainable = True

optimizer2=optimizers.Adam(learning_rate=5e-5)
model.compile(optimizer=optimizer2, loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5 , batch_size=batch_size, shuffle=False, 
          validation_split=0.1) #will try with 5 epochs later


print('Testing on unseen data:')
x_test = applications.resnet.preprocess_input(tst_img_data)
y_test = tst_lbl_data
test_loss, test_acc = model.evaluate(x_test,  y_test, verbose=1)


model.save('resnet50v2_model_letter.h5')

print("Saved model to disk")
