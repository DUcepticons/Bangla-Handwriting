import cv2 
import os 
import numpy as np 
from random import shuffle 
from tqdm import tqdm 
import datetime

#boilerplate for tensorflow 2.4+ GPU
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import tensorflow as tf

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

        
from tensorflow.keras import layers,Sequential,optimizers,applications, Model, applications 
from tensorflow.keras.preprocessing import image
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

from sklearn.model_selection import train_test_split


num_classes=4

#to view tensorboard data run in the folder : tensorboard --logdir logs/fit
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)




filenames= np.load('quality_data_96k_filenames.npy')
labels= np.load('quality_data_96k_labels.npy')

X_train_filenames, X_val_filenames, y_train, y_val = train_test_split(filenames, labels, test_size=0.2, random_state=1)

print(X_train_filenames.shape) 
print(y_train.shape)           

print(X_val_filenames.shape)   
print(y_val.shape)             

# You can save these files as well. As you will be using them later for training and validation of your model.
np.save('X_train_filenames.npy', X_train_filenames)
np.save('y_train.npy', y_train)

np.save('X_val_filenames.npy', X_val_filenames)
np.save('y_val.npy', y_val)


class My_Custom_Generator(tf.keras.utils.Sequence) :
  
  def __init__(self, image_filenames, labels, batch_size) :
    self.image_filenames = image_filenames
    self.labels = labels
    self.batch_size = batch_size
    
    
  def __len__(self) :
    return (np.ceil(len(self.image_filenames) / float(self.batch_size))).astype(np.int)
  
  
  def __getitem__(self, idx) :
    batch_x = self.image_filenames[idx * self.batch_size : (idx+1) * self.batch_size]
    batch_y = self.labels[idx * self.batch_size : (idx+1) * self.batch_size]
    
    return np.array([cv2.resize(cv2.imread('images-96k/' + str(file_name)), (224, 224))
               for file_name in batch_x])/255.0, np.array(batch_y)



#Code taken from: https://github.com/keras-team/keras/issues/9214
base_model = applications.ResNet50(weights='imagenet', include_top=False)
x = base_model.output
x = layers.GlobalMaxPooling2D()(x)
x = layers.Dense(512, activation='relu')(x)
x = layers.Dense(64, activation='relu')(x)
predictions = layers.Dense(num_classes, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)
#model.summary()

for layer in base_model.layers:
    layer.trainable = False


optimizer=optimizers.Adam(lr=1e-5)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

'''  
x_train = applications.resnet.preprocess_input(tr_img_data)
y_train = tr_lbl_data
'''


#use tensorboard --logdir logs/fit to see
callbacks=[ModelCheckpoint(filepath="checkpoints/resnet50_quality_predict.{epoch:02d}-{val_loss:.2f}.hdf5", monitor='val_loss', verbose=1, mode='auto', period=2),tensorboard_callback]
'''
#print(model.evaluate(x_train, y_train, batch_size=batch_size, verbose=1))
model.fit(x_train, y_train, epochs=2 , batch_size=batch_size, callbacks=callbacks, shuffle=False, 
          validation_split=0.1)
'''

#unfreezing all layers and retraining with low learning rate
for layer in model.layers:
    layer.trainable = True

optimizer2=optimizers.Adam(lr=5e-5)
model.compile(optimizer=optimizer2, loss='categorical_crossentropy', metrics=['accuracy'])


batch_size = 64

my_training_batch_generator = My_Custom_Generator(X_train_filenames, y_train, batch_size)
my_validation_batch_generator = My_Custom_Generator(X_val_filenames, y_val, batch_size)


model.fit_generator(generator=my_training_batch_generator, steps_per_epoch = int(77105 // batch_size), epochs=6 ,verbose=1,  callbacks=callbacks,  validation_data = my_validation_batch_generator, validation_steps = int(19277 // batch_size)) #will try with 5 epochs later

'''
print('Testing on unseen data:')
x_test = applications.resnet.preprocess_input(tst_img_data)
y_test = tst_lbl_data
test_loss, test_acc = model.evaluate(x_test,  y_test, verbose=1)
'''

model.save('resnet50_model_quality_one_large.h5')

print("Saved model to disk")
