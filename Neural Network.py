import tensorflow as tf
import numpy as np

train_data = np.load('train_data.npy',allow_pickle = True) 
test_data = np.load('test_data.npy',allow_pickle = True)  
train_data = np.transpose(train_data)
test_data = np.transpose(test_data)
x_train = train_data[0]
y_train = train_data[1]
x_test  = test_data[0]
y_test  = test_data[1]


print('Training data shape: ', x_train.shape)
print('Testing data shape : ', x_test.shape)






# The known number of output classes.
num_classes = 11

# Input image dimensions
img_rows, img_cols = 350, 450

# Channels go last for TensorFlow backend
x_train_reshaped = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test_reshaped = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)

# Convert class vectors to binary class matrices. This uses 1 hot encoding.
y_train_binary = tf.keras.utils.to_categorical(y_train, num_classes)
y_test_binary = tf.keras.utils.to_categorical(y_test, num_classes)