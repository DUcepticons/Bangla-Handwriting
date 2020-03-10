import tensorflow as tf
import numpy as np

train_data = np.load('train_data.npy') 
test_data = np.load('test_data.npy')  
(x_train, y_train) = (train_data[0],train_data[1])
(x_test, y_test) = (test_data[0],test_data[1])

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

print('Training data shape: ', x_train.shape)
print('Testing data shape : ', x_test.shape)