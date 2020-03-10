import tensorflow as tf
import numpy as np

(x_train, y_train) = np.load('train_data.npy') 
(x_test, y_test) = np.load('test_data.npy')



print('Training data shape: ', x_train.shape)
print('Testing data shape : ', x_test.shape)