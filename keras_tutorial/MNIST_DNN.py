'''Trains a simple deep NN on the MNIST dataset.
Gets to 98.40% test accuracy after 20 epochs
(there is *a lot* of margin for parameter tuning).
2 seconds per epoch on a K520 GPU.
'''

from __future__ import print_function

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop

batch_size = 128 # 32, 64, or 128 
num_classes = 10
epochs = 20 #larger， higher accurate. iteration, but there is a threshold of overfitting, resulting in high training accuracy and low test accuracy


# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary representation
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

###Build the model
model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(784,)))#  Relu , sigmoid, softmax,Tanh, Linear, or Step.
model.add(Dropout(0.2))# avoid overfitting, usually, 0.1,0.2,0.3
model.add(Dense(512, activation='relu'))#  Relu , sigmoid, softmax,Tanh, Linear, or Step.
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))#  Relu , sigmoid, softmax,Tanh, Linear, or Step.

model.summary()

model.compile(loss='categorical_crossentropy',  #Loss function，
              optimizer=RMSprop(),	#optimization function
              metrics=['accuracy']) # precision, or recall

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,# 1:show you an animated progress, 0 silent, 2 mention the number of epoch, like epoch 1/10 
                    validation_data=(x_test, y_test))
###Test
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])