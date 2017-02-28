
# coding: utf-8

# In[5]:

import numpy as np
from matplotlib import pyplot as plt

import pandas as pd
import keras
from keras.layers import Dense, Activation
from keras.models import Sequential
from keras.utils import np_utils


# In[6]:

ds = pd.read_csv('./train.csv')
data = ds.values

print data.shape


# In[7]:

data_length = data.shape[0] / 2
split = int(data_length*0.75)
X_train = data[:split, 1:]
X_val = data[split:data_length, 1:]
y_train = np_utils.to_categorical(data[:split, 0])            # To form one-hot vectors for each output
y_val = np_utils.to_categorical(data[split:data_length, 0])

print X_train.shape, y_train.shape
print X_val.shape, y_val.shape


# In[8]:

# Keras model

model = Sequential()

model.add(Dense(350, input_shape=(784,)))
model.add(Activation('tanh'))

model.add(Dense(10)) 
model.add(Activation('softmax'))

model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[11]:

hist = model.fit(X_train, y_train, nb_epoch = 50, shuffle=True, batch_size = 100, validation_data = (X_val, y_val))


# In[ ]:

plt.figure(0)
plt.plot(hist.history['loss'], 'b')
plt.plot(hist.history['val_loss'], 'r')

plt.figure(1)
plt.plot(hist.history['acc'], 'b')
plt.plot(hist.history['val_acc'], 'r')
plt.show()

