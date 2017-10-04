"""
Author: Gananath R
DeepSpecta: Spatial and temporal classification of sequence data
Contact: https://github.com/gananath

#Powder Xrd data downloaded from http://rruff.info/
"""

import pandas as pd
import numpy as np
import math as m
import copy

from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Conv2D,MaxPooling2D
from keras.layers.normalization import BatchNormalization


def padwithtens(vector, pad_width, iaxis, kwargs):
    vector[:pad_width[0]] = 0
    vector[-pad_width[1]:] = 0
    return vector
    
def seq2tensor(data,width=None,height=None,pad=1):
    if isinstance(data, pd.DataFrame):
        data=data.values
    if width==None or height==None:
        #taking sqrt to make a nxn matrix
        width=m.sqrt(data.shape[0])+1
        height=m.sqrt(data.shape[0])+1        
    arr=[]
    for i,j in enumerate(data.T):
        d=copy.deepcopy(j)
        #resizing
        d.resize((width,height),refcheck=False)
        #padding the edges of matrix with zeros
        if pad!=0:
            d=np.lib.pad(d, pad, padwithtens)
        arr.append(d)
    del i,j,d
    return np.array(arr)

def cnn_model(epochs=250,lrate = 0.01):
    # Create the model
    model = Sequential()
    model.add(Conv2D(97, (2, 2), input_shape=(2, 97, 97), padding='same', activation='tanh', kernel_constraint=maxnorm(3)))
    model.add(BatchNormalization())
    model.add(Dropout(0.8))
    model.add(Conv2D(97, (2, 2),padding='same', activation='tanh', kernel_constraint=maxnorm(3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dropout(0.8))
    model.add(Dense(512, activation='tanh', kernel_constraint=maxnorm(3)))
    model.add(BatchNormalization())
    model.add(Dropout(0.8))
    model.add(Dense(7, activation='softmax'))
    # Compile model
    decay = lrate/epochs
    sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model
