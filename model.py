import numpy as np
import pandas as pd
import os
from tqdm import tqdm

import pickle

from keras.utils import to_categorical
from scipy.io import wavfile
from scipy.fftpack import dct

from keras.layers import Conv2D, MaxPool2D, Flatten
from keras.layers import LeakyReLU, MaxPooling2D
from keras.layers import Dropout, Dense, TimeDistributed
from keras.models import Sequential
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.callbacks import History
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

def get_conv_model():
    """
    Convolutional Neural Network
    t = 6ms/step
    Accuracy: 0.9883833  Loss: 0.02879412667852015
    """
    model = Sequential()
    model.add(Conv2D(16, (3,3), activation='relu', strides=(1, 1),
                      padding='same',input_shape=input_shape))
    model.add(Conv2D(32, (3, 3), activation='relu', strides=(1,1),
                      padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu', strides=(1,1),
                      padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu', strides=(1,1),
                      padding='same'))
    model.add(MaxPool2D((2, 2)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(3, activation='softmax'))
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
    return model

def get_conv_model_2():
    """
    Convolutional Neural Network
    https://www.datacamp.com/community/tutorials/convolutional-neural-networks-python
    t = 883us/step
    Accuracy: 0.9542019  Loss: 0.11913820796104951
    """
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),activation='linear',padding='same',input_shape=input_shape))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D((2, 2),padding='same'))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, (3, 3), activation='linear',padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
    model.add(Dropout(0.25))
    model.add(Conv2D(128, (3, 3), activation='linear',padding='same'))
    model.add(LeakyReLU(alpha=0.1))                  
    model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
    model.add(Dropout(0.4))
    model.add(Flatten())
    model.add(Dense(128, activation='linear'))
    model.add(LeakyReLU(alpha=0.1))           
    model.add(Dropout(0.3))
    model.add(Dense(3, activation='softmax'))
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
    return model

def get_conv_model_3():
    """
    Acoustic event recognition using cochleagram image and convolutional neural networks (Sharan und Moir 2019) 
    
    t~869us/step
    Accuracy: 0.95289737  Loss: 0.11410356166798624
    """
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),activation='linear',padding='same',input_shape=input_shape))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D((2, 2),padding='same'))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, (3, 3), activation='linear',padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(128, (3, 3), activation='linear',padding='same'))
    model.add(LeakyReLU(alpha=0.1))                  
    model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
    model.add(Dropout(0.4))
    
    model.add(Flatten())
    
    model.add(Dense(128, activation='linear'))
    model.add(LeakyReLU(alpha=0.1))           
    
    model.add(Dense(64, activation='linear'))
    model.add(LeakyReLU(alpha=0.1))           
    
    model.add(Dense(3, activation='softmax'))
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
    return model

def get_conv_model_4():
    """
    Deep Convolutional Neural Networks and Data Augmentation for 
    Acoustic Event Detection (Takahashi et al. 2016)
    
    nicht ganz Nachbau möglich, wenig Infos
    t = 6-7ms/step
    Accuracy: 0.9509731  Loss: 0.1374367955702112
    """
    model = Sequential()
    model.add(Conv2D(64, kernel_size=(3, 3),activation='relu',padding='same',input_shape=input_shape))
   
    model.add(Dropout(0.25))
    model.add(Conv2D(64, (3, 3), activation='relu',padding='same'))

    model.add(MaxPooling2D(pool_size=(1, 2),padding='same'))
   
    model.add(Conv2D(128, (3, 3), activation='relu',padding='same'))              
    model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
    
    model.add(Conv2D(128, (3, 3), activation='relu',padding='same'))              
    model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
  
    model.add(Flatten())
    
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))      
    
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5)) 
    model.add(Dense(28, activation='relu'))      
    model.add(Dropout(0.5))  
    
    model.add(Dense(3, activation='softmax'))
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
    return model

def get_conv_model_5():
    """
    Convolutional Neural Network
    t = 4ms/step
    Accuracy: 0.9851621  Loss: 0.04522765875605145
    """
    model = Sequential()
    model.add(Conv2D(16, (3,3), activation='relu', strides=(1, 1),
                      padding='same',input_shape=input_shape))
    model.add(Dropout(0.25))
    model.add(Conv2D(32, (3, 3), activation='relu', strides=(1,1),
                      padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu', strides=(1,1),
                      padding='same'))
    # model.add(Conv2D(64, (3, 3), activation='relu', strides=(1,1),
    #                   padding='same'))
    model.add(MaxPool2D((2, 2)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    #model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(3, activation='softmax'))
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
    return model