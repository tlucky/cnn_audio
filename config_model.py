# import numpy as np
# import pandas as pd
import os
# from tqdm import tqdm

# import pickle

# from keras.utils import to_categorical
# from scipy.io import wavfile
# from scipy.fftpack import dct

from keras.layers import Conv2D, MaxPool2D, Flatten
from keras.layers import LeakyReLU, MaxPooling2D
from keras.layers import Dropout, Dense

from keras.models import Sequential
# from sklearn.utils.class_weight import compute_class_weight
# import matplotlib.pyplot as plt
# from keras.callbacks import ModelCheckpoint
# from keras.callbacks import EarlyStopping
# from keras.callbacks import History
# from sklearn.model_selection import train_test_split
# from sklearn.model_selection import KFold

class Config:
    def __init__(self, mode='conv', nfilt=40, nfeat=1, nfft=512, sample_rate=16000, 
                 low_freq_mel = 0, pre_emphasis = 0.97, frame_size = 0.025, 
                 new_len=13856, num_ceps = 12, cep_lifter = 22, freq_fft=10000):
        self.mode = mode
        self.nfilt = nfilt
        self.nfeat = nfeat
        self.nfft = nfft
        self.sample_rate = sample_rate
        self.low_freq_mel = low_freq_mel
        self.pre_emphasis = pre_emphasis
        self.frame_size = frame_size
        self.new_len = new_len

        self.num_ceps = num_ceps
        self.cep_lifter = cep_lifter
        self.freq_fft = freq_fft
        
        #self.step = int(rate/10)  # hier evtl das auch ändern
        self.model_path = os.path.join('models', mode + '.model')
        self.p_path = os.path.join('pickles', mode + '.p')
    def len_ms(self):
        return self.new_len / self.sample_rate * 1000 #ms

class ModelSpec:
    def __init__(self, input_shape):
        self.input_shape = input_shape
    
    def get_conv_model(self):
        """
        Convolutional Neural Network
        t = 6ms/step
        Accuracy: 0.9883833  Loss: 0.02879412667852015
        """
        self.model = Sequential()
        self.model.add(Conv2D(16, (3,3), activation='relu', strides=(1, 1),
                          padding='same',input_shape=self.input_shape))
        self.model.add(Conv2D(32, (3, 3), activation='relu', strides=(1,1),
                          padding='same'))
        self.model.add(Conv2D(64, (3, 3), activation='relu', strides=(1,1),
                          padding='same'))
        self.model.add(Conv2D(128, (3, 3), activation='relu', strides=(1,1),
                          padding='same'))
        self.model.add(MaxPool2D((2, 2)))
        self.model.add(Dropout(0.5))
        self.model.add(Flatten())
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dense(32, activation='relu'))
        self.model.add(Dense(3, activation='softmax'))
        self.model.summary()
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', 
                           metrics=['acc'])
        return self.model
    
    def get_conv_model_2(self):
        """
        Convolutional Neural Network
        https://www.datacamp.com/community/tutorials/convolutional-neural-networks-python
        t = 883us/step
        Accuracy: 0.9542019  Loss: 0.11913820796104951
        """
        self.model = Sequential()
        self.model.add(Conv2D(32, kernel_size=(3, 3),activation='linear',
                              padding='same',input_shape=self.input_shape))
        self.model.add(LeakyReLU(alpha=0.1))
        self.model.add(MaxPooling2D((2, 2),padding='same'))
        self.model.add(Dropout(0.25))
        self.model.add(Conv2D(64, (3, 3), activation='linear',padding='same'))
        self.model.add(LeakyReLU(alpha=0.1))
        self.model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
        self.model.add(Dropout(0.25))
        self.model.add(Conv2D(128, (3, 3), activation='linear',padding='same'))
        self.model.add(LeakyReLU(alpha=0.1))                  
        self.model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
        self.model.add(Dropout(0.4))
        self.model.add(Flatten())
        self.model.add(Dense(128, activation='linear'))
        self.model.add(LeakyReLU(alpha=0.1))           
        self.model.add(Dropout(0.3))
        self.model.add(Dense(3, activation='softmax'))
        self.model.summary()
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', 
                           metrics=['acc'])
        return self.model
    
    def get_conv_model_3(self):
        """
        Acoustic event recognition using cochleagram image and convolutional 
        neural networks (Sharan und Moir 2019) 
        
        t~869us/step
        Accuracy: 0.95289737  Loss: 0.11410356166798624
        """
        self.model = Sequential()
        self.model.add(Conv2D(32, kernel_size=(3, 3),activation='linear',
                              padding='same',input_shape=self.input_shape))
        self.model.add(LeakyReLU(alpha=0.1))
        self.model.add(MaxPooling2D((2, 2),padding='same'))
        self.model.add(Dropout(0.25))
        self.model.add(Conv2D(64, (3, 3), activation='linear',padding='same'))
        self.model.add(LeakyReLU(alpha=0.1))
        self.model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
        self.model.add(Dropout(0.25))
        
        self.model.add(Conv2D(128, (3, 3), activation='linear',padding='same'))
        self.model.add(LeakyReLU(alpha=0.1))                  
        self.model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
        self.model.add(Dropout(0.4))
        
        self.model.add(Flatten())
        
        self.model.add(Dense(128, activation='linear'))
        self.model.add(LeakyReLU(alpha=0.1))           
        
        self.model.add(Dense(64, activation='linear'))
        self.model.add(LeakyReLU(alpha=0.1))           
        
        self.model.add(Dense(3, activation='softmax'))
        self.model.summary()
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', 
                           metrics=['acc'])
        return self.model
    
    def get_conv_model_4(self):
        """
        Deep Convolutional Neural Networks and Data Augmentation for 
        Acoustic Event Detection (Takahashi et al. 2016)
        
        nicht ganz Nachbau möglich, wenig Infos
        t = 6-7ms/step
        Accuracy: 0.9509731  Loss: 0.1374367955702112
        """
        self.model = Sequential()
        self.model.add(Conv2D(64, kernel_size=(3, 3),activation='relu',
                              padding='same',input_shape=self.input_shape))
       
        self.model.add(Dropout(0.25))
        self.model.add(Conv2D(64, (3, 3), activation='relu',padding='same'))
    
        self.model.add(MaxPooling2D(pool_size=(1, 2),padding='same'))
       
        self.model.add(Conv2D(128, (3, 3), activation='relu',padding='same'))              
        self.model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
        
        self.model.add(Conv2D(128, (3, 3), activation='relu',padding='same'))              
        self.model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
      
        self.model.add(Flatten())
        
        self.model.add(Dense(1024, activation='relu'))
        self.model.add(Dropout(0.5))      
        
        self.model.add(Dense(1024, activation='relu'))
        self.model.add(Dropout(0.5)) 
        self.model.add(Dense(28, activation='relu'))      
        self.model.add(Dropout(0.5))  
        
        self.model.add(Dense(3, activation='softmax'))
        self.model.summary()
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', 
                           metrics=['acc'])
        return self.model
    
    def get_conv_model_5(self):
        """
        Convolutional Neural Network
        t = 4ms/step
        Accuracy: 0.9851621  Loss: 0.04522765875605145
        """
        self.model = Sequential()
        self.model.add(Conv2D(16, (3,3), activation='relu', strides=(1, 1),
                          padding='same',input_shape=self.input_shape))
        self.model.add(Dropout(0.25))
        self.model.add(Conv2D(32, (3, 3), activation='relu', strides=(1,1),
                          padding='same'))
        self.model.add(Conv2D(64, (3, 3), activation='relu', strides=(1,1),
                          padding='same'))
        # model.add(Conv2D(64, (3, 3), activation='relu', strides=(1,1),
        #                   padding='same'))
        self.model.add(MaxPool2D((2, 2)))
        self.model.add(Dropout(0.5))
        self.model.add(Flatten())
        #model.add(Dense(128, activation='relu'))
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dense(32, activation='relu'))
        self.model.add(Dense(3, activation='softmax'))
        self.model.summary()
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', 
                           metrics=['acc'])
        return self.model