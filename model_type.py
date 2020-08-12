"""
Different CNNs which can be trained
"""

from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.layers import Dropout, Dense, Activation
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.optimizers import Adam
import config

config = config.Config()

class ModelSpec:
    def __init__(self, input_shape):
        self.input_shape = input_shape
    
    def get_conv_model(self):
        """
        Model
        """
        self.model = Sequential()
        self.model.add(Conv2D(16, kernel_size=(3,3), padding='same',
                              input_shape=self.input_shape))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(1, 2),padding='same'))        
        
        self.model.add(Conv2D(32, (3,3), activation='relu',padding='same'))
        self.model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))   

        self.model.add(Conv2D(32, (3,3), activation='relu',padding='same'))
        self.model.add(MaxPooling2D(pool_size=(2, 2),padding='same')) 
        
        self.model.add(Conv2D(64, (3,3), activation='relu',padding='same'))       
        self.model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
 
        self.model.add(Conv2D(64, (3,3), activation='relu',padding='same'))   
        self.model.add(Flatten())   
        # self.model.add(Dropout(0.5))
        
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dropout(0.5))
        
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dropout(0.5))
        
        optimizer = Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999)
        self.model.add(Dense(config.num_classes))#, activation='softmax'))
        self.model.add(Activation('sigmoid'))
        self.model.summary()
        self.model.compile(loss='categorical_crossentropy', optimizer=optimizer, 
                           metrics=['accuracy'])
        return self.model
