"""
Different CNNs which can be trained
"""

from keras.layers import Conv2D, MaxPool2D, Flatten
from keras.layers import LeakyReLU, MaxPooling2D
from keras.layers import Dropout, Dense
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers import Activation
from keras.optimizers import Adam
class ModelSpec:
    def __init__(self, input_shape):
        self.input_shape = input_shape
    
    def get_conv_model(self):
        """
        Acoustic event recognition using cochleagram image and convolutional 
        neural networks (Sharan und Moir 2019) 
        
        t~869us/step
        Accuracy: 0.95289737  Loss: 0.11410356166798624
        """
        self.model = Sequential()
        self.model.add(Conv2D(8, kernel_size=(7,7),activation='tanh',
                              padding='same',input_shape=self.input_shape))
        self.model.add(BatchNormalization()))
        self.model.add(MaxPooling2D(pool_size=(1, 2),padding='same'))        
        
        self.model.add(Conv2D(16, (5,5), activation='relu',padding='same'))
        self.model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
        self.model.add(Dropout(0.2))
        
        self.model.add(Conv2D(16, (3,3), activation='relu',padding='same'))
        self.model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
        self.model.add(Dropout(0.2))
        
        self.model.add(Conv2D(32, (3,3), activation='relu',padding='same'))       
        self.model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
        self.model.add(Dropout(0.2))
        
        self.model.add(Conv2D(32, (3,3), activation='relu',padding='same'))
             
        self.model.add(Flatten())    
        self.model.add(Dropout(0.2))
        self.model.add(Dense(64, activation='relu'))
        
        optimizer = Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999)
        self.model.add(Dense(3))#, activation='softmax'))
        self.model.add(Activation('sigmoid'))
        self.model.summary()
        self.model.compile(loss='categorical_crossentropy', optimizer=optimizer, 
                           metrics=['accuracy'])
        return self.model
    

