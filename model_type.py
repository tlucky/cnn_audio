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
        Convolutional Neural Network
        t = 6ms/step
        Accuracy: 0.9883833  Loss: 0.02879412667852015
        """
        self.model = Sequential()
        self.model.add(Conv2D(16, (3,3), activation='relu', strides=(1, 1),
                          padding='same',input_shape=self.input_shape))
        self.model.add(Conv2D(32, (3, 3), activation='relu', strides=(1,1),
                          padding='same'))
        self.model.add(MaxPool2D((2, 2)))
        self.model.add(Dropout(0.5))
        self.model.add(Conv2D(64, (3, 3), activation='relu', strides=(1,1),
                          padding='same'))
        self.model.add(MaxPool2D((2, 2)))
        self.model.add(Dropout(0.5))
        self.model.add(Conv2D(128, (3, 3), activation='relu', strides=(1,1),
                          padding='same'))
        self.model.add(MaxPool2D((2, 2)))
        self.model.add(Dropout(0.5))
        self.model.add(Flatten())
        # self.model.add(Dense(128, activation='relu'))
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dense(32, activation='relu'))
        self.model.add(Dense(3, activation='sigmoid'))
        self.model.summary()
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', 
                           metrics=['accuracy'])
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
        self.model.add(MaxPooling2D((1, 2),padding='same'))
        self.model.add(Dropout(0.25))
        self.model.add(Conv2D(64, (3, 3), activation='linear',padding='same'))
        self.model.add(LeakyReLU(alpha=0.1))
        self.model.add(MaxPooling2D(pool_size=(1, 2),padding='same'))
        self.model.add(Dropout(0.25))
        self.model.add(Conv2D(128, (3, 3), activation='linear',padding='same'))
        self.model.add(LeakyReLU(alpha=0.1))                  
        self.model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
        self.model.add(Dropout(0.4))
        self.model.add(Flatten())
        self.model.add(Dense(48, activation='linear'))
        self.model.add(LeakyReLU(alpha=0.1))           
        self.model.add(Dropout(0.3))
        self.model.add(Dense(3, activation='sigmoid'))
        self.model.summary()
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', 
                           metrics=['accuracy'])
        return self.model
    
    def get_conv_model_3(self):
        """
        Acoustic event recognition using cochleagram image and convolutional 
        neural networks (Sharan und Moir 2019) 
        
        t~869us/step
        Accuracy: 0.95289737  Loss: 0.11410356166798624
        """
        self.model = Sequential()
        self.model.add(Conv2D(32, kernel_size=(5,5),activation='linear',
                              padding='same',input_shape=self.input_shape))
        self.model.add(LeakyReLU(alpha=0.1))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D((4, 7),padding='same'))
        self.model.add(Dropout(0.3))
        
        self.model.add(Conv2D(64, (3,3), activation='linear',padding='same'))
        self.model.add(LeakyReLU(alpha=0.1))
        self.model.add(MaxPooling2D(pool_size=(4, 4),padding='same'))
        self.model.add(Dropout(0.3))
        
        # self.model.add(Conv2D(128, (3,3), activation='linear',padding='same'))
        # self.model.add(LeakyReLU(alpha=0.1))
        # self.model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
        # self.model.add(Dropout(0.3))
        
        # self.model.add(Conv2D(128, (3, 3), activation='linear',padding='same'))
        # self.model.add(LeakyReLU(alpha=0.1))                  
        # # self.model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
        # self.model.add(Dropout(0.5))
        
        self.model.add(Flatten())
        
        self.model.add(Dense(1024, activation='linear'))
        self.model.add(LeakyReLU(alpha=0.1))           
        self.model.add(Dropout(0.3))
        
        
        # self.model.add(Dense(256, activation='linear'))
        # self.model.add(LeakyReLU(alpha=0.1))           
        # self.model.add(Dropout(0.3))
        
        # self.model.add(Dense(128, activation='linear'))
        # self.model.add(LeakyReLU(alpha=0.1))           
        # self.model.add(Dropout(0.3))
        
        # self.model.add(Dense(48, activation='linear'))
        # self.model.add(LeakyReLU(alpha=0.1))           
        
        self.model.add(Dense(3, activation='sigmoid'))
        self.model.summary()
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', 
                           metrics=['accuracy'])
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
        self.model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',
                              padding='same',input_shape=self.input_shape))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D(pool_size=(1, 2),padding='same'))
        
        self.model.add(Conv2D(32, (3, 3), activation='relu',padding='same'))
        # self.model.add(MaxPooling2D(pool_size=(1, 2),padding='same'))
        self.model.add(Dropout(0.5))
        
        self.model.add(Conv2D(64, (3, 3), activation='relu',padding='same'))
        self.model.add(BatchNormalization())              
        self.model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
        self.model.add(Dropout(0.5))
        
        self.model.add(Conv2D(64, (3, 3), activation='relu',padding='same'))
        self.model.add(BatchNormalization())              
        self.model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
        self.model.add(Dropout(0.5))
        
        # self.model.add(Conv2D(64, (3, 3), activation='relu',padding='same'))              
        # self.model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
        # self.model.add(Dropout(0.25))
        # self.model.add(Conv2D(128, (3, 3), activation='relu',padding='same'))              
        # self.model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
      
        self.model.add(Flatten())
        
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dropout(0.5))      
        # self.model.add(Dense(192, activation='relu'))
        # self.model.add(Dropout(0.5)) 
        # self.model.add(Dense(50, activation='relu'))
        # self.model.add(Dropout(0.5)) 
        # self.model.add(Dense(32, activation='relu'))      
        # self.model.add(Dropout(0.5))  

        self.model.add(Dense(3, activation='sigmoid'))
        self.model.summary()
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', 
                           metrics=['accuracy'])
        return self.model
    
    def get_conv_model_5(self):
        """
        Acoustic event recognition using cochleagram image and convolutional 
        neural networks (Sharan und Moir 2019) 
        
        not working
        """
        self.model = Sequential()
        self.model.add(Conv2D(64, kernel_size=(5,5),activation='tanh',
                              padding='same',input_shape=self.input_shape))
        self.model.add(BatchNormalization())
        # self.model.add(LeakyReLU(alpha=0.1))

        self.model.add(MaxPooling2D((3, 3),padding='same'))
        # self.model.add(Dropout(0.25))
        
        self.model.add(Conv2D(64, (5, 5), activation='relu',padding='same'))
        self.model.add(BatchNormalization())
        # self.model.add(LeakyReLU(alpha=0.1))
        self.model.add(MaxPooling2D(pool_size=(3, 3),padding='same'))
        # self.model.add(Dropout(0.25))
        
        self.model.add(Conv2D(128, (3,3), activation='relu',padding='same'))
        self.model.add(BatchNormalization())
        # self.model.add(LeakyReLU(alpha=0.1))
        self.model.add(MaxPooling2D(pool_size=(4, 4),padding='same'))
        self.model.add(Dropout(0.25))
        
        # self.model.add(Conv2D(128, (3, 3), activation='linear',padding='same'))
        # self.model.add(LeakyReLU(alpha=0.1))                  
        # # self.model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
        # self.model.add(Dropout(0.5))
        
        self.model.add(Flatten())
        
        self.model.add(Dense(512, activation='relu'))
        # self.model.add(BatchNormalization())
        # self.model.add(LeakyReLU(alpha=0.1))           
        # self.model.add(Dropout(0.3))
        self.model.add(Dropout(0.25))
        
        # self.model.add(Dense(256, activation='relu'))
        # self.model.add(LeakyReLU(alpha=0.1))           
        # self.model.add(Dropout(0.3))
        
        # self.model.add(Dense(128, activation='relu'))
        # self.model.add(LeakyReLU(alpha=0.1))           
        # self.model.add(Dropout(0.3))
        
        # self.model.add(Dense(48, activation='linear'))
        # self.model.add(LeakyReLU(alpha=0.1))           
        optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)
        self.model.add(Dense(3))#, activation='softmax'))
        self.model.add(Activation('sigmoid'))
        # self.model.add(BatchNormalization())
        self.model.summary()
        self.model.compile(loss='categorical_crossentropy', optimizer=optimizer, 
                           metrics=['accuracy'])
        return self.model
    
    def get_conv_model_6(self):
        """
        Acoustic event recognition using cochleagram image and convolutional 
        neural networks (Sharan und Moir 2019) 
        
        t~869us/step
        Accuracy: 0.95289737  Loss: 0.11410356166798624
        """
        self.model = Sequential()
        self.model.add(Conv2D(8, kernel_size=(7,7),activation='tanh',
                              padding='same',input_shape=self.input_shape))
        self.model.add(BatchNormalization())
        # self.model.add(LeakyReLU(alpha=0.1))
        
        # self.model.add(Conv2D(8, (7,7), activation='relu',padding='same'))
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
    
    def get_conv_model_7(self):
        """
        Acoustic event recognition using cochleagram image and convolutional 
        neural networks (Sharan und Moir 2019) 
        
        t~869us/step
        Accuracy: 0.95289737  Loss: 0.11410356166798624
        """
        self.model = Sequential()
        
        self.model.add(Conv2D(32, kernel_size=(5,5),activation='tanh',
                              padding='same',input_shape=self.input_shape))
        self.model.add(BatchNormalization())      
        self.model.add(MaxPooling2D(pool_size=(2, 3),padding='same'))
        self.model.add(Dropout(0.2))
        
        self.model.add(BatchNormalization())
        self.model.add(Conv2D(32, (5,5), activation='relu',padding='same'))
        self.model.add(BatchNormalization())

        
        self.model.add(Conv2D(32, (3,3), activation='relu',padding='same'))        
        self.model.add(MaxPooling2D(pool_size=(2, 3),padding='same'))
        self.model.add(Dropout(0.2))
        

        self.model.add(BatchNormalization())
        
        self.model.add(Conv2D(64, (3,3), activation='relu',padding='same'))        
        self.model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
        self.model.add(Dropout(0.2))
        
        self.model.add(Flatten())
        

        self.model.add(Dense(128, activation='relu'))
        
        optimizer = Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999)
        self.model.add(Dense(3))#, activation='softmax'))
        self.model.add(Activation('sigmoid'))
        self.model.summary()
        self.model.compile(loss='categorical_crossentropy', optimizer=optimizer, 
                           metrics=['accuracy'])
        return self.model
    
    def get_conv_model_8(self):
        """
        Tak16
        """
        self.model = Sequential()
        
        
        self.model.add(Conv2D(32, kernel_size=(3,3),activation='relu',
                              padding='same',input_shape=self.input_shape))
        self.model.add(BatchNormalization())
        self.model.add(Conv2D(32, (3,3), activation='relu',padding='same')) 
        self.model.add(MaxPooling2D(pool_size=(1, 2),padding='same'))
        
        
        self.model.add(Conv2D(64, (3,3), activation='relu',padding='same'))
        self.model.add(BatchNormalization())
        # self.model.add(MaxPooling2D(pool_size=(2, 2),padding='same')) 
        self.model.add(Conv2D(64, (3,3), activation='relu',padding='same')) 
        self.model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))    


        # self.model.add(Conv2D(256, (3,3), activation='relu',padding='same')) 
        # self.model.add(Conv2D(256, (3,3), activation='relu',padding='same')) 
        # self.model.add(MaxPooling2D(pool_size=(2, 1),padding='same'))  

        
        # self.model.add(Dropout(0.5))
        
        self.model.add(Flatten())
        # self.model.add(BatchNormalization())

        self.model.add(Dense(512, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(BatchNormalization())
        self.model.add(Dense(512, activation='relu'))
        self.model.add(Dropout(0.5))
        
        optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)
        self.model.add(Dense(3))#, activation='softmax'))
        self.model.add(Activation('sigmoid'))
        self.model.summary()
        self.model.compile(loss='categorical_crossentropy', optimizer=optimizer, 
                           metrics=['accuracy'])
        return self.model
    
