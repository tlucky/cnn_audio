"""
For building the X, y for training the CNN model.
Also for training the CNN model. 
"""

import numpy as np
import pandas as pd
import os
import pickle
from tqdm import tqdm
from keras.utils import to_categorical
from scipy.io import wavfile
import matplotlib.pyplot as plt
from keras.callbacks import History
from keras.utils import plot_model
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from librosa.feature import melspectrogram
from scipy.signal import medfilt2d, wiener
import random
from librosa.core import amplitude_to_db, power_to_db

import config
import model_type
    
def work_status(begin_str):
    """
    For the definition of the different classes
    """
    if begin_str.startswith('O') == True:
        return (0)  # Open
    if begin_str.startswith('C') == True:
        return (1)  # Close
    if begin_str.startswith('N') == True:
        return (2)  # Error

def check_data():
    """
    Checks in the pickles folder for existing model. 
    If their is an existing file it returns X, y from the pickle folder
    """
    if os.path.isfile(config.p_path):
        print('Loading existing data for {} model'.format(config.mode))
        with open(config.p_path, 'rb') as handle:
            tmp = pickle.load(handle)
            return tmp
    else: 
        return None

def build_X_y():
    """
    Building X and y for the input and output of the CNN
    """   
    # tmp = check_data()
    # if tmp:
    #     return tmp.data[0], tmp.data[1]  # return X, y from the pickle folder

    X = []
    y = []
    for index, file in tqdm(enumerate(df['fname'])):
        # if file[0] == 'O' or file[0] == 'C':        
        sample_rate, signal = wavfile.read('clean/'+file)  # Read & 1.processing
        mel = melspectrogram(y=signal, 
                             sr=config.sample_rate, 
                             n_mels=config.n_mels, 
                             n_fft=config.n_fft, 
                             hop_length=config.hop_length, 
                             window=config.window)    
        
        S = amplitude_to_db(mel)   
        S[0] = (2*S.mean() + S[0])/3  # Reducing Noise
        S[1] = (S.mean() + 2*S[1])/3  # Reducing Noise
        
        random_int = random.randint(0,3)  # Radom state using different filters
        if random_int == 1:
            S = medfilt2d(S)
        if random_int == 2:  
            S = wiener(S)
        if random_int == 3: 
            S = S
            
        X.append(S) 
        fname=work_status(file)
        y.append(fname)     
    X = np.array(X)
    print(X.shape)
    X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
    y = np.array(y)
    y = to_categorical(y, num_classes=config.num_classes)    
    config.data = (X, y)
    with open(config.p_path, 'wb') as handle:
        pickle.dump(config, handle, protocol=2)        
    return X, y

def training_type(model_name, epochs=20, batch_size=512, cv=0):
    """
    Choose the CNN model, number of epochs, batch_size and the number of 
    crossvalidations. In case the cv = 0 or cv = 1 their is no cv applied.
    Only a split of X and y into training and test set.
    """   
    accuracy = []
    loss = []
    if (cv == 0) or (cv == 1):
        #  Split into training and test data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, 
                                                            shuffle=True, 
                                                            random_state=1)        
        model = model_name
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, 
                  verbose=1, callbacks=[history], shuffle=True)
        print('Model evaluation ',model.evaluate(X_test,y_test))
        accuracy.append(history.history['accuracy'])
        loss.append(history.history['loss'])            
    if cv > 1:
        #  Cross validation 
        #  https://androidkt.com/k-fold-cross-validation-with-tensorflow-keras/
        
        n_split = cv
        for train_index, test_index in KFold(n_split).split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]          
            model=model_name        
            model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, 
                      shuffle=True, verbose=1, callbacks=[history])
            print('Model evaluation ',model.evaluate(X_test,y_test))
            accuracy.append(history.history['accuracy'])
            loss.append(history.history['loss'])
            #  Results and grafic   
        avg_accuracy = np.mean([accuracy[i][-1] for i in range(len(accuracy))])
        avg_loss = np.mean([loss[i][-1] for i in range(len(loss))])
        avg_accuracy = np.around(avg_accuracy,decimals=3)
        avg_loss = np.around(avg_loss,decimals=3)
        print('Avg Loss: ' +str(avg_loss)+' Avg Accuracy: ' +str(avg_accuracy))
    return accuracy, loss


#  Program
if os.name == 'nt':  # Check if Windows is the OS
    os.environ["PATH"] += os.pathsep+'C:/Program Files (x86)/Graphviz2.38/bin/'
config = config.Config()
#  Importing data
df = pd.DataFrame(columns=['fname', 'label', 'length'],)  
df['fname'] = os.listdir('./clean/')
for index, row in df.iterrows():
    row['label'] = work_status(row['fname'])    
    rate, signal = wavfile.read('clean/'+row['fname'])
    row['length'] = signal.shape[0]# / rate

#  count the different labels and their distribution
classes = list(np.unique(df.label))
class_dist = df.groupby(['label'])['label'].count()/len(df)
prob_dist = class_dist / class_dist.sum()

#  Model
X, y  = build_X_y()
input_shape = (X.shape[1], X.shape[2], 1
               )
model_definition = model_type.ModelSpec(input_shape)

y_flat = np.argmax(y, axis=1)
history = History()

#  Choose the CNN model from the file model_type and save into folder models
model = model_definition.get_conv_model_2()
plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)

accuracy, loss = training_type(model, epochs = 70, cv=5)  # CV or not
model.save('models/CNN_v3_32x64_sigmoid.h5')  # For saving the CNN model

#  Print accuracy and loss of the CNNs
epochs = range(len(accuracy[0]))
for index, value in enumerate(accuracy):
    plt.plot(epochs, value, alpha = 0.5,
              label='Training accuracy no. ' + str(index+1).format(i=index))
for index, value in enumerate(loss):
    plt.plot(epochs, value, '--', alpha = 0.9,
              label='Training loss no. ' + str(index+1).format(i=index))
plt.title('Training accuracy and Training loss')
plt.legend()
plt.show()

