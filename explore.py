import numpy as np
import pandas as pd
import os
import pickle
from tqdm import tqdm
from keras.utils import to_categorical
from scipy.io import wavfile
from scipy.fftpack import dct

# from keras.layers import Conv2D, MaxPool2D, Flatten
# from keras.layers import LeakyReLU, MaxPooling2D
# from keras.layers import Dropout, Dense, TimeDistributed
# from keras.models import Sequential
# from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
# from keras.callbacks import ModelCheckpoint
# from keras.callbacks import EarlyStopping
from keras.callbacks import History
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

import config_model
import signal_processing as sp   
    
def work_status(begin_str):
    """
    For the definition of the different classes
    """
    if begin_str.startswith('O') == True:
        return (0)  # Open
    if begin_str.startswith('C') == True:
        return (1)  # Close
    if begin_str.startswith('Q') == True:
        return (2)  # Error

# def visualisation

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
    tmp = check_data()
    if tmp:
        return tmp.data[0], tmp.data[1]  # return X, y from the pickle folder

    X = []
    _min, _max = float('inf'), -float('inf')
    for index, file in tqdm(enumerate(df['fname'])):        
        sample_rate, emphasized_signal = sp.read_wav(file)  # Read & 1. processing
        frames = sp.framing(sample_rate, emphasized_signal)  # Framing       
        pow_frames, mag_frames = sp.calc_stft(frames)  # Power and FFT      
        filter_banks = sp.calc_fbanks(sample_rate, pow_frames)  # Filter Banks       
        mfcc = sp.calc_mfcc(filter_banks)  # Mel-frequency Cepstral Coefficients (MFCCs)     
        _min = min(np.amin(mfcc), _min)
        _max = max(np.amin(mfcc), _max)
        X.append(mfcc)
    config.min = _min
    config.max = _max
    X = np.array(X)
    X = (X - _min) / (_max - _min)
    X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
    y = np.array([])
    y = to_categorical(df.label, num_classes=3)    
    config.data = (X, y)
    with open(config.p_path, 'wb') as handle:
        pickle.dump(config, handle, protocol=2)
        
    return X, y

def training_type(model_name, epochs=20, batch_size=32, cv=0):
    """
    Choose the CNN model, number of epochs, batch_size and the number of 
    crossvalidations. In case the cv = 0 or cv = 1 their is no cv applied.
    Only a split of X and y into training and test set.
    """   
    accuracy = []
    loss = []
    if (cv == 0) or (cv == 1):
        #  Split into training and test data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)        
        model = model_name
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, 
                  verbose=1, callbacks=[history])
        print('Model evaluation ',model.evaluate(X_test,y_test))
        accuracy.append(history.history['acc'])
        loss.append(history.history['loss'])            
    if cv > 1:
        #  Cross validation 
        #  https://androidkt.com/k-fold-cross-validation-with-tensorflow-keras/
        
        n_split = cv
        for train_index, test_index in KFold(n_split).split(X):
            X_train,X_test = X[train_index],X[test_index]
            y_train,y_test = y[train_index],y[test_index]          
            model=model_name        
            model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, 
                      verbose=1, callbacks=[history])
            print('Model evaluation ',model.evaluate(X_test,y_test))
            accuracy.append(history.history['acc'])
            loss.append(history.history['loss'])
            #  Results and grafic   
        avg_accuracy = np.mean([accuracy[i][-1] for i in range(len(accuracy))])
        avg_loss = np.mean([loss[i][-1] for i in range(len(loss))])
        print('Avg Accuracy: ' + str(avg_accuracy) + '  Avg Loss: ' + str(avg_loss))
    # save the model to disk
    filename = 'finalized_model.sav'
    pickle.dump(model, open(filename, 'wb'))
    return accuracy, loss


#  Program
config = config_model.Config()
#  Importing data
df = pd.DataFrame(columns=['fname', 'label', 'length'],)  
df['fname'] = os.listdir('./clean/')
for index, row in df.iterrows():
    row['label'] = work_status(row['fname'])    
    rate, signal = wavfile.read('clean/'+row['fname'])
    row['length'] = signal.shape[0] / rate

#  count the different labels and their distribution
classes = list(np.unique(df.label))
class_dist = df.groupby(['label'])['label'].count()/len(df)
prob_dist = class_dist / class_dist.sum()

#  Model
X, y = build_X_y()
input_shape = (X.shape[1], X.shape[2], 1)
model_definition = config_model.ModelSpec(input_shape)

#file = df['fname']
y_flat = np.argmax(y, axis=1)
history = History()

#  Choose the CNN model from the file config_model
conv_model = model_definition.get_conv_model_2()
accuracy, loss = training_type(conv_model, epochs = 10, cv=5)  # CV or not

#  Prints of the CNNs
epochs = range(len(accuracy[0]))
for index, value in enumerate(accuracy):
    plt.plot(epochs, value, alpha =0.5,
             label='Training accuracy no. '+str(index+1).format(i=index))
for index, value in enumerate(loss):
    plt.plot(epochs, value, '--', alpha =0.9,
             label='Training loss no. '+str(index+1).format(i=index))
plt.title('Training accuracy and Training loss')
plt.legend()
plt.show()




