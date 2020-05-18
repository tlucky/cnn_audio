"""
For prediction the sound event and linking the OPC UA Server.
"""

import pyaudio
import numpy as np
from tensorflow.keras.models import load_model
import pickle
from datetime import datetime
from librosa.feature import melspectrogram
# import librosa.core
import config
import server

from librosa.core import amplitude_to_db, power_to_db, stft
import librosa
import matplotlib.pyplot as plt
import librosa.display

def model_predict():
    """
    Prediction of the classes 'open' and 'close'
    """
    # global singal_old
    #  Create a numpy array of audio data

    signal = np.frombuffer(stream.read(int(config.new_len*0.75),
                                       exception_on_overflow = False),
                                       dtype=np.float32)
    # conf = np.zeros(int(14000*0.25), dtype=float)
    signal_buffer_old = config.signal_old
    config.signal_old = signal[7000:]
    signal_merged = np.append(signal_buffer_old, signal)
    
    # import random
    # from scipy.io import wavfile
    # import time
    # import os
    # time.sleep(2)
    # f = random.choice(os.listdir('clean_test/'))
    # print (f[0])
    # sample_rate, signal = wavfile.read('clean_test/'+f)

    mel = melspectrogram(y=signal_merged, 
                         sr=config.sample_rate, 
                         n_mels=config.n_mels, 
                         n_fft=config.n_fft, 
                         hop_length=config.hop_length, 
                         window=config.window) 

    X = librosa.amplitude_to_db(mel)

    X = X.reshape(1, X.shape[0], X.shape[1], 1)
    
    #  Prediction
    prediction = model.predict([X])    
    
    x_ax = np.arange(0,config.new_len)/config.sample_rate
    plt.figure(figsize=(12, 8))
    plt.subplot(2,1,1)
    plt.ylim(-0.4, 0.4)
    plt.xlim(0,max(x_ax))
    plt.plot(x_ax,signal_merged)

    plt.subplot(2,1,2)       
    librosa.display.specshow(librosa.amplitude_to_db(mel), sr=config.sample_rate, 
                             hop_length=config.hop_length,
                             x_axis='ms', y_axis='mel')
    # plt.colorbar(format='%+2.0f dB')
    plt.title('Acc open: ' +str(np.around(prediction[0][0],3)) 
              + '  |  Acc close: ' + str(np.around(prediction[0][1],3)))
    plt.clim(-65,-10)
    plt.show()
    
    # prediction[0] = np.around(prediction[0],decimals=4)
    print(prediction[0])
    # max_prediction = np.amax(prediction[0])
    if prediction[0][0] > config.threshold[0]:
        print('Acc: '+str(prediction[0][0])+'  Class: '+str(config.classes[0]))         
    if prediction[0][1] > config.threshold[1]:
        print('Acc: '+str(prediction[0][1])+'  Class: '+str(config.classes[1]))     

    return prediction[0][0], prediction[0][1]


#  Program
config = config.Config()    
if __name__ == "__main__":  

    #  Load CNN
    model = load_model('models/CNN_v3_32x64_sigmoid.h5')
    model.layers[0].input_shape
    
    #  Initializing PyAudio
    p=pyaudio.PyAudio()
    stream=p.open(format=pyaudio.paFloat32, channels=1, 
                  rate=config.sample_rate, input=True)
    config.signal_old = np.zeros(int(config.new_len*0.25), dtype=float) 
    open_merker = 0
    #  Start and define the OPC UA Server
    # opc_server = server.ServerClass()
    # opc_server.define_server()
    # print('Started prediction and OPC UA Server.')
    #  Start predicting loop
    while True:        
        
        open_pred, close_pred = model_predict()
      
        #  Load valve01 properties to get the cycles (open-close-open)
        #  Read number of cycles from pickle
        with open('pickles/valve01.p', 'rb') as handle:
            valve_prop = pickle.load(handle)
        if open_pred > config.threshold[0]:
            open_merker = 1
            valve_prop['status']='open'
            open_pred = np.round(open_pred, 3)
            valve_prop['accuracy'] = open_pred.astype(float)
        if (close_pred > config.threshold[1]) and (open_merker == 1):
            open_merker = 0
            valve_prop['cycles'] += 1
            valve_prop['updated'] = datetime.now()
            close_pred = np.round(close_pred, 3)
            valve_prop['accuracy'] = close_pred.astype(float)
            valve_prop['status']='close'
            print(valve_prop['cycles'])
        #  Update new number of clyces    
        with open('pickles/valve01.p', 'wb') as handle:
            pickle.dump(valve_prop, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        # #  Send data to OPC UA Server
        # opc_server.use_server(valve_prop['status'], valve_prop['updated'], 
        #                       valve_prop['accuracy'], valve_prop['cycles'])
     
    #  Close the audio stream
    stream.stop_stream()
    stream.close()
    p.terminate()
    
    #  Close OPC UA Server
    # opc_server.stop_server()
