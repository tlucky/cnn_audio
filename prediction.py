"""
For prediction the sound event and linking the OPC UA Server.
"""

import pyaudio
import numpy as np
from tensorflow.keras.models import load_model
import pickle
from datetime import datetime
from librosa.feature import melspectrogram
import librosa.core
import config
import server

import matplotlib.pyplot as plt
import librosa.display
from scipy.io import wavfile

# def analysing():  
    #  Analysing data
    # import matplotlib.pyplot as plt
    # y_prob=np.array(y_prob)
    
    # plt.plot(y_prob[:,0],color='green',label=classes[0])
    # plt.plot(y_prob[:,1],color='red',label=classes[1])
    # plt.plot(y_prob[:,2],color='black',label=classes[2],alpha=0.7)
    # plt.plot(np.ones(len(y_prob))*config.threshold[0],color='black',
    #          linestyle='--', label = 'Grenzwert: ' + classes[0])
    # plt.plot(np.ones(len(y_prob))*config.threshold[1],color='black',
    #          linestyle='--', label = 'Grenzwert: ' + classes[1])
    # # plt.ylim(0,0.5)
    # plt.legend()
    # plt.show()


def model_predict():
    """
    Prediction of the classes 'open' and 'close'
    """
    
    #  Create a numpy array of audio data
    signal = np.frombuffer(stream.read(int(config.new_len),
                                       exception_on_overflow = False),
                                       dtype=np.float32)
    
    # f = 'C015_110.WAV'
    # f = 'CQ1117_102.WAV'
    # f = 'O1910_111.WAV'
    # sample_rate, signal = wavfile.read('clean/'+f)
    
    mel = melspectrogram(y=signal, sr=config.sample_rate, n_mels=49, n_fft=220, 
                              hop_length=110, window='hann')
    # mel = mel[2:]
    # mel = melspectrogram(y=signal, sr=16000, n_mels=32, n_fft=256, 
    #                       hop_length=128, window='hann')
    X = librosa.amplitude_to_db(mel)
    plt.imshow(X)
    plt.show()
    X = X.reshape(1, X.shape[0], X.shape[1], 1)
    
    #  Prediction
    prediction = model.predict([X])    
    
    # prediction[0] = np.around(prediction[0],decimals=4)
    print(prediction[0])
    # max_prediction = np.amax(prediction[0])
    if prediction[0][0] > config.threshold[0]:
        print('Acc: '+str(prediction[0][0])+'  Class: '+str(config.classes[0]))         
    if prediction[0][1] > config.threshold[1]:
        print('Acc: '+str(prediction[0][1])+'  Class: '+str(config.classes[1]))     
    # y_prob.append(prediction[0])
    return prediction[0][0], prediction[0][1]


#  Program
config = config.Config()    
if __name__ == "__main__":  

    #  Load CNN
    model = load_model('models/CCN_8_49.h5')
    model.layers[0].input_shape
    
    #  Initializing PyAudio
    p=pyaudio.PyAudio()
    stream=p.open(format=pyaudio.paFloat32,channels=1, 
                  rate=config.sample_rate, input=True)
     
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
