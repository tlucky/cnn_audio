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
import time

from librosa.core import amplitude_to_db, power_to_db, stft
import librosa
import matplotlib.pyplot as plt
import librosa.display

def signal_visualisation(prediction, mel, signal_merged):
    #  Plots
    x_ax = np.arange(0,config.new_len)/config.sample_rate*1000
    plt.style.use('ggplot')
    plt.figure(figsize=(3, 4),dpi=400)
    plt.rcParams.update({'font.size': 5})
    plt.subplot(2,1,1)
    plt.ylim(-0.4, 0.4)
    plt.xlim(0,max(x_ax))
    plt.ylabel('Amplitude')
    plt.plot(x_ax,signal_merged, color = 'black', linewidth=0.5)
    plt.subplot(2,1,2)       
    librosa.display.specshow(librosa.amplitude_to_db(mel), sr=config.sample_rate, 
                             hop_length=config.hop_length,
                             x_axis='ms', y_axis='mel')
    # plt.colorbar(format='%+2.0f dB')
    plt.title('Acc open: ' +str(np.around(prediction[0][0],3)) 
              + '  |  Acc close: ' + str(np.around(prediction[0][1],3)), y= 0.95)
    plt.clim(-90,22)
    plt.show()
    
    print(prediction[0])
    if prediction[0][0] > config.threshold[0]:
        print('Acc: '+str(prediction[0][0])+'  Class: '+str(config.classes[0]))         
    if prediction[0][1] > config.threshold[1]:
        print('Acc: '+str(prediction[0][1])+'  Class: '+str(config.classes[1]))     

    
def evaluation_using_random():
    """
    Using a random input file from the 'clean/' folder
    and returns the signal from this file
    """
    import random
    from scipy.io import wavfile
    import os
    time.sleep(2)
    f = random.choice(os.listdir('clean/'))
    print (f[0])
    sample_rate, signal_merged = wavfile.read('clean/'+f)
    return signal_merged

def model_predict():
    """
    Prediction of the classes 'open' and 'close'
    """
    #  Create a numpy array of audio data
    

    signal = np.frombuffer(stream.read(config.new_len,
                                       exception_on_overflow = False),
                                       dtype=np.float32)
    # signal_merged = np.append(config.signal_old, signal)
    # config.signal_old = signal_merged[signal_len:]
    signal_merged = signal
    #  Using random signal from folder 'clean' (turned off)
    # signal_merged = evaluation_using_random()
    
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
    # start_time = time.time()
    #  Show the prediction, signal and mel (turned off)
    signal_visualisation(prediction, mel, signal_merged)
    # print(str((time.time() - start_time)*1000)+'ms')
    return prediction[0][0], prediction[0][1]


#  Program
config = config.Config()    
if __name__ == "__main__":  
    
    signal_len = int(config.new_len*0.75)
    signal_len_buffer = config.new_len - signal_len
    
    #  Load CNN
    model = load_model('models/CNN.h5')
    model.layers[0].input_shape
    
    #  Initializing PyAudio
    p=pyaudio.PyAudio()
    stream=p.open(format=pyaudio.paFloat32, channels=1, 
                  rate=config.sample_rate, input=True)
    config.signal_old = np.zeros(signal_len_buffer, dtype=float) 
    open_merker = 0
    #  Start and define the OPC UA Server
    opc_server = server.ServerClass()
    opc_server.define_server()
    print('Started prediction and OPC UA Server.')
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
        
        # Send data to OPC UA Server
        opc_server.use_server(valve_prop['status'], valve_prop['updated'], 
                              valve_prop['accuracy'], valve_prop['cycles'])
        
    #  Close the audio stream
    stream.stop_stream()
    stream.close()
    p.terminate()
    
    #  Close OPC UA Server
    opc_server.stop_server()
