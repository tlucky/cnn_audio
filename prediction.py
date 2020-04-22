import pyaudio
import numpy as np
from tensorflow.keras.models import load_model

import signal_processing as sp
import config

config = config.Config()
sample_length = config.new_len # number of data points to read at a time
sample_rate = config.sample_rate # time resolution of the recording device (Hz) 
   
classes = np.array(['open', 'close','noise'])

#  Load CNN
model = load_model('models/CCN_v2_16k_3.h5')
model.layers[0].input_shape

# Threshold for the classes
threshold = np.array([0.8, 0.7])

#  Initializing PyAudio
p=pyaudio.PyAudio()
stream=p.open(format=pyaudio.paFloat32,channels=1, rate=sample_rate,input=True)

overlap = 0.5
signal_old = np.zeros(int(sample_length * overlap))
signal=[]
y_prob=[]

while True:
    #  Create a numpy array of audio data
    signal = np.frombuffer(stream.read(int(sample_length*overlap),
                                       exception_on_overflow = False),
                                       dtype=np.float32)
    signal = signal-0.02  # For adjustment of signal

    #  Overlapping
    signal_overlap = np.append(signal_old, signal)
    signal_old = signal  # for next cycle
    # signal_overlap = signal

    #  Preprocessing
    frames = sp.framing(sample_rate, signal_overlap)  # Framing       
    pow_frames, mag_frames = sp.calc_stft(frames)  # Power and FFT      
    filter_banks = sp.calc_fbanks(sample_rate, pow_frames)  # Filter Banks       
    mfcc = sp.calc_mfcc(filter_banks)  # Mel-frequency Cepstral Coefficients
    X = mfcc
    with open('models/minmax.txt', 'r') as text_file:
        c_min, c_max = text_file.read().split(',')
    c_min, c_max = float(c_min), float(c_max)
    X = (X - c_min) / (-c_max - c_min)
    X = X.reshape(1, X.shape[0], X.shape[1], 1)
    
    #  Prediction
    prediction = model.predict([X])    
    print(np.around(prediction[0],decimals=3))
    max_prediction = np.amax(prediction[0])
    if prediction[0][0] > threshold[0]:
        print('Acc: '+str(prediction[0][0]) + '  Class: '+str(classes[0]))         
    if prediction[0][1] > threshold[1]:
        print('Acc: '+str(prediction[0][1]) + '  Class: '+str(classes[1]))     
    y_prob.append(prediction[0])
    
#  Close the stream
stream.stop_stream()
stream.close()
p.terminate()

#  Analysing data
# import matplotlib.pyplot as plt
# y_prob=np.array(y_prob)

# plt.plot(y_prob[:,0],color='green',label=classes[0])
# plt.plot(y_prob[:,1],color='red',label=classes[1])
# plt.plot(y_prob[:,2],color='black',label=classes[2],alpha=0.7)
# plt.plot(np.ones(len(y_prob))*threshold[0],color='black',linestyle='--', label = 'Grenzwert: ' + classes[0])
# plt.plot(np.ones(len(y_prob))*threshold[1],color='black',linestyle='--', label = 'Grenzwert: ' + classes[1])
# # plt.ylim(0,0.5)
# plt.legend()
# plt.show()