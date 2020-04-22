"""
Cleaning of the .wav files folder "wavefile" and saves them in folder "clean"
if folder "clean" is empty.

Same length from config_model.new_len
"""
import os
import pandas as pd
from scipy.io import wavfile
from tqdm import tqdm
import librosa
import numpy as np

import signal_processing as sp
import config

def work_status(begin_str):
    """
    For the definition of the different classes
    """
    if begin_str.startswith('O') == True:
        return (0)  # Open
    if begin_str.startswith('C') == True:
        return (1)  # Close
    
config = config.Config()
sample_length = config.new_len

if len(os.listdir('clean')) == 0:
    df = pd.DataFrame(columns=['fname', 'label', 'length'],)  
    df['fname'] = os.listdir('./wavfile/')
    for index, row in df.iterrows():
        # row['label'] = work_status(row['fname'])    
        rate, signal = wavfile.read('wavfile/'+row['fname'])
        row['length'] = signal.shape[0] / rate
    
    for f in tqdm(df.fname):
        # f= 'C2203.WAV'
        # f= 'O2206.WAV'

        #  Importing and adjusting sample rate
        signal, sample_rate = librosa.load('./wavfile/'+f, sr=config.sample_rate)  
        signal = signal+0.5  # Adjusting the signal (offset)
        signal = signal[:-21]
        signal = signal[21:]  
        signal_mani = signal
        
        # Data augmentation        
        pitch_shift = np.random.uniform(low=0.8, high=1.2, size=(3,))
        
        for p_index in range (0, len(pitch_shift)):  #  for pitch shifting
            max_quarter = max(abs(signal[ : int(len(signal)*0.4)]))
            signal_index = np.where(abs(signal) == max_quarter)[0][0]
            
            for t_index in range (0, 3):  #  for time shifting                
                time_shift = np.random.randint(3000, 9000)  # Data augmentation
                signal_mani = signal * pitch_shift[p_index]
                signal_cut = signal_mani[signal_index-time_shift : 
                                         signal_index-time_shift+sample_length]
                signal_cut = np.append(signal_cut,signal)  # Append signal
                signal_cut = signal_cut[:sample_length]
                f_1 = f[:-4] + str(p_index) + str(t_index) + f[-4:]  # New file name
                if len(signal_cut) == sample_length:  # Check signal length
                    wavfile.write(filename='./clean/'+f_1, rate=sample_rate, 
                                  data=signal_cut)
                
                # Create 'Noise' class for better prediction
                if f[0] == 'C':  
                    max_signal = max(abs(signal))
                    max_index = np.where(abs(signal) == max_signal)[0][0]
                    signal_noise = np.append(signal_mani[: max_index-time_shift], 
                                             signal_mani[max_index+time_shift: ])                    
                    # Delete peak again
                    max_signal = max(abs(signal_noise))
                    max_index = np.where(abs(signal_noise) == max_signal)[0][0]
                    
                    if max_signal > 0.1:  # Cuts again if noise > 0.1 
                        signal_noise = np.append(signal_noise[0:max_index-400], 
                                                 signal_noise[max_index+400:])                
                    
                    signal_noise = signal_noise[:sample_length]
                    # New file name
                    f_2 = 'N' + f[1:-4] + str(p_index) + str(t_index) + f[-4:] 
                    
                    if len(signal_noise) == sample_length:  # Check signal length 
                        wavfile.write(filename='./clean/'+f_2, rate=sample_rate, 
                                      data=signal_noise)
       
        # import matplotlib.pyplot as plt
        # plt.plot(signal)
        # plt.plot(signal_cut)
        # plt.show()
        
        
