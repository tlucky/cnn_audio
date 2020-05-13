import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from scipy.io import wavfile
import signal_processing as sp   
import config
from librosa.feature import melspectrogram
from librosa.core import amplitude_to_db, power_to_db, stft
from librosa.display import specshow
import librosa

def work_status(begin_str):
    """
    For the definition of the different classes
    """
    if begin_str.startswith('O') == True:
        return (0)  # Open
    if begin_str.startswith('C') == True:
        return (1)  # Close
    if begin_str.startswith('N') == True:
        return (2)  # Noise
              
       
#  Program      
config = config.Config()
  
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

#  Create dictionaries
signals = {}
framed = {}
ffts = {}
stfts = {}
mel_amps = {}
mel_pows = {}

dict_status = {0:'Open', 1:'Close', 2:'Noise'} 

# from random import randrange
# for index in range (0,100) :  
#     ran = randrange(1000) 

#  Calculation
for c in classes:   
    file = df[df.label==c].iloc[4,0]
    sample_rate, signal = wavfile.read('clean/'+file)    
    Y, freq = sp.calc_fft(signal, sample_rate)  # FFT    
    
    stft_signal = np.abs(stft(signal, n_fft=220, hop_length=110, window='hann'))
    mel = melspectrogram(y=signal, sr=sample_rate, n_mels=32, n_fft=220, 
                          hop_length=110, window='hann')
    mel_amp = amplitude_to_db(mel)
    mel_pow = power_to_db(mel)
    
    #  Store in dictionaries
    c = dict_status[c]
    signals[c] = signal
    ffts[c] = Y, freq    
    stfts[c] = stft_signal
    mel_amps[c] = mel_amp
    mel_pows[c] = mel_pow      
  
#  Plots  
x_ax = np.arange(0,config.new_len)/config.sample_rate
plt.figure(figsize=(12, 8))

plt.subplot(3,2,1)
plt.text(0.35, 0.6, 'Öffnen', fontsize=20)
plt.title('Signal')
plt.ylabel('Amplitude')
plt.ylim(-0.4, 0.4)
plt.xticks([])
plt.xlim(0,max(x_ax))
# plt.xticks([])
plt.plot(x_ax, list(signals.values())[0]) 
# plt.grid(linestyle='-')

plt.subplot(3,2,2)
plt.text(0.35, 0.6, 'Schließen', fontsize=20)
plt.title('Signal')
plt.ylim(-0.4, 0.4)
plt.xlim(0,max(x_ax))
plt.xticks([])
plt.yticks([])
plt.plot(x_ax, list(signals.values())[1]) 
# plt.grid(linestyle='-')

plt.subplot(3,2,3)
plt.title('Spektrogramm')
im = librosa.amplitude_to_db(list(stfts.values())[0])
librosa.display.specshow(im, y_axis='linear', sr=32000, hop_length=220)
plt.clim(-65,20)
# plt.colorbar(format='%+2.0f dB', fraction=0.1, pad=0.04)

plt.subplot(3,2,4)
plt.title('Spektrogramm')
im = librosa.amplitude_to_db(list(stfts.values())[1])
librosa.display.specshow(im, sr=16000, hop_length=110)
plt.clim(-65,20)
# plt.colorbar(format='%+2.0f dB', fraction=0.1, pad=0.04)

plt.subplot(3,2,5)
plt.title('Mel Spektrogramm')
im = list(mel_pows.values())[0]
librosa.display.specshow(im, sr=16000, hop_length=110, x_axis='ms', y_axis='mel')
plt.clim(-65,20)
# plt.colorbar(format='%+2.0f dB')

plt.subplot(3,2,6)
plt.title('Mel Spektrogramm')
im = list(mel_pows.values())[1]
librosa.display.specshow(im, sr=16000, hop_length=110, x_axis='ms', y_axis='mel')
plt.clim(-65,20)
plt.yticks([])
plt.ylabel('')
# plt.colorbar(format='%+2.0f dB', fraction=0.1, pad=0.04)

plt.subplots_adjust(hspace = 0.2)
plt.tight_layout()
plt.savefig('figures/p' + str(4) + '.png')