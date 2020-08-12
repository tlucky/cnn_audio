import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from scipy.io import wavfile
import signal_processing as sp   
import config
from librosa.feature import melspectrogram
from librosa.core import amplitude_to_db, power_to_db, stft
# from librosa.display import specshow
import librosa.display
from scipy.signal import wiener, medfilt2d

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
# mel_amps = {}
mel_dbs = {}
dict_status = {0:'Open', 1:'Close', 2:'Noise'} 

# from random import randrange
# for index in range (0,100) :  
#     ran = randrange(1000) 

#  Calculation
for c in classes:   
    file = df[df.label==c].iloc[20,0]
    sample_rate, signal = wavfile.read('clean/'+file)   
    
    Y, freq = sp.calc_fft(signal, sample_rate)  # FFT    
    
    
    
    stft_signal = np.abs(stft(signal, 
                              n_fft=config.n_fft, 
                              hop_length=config.hop_length, 
                              window=config.window))
    stft_signal = amplitude_to_db(stft_signal, ref=np.max)  
    mel = melspectrogram(y=signal, 
                         sr=config.sample_rate, 
                         n_mels=config.n_mels, 
                         n_fft=config.n_fft, 
                         hop_length=config.hop_length, 
                         window=config.window)  
    mel[0] = (2*mel.mean() + mel[0])/3  # Reducing Noise
    mel[1] = (mel.mean() + 2*mel[1])/3  # Reducing Noise
    mel_db = amplitude_to_db(mel, ref=np.max)    
    # mel_db = power_to_db(mel)    
    # mel_pow = medfilt2d(mel_pow)
    # mel_pow = wiener(mel_pow)
    
    #  Store in dictionaries
    c = dict_status[c]
    signals[c] = signal
    ffts[c] = Y, freq    
    stfts[c] = stft_signal
    # mel_amps[c] = mel_amp
    mel_dbs[c] = mel_db      
  
    
#  Plots  
plt.style.use('ggplot')
x_ax = np.arange(0,config.new_len)/config.sample_rate
plt.figure(figsize=(5.9, 4),dpi=400)

plt.subplot(3,2,1)
plt.text(0.35, 0.6, 'Öffnen', fontsize=20)
plt.title('Signal')
plt.ylabel('Amplitude')
plt.ylim(-0.4, 0.4)
plt.xticks([])
plt.xlim(0,max(x_ax))
# plt.xticks([])
plt.plot(x_ax, list(signals.values())[0], color = 'black', linewidth=0.5) 
# plt.grid(linestyle='-')

plt.subplot(3,2,2)
plt.text(0.35, 0.6, 'Schließen', fontsize=20)
plt.title('Signal')
plt.ylim(-0.4, 0.4)
plt.xlim(0,max(x_ax))
plt.xticks([])
plt.yticks([])
plt.plot(x_ax, list(signals.values())[1], color = 'black', linewidth=0.5) 
# plt.grid(linestyle='-')

plt.subplot(3,2,3)
plt.title('Spektrogramm')
im = list(stfts.values())[0]
librosa.display.specshow(im, y_axis='linear', sr=32000, hop_length=220)
plt.clim(-90,22)
# plt.colorbar(format='%+2.0f dB', fraction=0.1, pad=0.04)

plt.subplot(3,2,4)
plt.title('Spektrogramm')
im = list(stfts.values())[1]
librosa.display.specshow(im, sr=16000, hop_length=110)
plt.clim(-90,22)
plt.colorbar(format='%+2.0f dB', fraction=0.1, pad=0.04)

plt.subplot(3,2,5)
plt.title('Mel Spektrogramm')
im = list(mel_dbs.values())[0]
librosa.display.specshow(im, sr=16000, hop_length=220, x_axis='ms', y_axis='mel')
plt.clim(-90,22)
# plt.colorbar(format='%+2.0f dB')

plt.subplot(3,2,6)
plt.title('Mel Spektrogramm')
im = list(mel_dbs.values())[1]
librosa.display.specshow(im, sr=16000, hop_length=220, x_axis='ms', y_axis='mel')
plt.clim(-90,22)
plt.yticks([])
plt.ylabel('')
plt.colorbar(format='%+2.0f dB', fraction=0.1, pad=0.04)

plt.subplots_adjust(hspace = 0.2)
plt.tight_layout()
plt.savefig('figures/p' + str(4) + '.png')
plt.show()

#########################
plt.figure(figsize=(5, 7), dpi=400)

plt.subplot(4,1,1)
# plt.text(0.35, 0.6, 'Öffnen', fontsize=20)
# plt.title('Signal')
plt.ylabel('Amplitude')
plt.ylim(-0.4, 0.4)
plt.xticks([])
plt.xlim(0,max(x_ax))
# plt.xticks([])
plt.plot(x_ax, list(signals.values())[1], color='black', linewidth=0.5) 
# plt.locator_params(nbins=4, axis='y')

plt.subplot(4,1,2)
# plt.title('Spektrogramm - linear skaliert')
im = list(stfts.values())[1]
librosa.display.specshow(im, sr=32000, hop_length=220, x_axis='ms', y_axis='linear')
# plt.clim(-90,22)
plt.xticks([])
plt.xlabel('')
plt.locator_params(nbins=4, axis='y')
cbar = plt.colorbar(format='%+2.0f dB', fraction=0.1, pad=0.04)
cbar.ax.locator_params(nbins=5)


plt.subplot(4,1,3)
# plt.title('Spektrogramm - logarithmisch skaliert')
im = list(stfts.values())[1]
librosa.display.specshow(im, sr=32000, hop_length=220, x_axis='ms', y_axis='log')
# plt.clim(-90,22)
plt.xticks([])
plt.xlabel('')
# plt.locator_params(nbins=4, axis='y')
cbar = plt.colorbar(format='%+2.0f dB', fraction=0.1, pad=0.04)
cbar.ax.locator_params(nbins=5)


plt.subplot(4,1,4)
# plt.title('Mel-Spektrogramm')
im = list(mel_dbs.values())[1]
librosa.display.specshow(im, sr=16000, hop_length=220, x_axis='ms', y_axis='mel')
# plt.clim(-90,22)
# plt.yticks([])
# plt.ylabel('')
cbar = plt.colorbar(format='%+2.0f dB', fraction=0.1, pad=0.04)
cbar.ax.locator_params(nbins=5)
# plt.locator_params(nbins=4, axis='y')
plt.subplots_adjust(hspace = 0.2)
plt.tight_layout()
plt.savefig('figures/p' + str(4) + '.png')
plt.tight_layout()

plt.show()

######

#  Mel-Filterbank
mels = librosa.filters.mel(sr=config.sample_rate, n_fft=config.n_fft, 
                                n_mels=config.n_mels/2)
mels /= np.max(mels, axis=-1)[:, None]
f = np.linspace(0,config.sample_rate,int(config.n_fft/2)+1)
plt.figure(figsize=(5.9, 4),dpi=300)
# plt.title('Mel-Filterbank')
plt.ylim(0,1)
plt.xlim(-100,max(f))
# plt.xlabel('H_mel (n)')
plt.yticks([0,0.5,1])
plt.xlabel('Frequenz [Hz]')
plt.plot(f,mels.T)
plt.tight_layout()
plt.show()

#  Activation Functions
plt.figure(figsize=(5.9, 2),dpi=400)
x = np.linspace(-10, 10, 10000) 

plt.subplot(1,3,1)
# plt.title('Tanh')
y = np.tanh(x) 
plt.plot(x, y, color = 'black') 
plt.plot([0,0],[-10,10], color = 'grey')
plt.plot([-10,10],[0,0], color = 'grey')
plt.xticks([])
plt.yticks([-1,0,1])
plt.ylim(-1.05,1.05)
plt.xlim(-3,3)

plt.subplot(1,3,2)
# plt.title('Sigmoid')
plt.plot([0,0],[-10,10], color = 'grey')
plt.plot([-10,10],[0,0], color = 'grey')
y = 1/(1+np.exp(-x))
plt.plot(x, y, color = 'black') 
plt.xticks([])
plt.yticks([-1,0,1])
plt.ylim(-1.05,1.05)
plt.xlim(-3,3)

plt.subplot(1,3,3)
# plt.title('ReLU')
plt.plot([0,0],[-10,10], color = 'grey')
plt.plot([-10,10],[0,0], color = 'grey')
y = np.zeros(int(len(x)/2))
y = np.append(y, x[int(len(x)/2):],axis = 0)
plt.plot(x, y, color = 'black') 
plt.plot(x, y, color = 'black') 
# plt.grid()
plt.xticks([])
plt.yticks([-1,0,1])
plt.ylim(-1.05,1.05)
plt.xlim(-3,3)
plt.tight_layout()
plt.show()

########

plt.figure(figsize=(5.9, 6),dpi=400)
 
plt.subplot(3,1,1)
signal, sample_rate = librosa.load('./wavfile/C1713.wav', sr=config.sample_rate)
signal = signal + 0.5  # Adjusting the signal (offset)
signal = signal[:-21]
signal = signal[21:] 
x_ax = np.arange(0,len(signal))/config.sample_rate
plt.ylabel('Amplitude')
plt.ylim(-0.4, 0.4)
plt.xlabel('s')
x_lim = max(x_ax)
plt.xlim(0,x_lim)
plt.plot(x_ax, signal, color = 'black', linewidth=0.5) 

plt.subplot(3,1,2)
signal, sample_rate = librosa.load('./wavfile/CQ1020.wav', sr=config.sample_rate)
signal = signal + 0.5  # Adjusting the signal (offset)
signal = signal[:-21]
signal = signal[21:] 
x_ax = np.arange(0,len(signal))/config.sample_rate
plt.ylabel('Amplitude')
plt.ylim(-0.4, 0.4)
plt.xlabel('s')
plt.xlim(0,x_lim)
plt.plot(x_ax, signal, color = 'black', linewidth=0.5) 

plt.subplot(3,1,3)
signal, sample_rate = librosa.load('./wavfile/O118_1.wav', sr=config.sample_rate)
signal = signal + 0.5  # Adjusting the signal (offset)
signal = signal[:-21]
signal = signal[21:] 
x_ax = np.arange(0,len(signal))/config.sample_rate
plt.ylabel('Amplitude')
plt.ylim(-0.4, 0.4)
plt.xlabel('s')
plt.xlim(0,x_lim)
plt.plot(x_ax, signal, color = 'black', linewidth=0.5) 
plt.show()


####### Data augmentation
plt.figure(figsize=(5.9, 8),dpi=400)
 
plt.subplot(4,1,1)
signal, sample_rate = librosa.load('./wavfile/O1214_1.wav', sr=config.sample_rate)
signal = signal + 0.5  # Adjusting the signal (offset)
signal = signal[:-21]
signal = signal[21:] 
x_ax = np.arange(0,len(signal))/config.sample_rate
plt.ylabel('Amplitude')
plt.ylim(-0.4, 0.4)
plt.xlabel('s')
x_lim = max(x_ax)
plt.xlim(0,x_lim)
plt.plot(x_ax, signal, color = 'black', linewidth=0.5) 

plt.subplot(4,1,2)
signal, sample_rate = librosa.load('./clean/O1214_1.wav', sr=config.sample_rate)
x_ax = np.arange(0,len(signal))/config.sample_rate
plt.ylabel('Amplitude')
plt.ylim(-0.4, 0.4)
plt.xlabel('s')
plt.xlim(0,x_lim)
plt.plot(x_ax, signal, color = 'black', linewidth=0.5) 

plt.subplot(4,1,3)
signal, sample_rate = librosa.load('./clean/O1214_1p0t1.wav', sr=config.sample_rate)
x_ax = np.arange(0,len(signal))/config.sample_rate
plt.ylabel('Amplitude')
plt.ylim(-0.4, 0.4)
plt.xlabel('s')
plt.xlim(0,x_lim)
plt.plot(x_ax, signal, color = 'black', linewidth=0.5) 


plt.subplot(4,1,4)
signal, sample_rate = librosa.load('./clean/O1214_1p1t0.wav', sr=config.sample_rate)
x_ax = np.arange(0,len(signal))/config.sample_rate
plt.ylabel('Amplitude')
plt.ylim(-0.4, 0.4)
plt.xlabel('s')
plt.xlim(0,x_lim)
plt.plot(x_ax, signal, color = 'black', linewidth=0.5) 
plt.show()
