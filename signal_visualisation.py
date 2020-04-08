import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from scipy.io import wavfile
import signal_processing as sp   

import imageio

import config_model
config = config_model.Config()

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
                  
def plot_wav(signals):
    fig, axes = plt.subplots(nrows=1, ncols=3, sharex=False,
                             sharey=True, figsize=(20,5))
    fig.suptitle('Emphasized Signal', size=16)
    
    scale = config.new_len/config.sample_rate*1000  # ms
    x_ticklabels = np.arange(0, int(scale),int(scale/8))
    scale = len(signals['Open'])
    x_ticks = np.arange(0, scale,int(scale/8))

    axes[0].set_ylabel('Amplitude [?]')

  
    for y in range(3):
            axes[y].set_title(list(signals.keys())[y])
            axes[y].plot(list(signals.values())[y])  
            axes[y].set_xlim(0, scale)
            axes[y].set_xticks(x_ticks)
            axes[y].set_xticklabels(x_ticklabels)
            axes[y].set_xlabel('Zeit [ms]')
            axes[y].set_xlim
    #plt.show()
        
def plot_fft(mag_frames):        
    fig, axes = plt.subplots(nrows=1, ncols=3, sharex=False,
                             sharey=True, figsize=(20,5))
    fig.suptitle('STFT', size=16)    
    scale = config.new_len/config.sample_rate*1000  # ms
    x_ticklabels = np.arange(0, int(scale),int(scale/8))
    scale = len(signals['Open'])
    x_ticks = np.arange(0, scale,int(scale/8))
    axes[0].set_ylabel('Frequenz [Hz]')  

    for y in range(3):
        axes[y].set_title(list(ffts.keys())[y])
        axes[y].imshow(list(ffts.values())[y].T, #extent=[0,T,0,4],
                cmap=plt.cm.jet, aspect='auto' )
        axes[y].invert_yaxis()
        #axes[y].set_xticks(x_ticks)
        #axes[y].set_xticklabels(x_ticklabels)
        axes[y].set_xlabel('Zeit [ms]') 
    #plt.show()
        
def plot_banks(fbanks): 
    fig, axes = plt.subplots(nrows=1, ncols=3, sharex=False,
                             sharey=True, figsize=(20,5))
    fig.suptitle('Filter Banks', size=16)
    axes[0].set_ylabel('Frequenz [Hz]')


    for y in range(3):
        axes[y].set_title(list(fbanks.keys())[y])
        axes[y].imshow(list(fbanks.values())[y].T,
                cmap=plt.cm.jet, aspect='auto' )
        axes[y] = plt.gca()
        axes[y].invert_yaxis()
        axes[y].set_xlabel('Zeit [ms]')  
    #plt.show()       

def plot_banks_norm(fbanks):        
    fig, axes = plt.subplots(nrows=1, ncols=3, sharex=False,
                             sharey=True, figsize=(20,5))
    fig.suptitle('Filter Banks Mean Normalization', size=16)
    axes[0].set_ylabel('Frequenz [Hz]')
    axes[0].set_xlabel('Zeit [ms]')  

    for y in range(3):
        axes[y].set_title(list(fbanks.keys())[y])
        axes[y].imshow(list(fbanks.values())[y].T,
                cmap=plt.cm.jet, aspect='auto')
        axes[y] = plt.gca()
        axes[y].invert_yaxis()
        axes[y].set_xlabel('Zeit [ms]')  
    #plt.show()   

def plot_mfcc(mfccs):
    fig, axes = plt.subplots(nrows=1, ncols=3, sharex=False,
                             sharey=True, figsize=(20,5))
    fig.suptitle('MFCC', size=16)
    axes[0].set_ylabel('Cepstrum Index')
    axes[0].set_xlabel('Zeit [ms]') 
    
    for y in range(3):
        axes[y].set_title(list(mfccs.keys())[y])
        axes[y].imshow(list(mfccs.values())[y].T,
                cmap=plt.cm.jet, aspect='auto')
        axes[y] = plt.gca()
        axes[y].invert_yaxis()
        axes[y].set_xlabel('Zeit [ms]')  
    plt.show() 
       
        
#  Program        
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
fbanks = {}
fbanks_norm = {}
mfccs = {}
dict_status = {0:'Open', 1:'Close', 2:'Error'} 

#  Calculation
for c in classes:   
    file = df[df.label==c].iloc[3,0]
    sample_rate, emphasized_signal = sp.read_wav(file)  # Read & 1. processing
    frames = sp.framing(sample_rate, emphasized_signal)  # Framing       
    pow_frames, mag_frames = sp.calc_fft(frames)  # Power and FFT      
    filter_banks = sp.calc_fbanks(sample_rate, pow_frames)  # Filter Banks  
    fbnorm = filter_banks - (np.mean(filter_banks, axis=0) + 1e-8) # Mean Normalization      
    mfcc = sp.calc_mfcc(filter_banks)  # MFCC
    #  Store in dict
    c = dict_status[c]
    signals[c] = emphasized_signal
    framed[c] = frames
    ffts[c] = mag_frames
    fbanks[c] = filter_banks
    mfccs[c] = mfcc      
    fbanks_norm[c]=fbnorm

#  Plotting    
plot_wav(signals)
plot_fft(mag_frames)
plot_banks(fbanks)
plot_banks_norm(fbanks_norm)
plot_mfcc(mfccs)

