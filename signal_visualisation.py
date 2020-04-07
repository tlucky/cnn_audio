import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from scipy.io import wavfile
import signal_processing as sp   

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
    #xmin, xmax = axes([0, 13856])
    #values_x = range(0, 13856,int(13856/8))
    
    scale = config.new_len/config.sample_rate*1000  # ms
    values_x = np.arange(0, int(scale),int(scale/8))
    scale1 = len(signals['Open'])
    label_x = np.arange(0, scale1,int(scale1/8))
    #set_axis_style(axes,values_x)
    i = 0    
    for y in range(3):
            axes[y].set_title(list(signals.keys())[i])
            axes[y].plot(list(signals.values())[i])
            
            axes[y].set_xticks(label_x)
            axes[y].set_xticklabels(values_x)
            axes[y].set_ylabel('Hz')
            axes[y].set_xlabel('ms')
            i += 1
    #plt.show()
        
def plot_fft(mag_frames):        
    fig, axes = plt.subplots(nrows=1, ncols=3, sharex=False,
                             sharey=True, figsize=(20,5))
    fig.suptitle('FFT', size=16)
    i = 0
    for y in range(3):
        axes[y].set_title(list(ffts.keys())[i])
        axes[y].imshow(list(ffts.values())[i],
                cmap=plt.cm.jet, aspect='auto' )
        i += 1
    #plt.show()
        
def plot_banks(fbanks): 
    fig, axes = plt.subplots(nrows=1, ncols=3, sharex=False,
                             sharey=True, figsize=(20,5))
    fig.suptitle('Filter Banks', size=16)
    i = 0
    for y in range(3):
        axes[y].set_title(list(fbanks.keys())[i])
        axes[y].imshow(list(fbanks.values())[i],
                cmap=plt.cm.jet, aspect='auto' )
        axes[y] = plt.gca()
        axes[y].invert_yaxis()
        i += 1
    #plt.show()       

def plot_banks_norm(fbanks):        
    fig, axes = plt.subplots(nrows=1, ncols=3, sharex=False,
                             sharey=True, figsize=(20,5))
    fig.suptitle('Filter Banks Mean Normalization', size=16)
    i = 0
    for y in range(3):
        axes[y].set_title(list(fbanks.keys())[i])
        axes[y].imshow(list(fbanks.values())[i],
                cmap=plt.cm.jet, aspect='auto')
        axes[y] = plt.gca()
        axes[y].invert_yaxis()
        i += 1
    #plt.show()   

def plot_mfcc(mfccs):
    fig, axes = plt.subplots(nrows=1, ncols=3, sharex=False,
                             sharey=True, figsize=(20,5))
    fig.suptitle('MFCC', size=16)
    i = 0
    for y in range(3):
        axes[y].set_title(list(mfccs.keys())[i])
        axes[y].imshow(list(mfccs.values())[i],
                cmap=plt.cm.jet, aspect='auto')
        axes[y] = plt.gca()
        axes[y].invert_yaxis()
        i += 1
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
    file = df[df.label==c].iloc[4,0]
    sample_rate, emphasized_signal = sp.read_wav(file)  # Read & 1. processing
    frames = sp.framing(sample_rate, emphasized_signal)  # Framing       
    pow_frames, mag_frames = sp.calc_fft(frames)  # Power and FFT      
    filter_banks = sp.calc_fbanks(sample_rate, pow_frames).T  # Filter Banks  
    fbnorm = filter_banks - (np.mean(filter_banks, axis=0) + 1e-8) # Mean Normalization      
    mfcc = sp.calc_mfcc(filter_banks).T  # MFCC
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

