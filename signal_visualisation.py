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

# def rounddown(x, round_to):
#     return int(int(x / round_to)) * round_to

def rescale_axis(max_old_total, max_new_total, max_new_scale, n_ticks): #, max_old=config.len_ms()):
    """
    Calculates the arrays which are needed for resizing the y-axis    
    """   
    ax_max = max_old_total / max_new_total * max_new_scale + 1 #50.88  
    step = max_old_total / max_new_total * max_new_scale / (n_ticks - 1) # 6.23
    new_dist = np.arange(0, ax_max, step) # für 3    
    label = np.linspace(0, max_new_scale, n_ticks,dtype=int)#-1)), int)        
    return new_dist, label
                  
def plot_wav(signals):
    fig, ax = plt.subplots(nrows=1, ncols=3, sharex=False,
                             sharey=True, figsize=(20,5))
    fig.suptitle('Emphasized Signal', size=16)
    x_tick = signals['Open'].shape[0]     
    xnew_dist, xlabel = rescale_axis(max_old_total=x_tick, max_new_scale=800,
                                     max_new_total=config.len_ms(), n_ticks=9)
    ax[0].set_ylabel('Amplitude')
  
    for y in range(3):
            ax[y].set_title(list(signals.keys())[y])
            ax[y].plot(list(signals.values())[y])  
            ax[y].set_xlim(0, config.new_len)
            ax[y].set_xticks(xnew_dist)
            ax[y].set_xticklabels(xlabel)
            ax[y].set_xlabel('Zeit [ms]')
            ax[y].grid(linestyle='-')

def plot_fft(fft):
    fig, ax = plt.subplots(nrows=1, ncols=3, sharex=False,
                             sharey=True, figsize=(20,5))
    fig.suptitle('Fourier Tranformation', size=16)
    ax[0].set_ylabel('Amplitude')
    for y in range(3):
        data = list(ffts.values())[y]
        Y, freq = data[0], data[1]
        ax[y].set_title(list(ffts.keys())[y])
        ax[y].plot(freq, Y)  
        ax[y].set_xlim(0, freq[-1])
        if y==0: ax[y].set_ylim(0, max(Y)*1.05)
        ax[y].set_xlabel('Frequenz [kHz]')
        ax[y].fill_between(freq, Y)  # Fills the area under the plot
        ax[y].grid(linestyle='-')

def plot_fft_hamming(fft_hamming):
    fig, ax = plt.subplots(nrows=1, ncols=3, sharex=False,
                             sharey=True, figsize=(20,5))
    fig.suptitle('Fourier Transformation mit Hamming Fenster', size=16)
    ax[0].set_ylabel('Amplitude')
    for y in range(3):
        data = list(ffts_hamming.values())[y]
        Y, freq = data[0], data[1]
        ax[y].set_title(list(ffts_hamming.keys())[y])
        ax[y].plot(freq, Y)  
        ax[y].set_xlim(0, freq[-1])
        if y==0: ax[y].set_ylim(0, max(Y)*1.05)
        ax[y].set_xlabel('Frequenz [kHz]')
        ax[y].fill_between(freq, Y)  # Fills the area under the plot
        ax[y].grid(linestyle='-')

def plot_stft(mag_frames):        
    fig, ax = plt.subplots(nrows=1, ncols=3, sharex=False,
                             sharey=True, figsize=(20,5))
    fig.suptitle('STFT', size=16)      
    # for the rescaling of the axes
    x_tick, y_tick = stfts['Open'].shape  

    xnew_dist, xlabel = rescale_axis(max_old_total=x_tick, max_new_scale=800,
                                     max_new_total=config.len_ms(), n_ticks=9)
    
    ynew_dist, ylabel = rescale_axis(max_old_total=y_tick,max_new_scale=10,
                                     max_new_total=10, n_ticks=11)
    ax[0].set_yticks(ynew_dist)
    ax[0].set_yticklabels(ylabel)  # Range from 0 - 10.000 Hz    
    ax[0].set_ylabel('Frequenz [kHz]')      
    for y in range(3):
        ax[y].set_title(list(stfts.keys())[y])
        im = ax[y].imshow(list(stfts.values())[y].T,#extent=[0,T,0,4],
                       cmap=plt.cm.jet, aspect='auto',vmin=0.0,vmax=17.5) # 
        ax[y].set_xlim(0, x_tick-0.5)
        ax[y].set_xticks(xnew_dist)
        ax[y].set_xticklabels(xlabel)
        ax[y].set_xlabel('Zeit [ms]')
        ax[y].invert_yaxis() 
        ax[y].set_ylim(0, y_tick)

    fig.colorbar(im,ax=ax[2])  # shows the colorbar one
        
def plot_banks(fbanks): 
    fig, ax = plt.subplots(nrows=1, ncols=3, sharex=False,
                             sharey=True, figsize=(15,5))
    fig.suptitle('Filter Banks', size=16)
    
    x_tick, y_tick = fbanks['Open'].shape  
    xnew_dist, xlabel = rescale_axis(max_old_total=x_tick, max_new_scale=800,
                                     max_new_total=config.len_ms(), n_ticks=9)
    
    ynew_dist, ylabel = rescale_axis(max_old_total=y_tick,max_new_scale=10,
                                     max_new_total=10, n_ticks=11)
    
    ax[0].set_ylabel('Frequenz [kHz]')
    
    for y in range(3):
        ax[y].set_title(list(fbanks.keys())[y])
        im = ax[y].imshow(list(fbanks.values())[y].T,
                cmap=plt.cm.jet, aspect='auto' )
        ax[y].set_ylim(y_tick, 0)
        ax[y].set_xlim(0, x_tick-0.5)
        ax[y].set_xticks(xnew_dist)
        ax[y].set_xticklabels(xlabel)
        ax[y].set_xlabel('Zeit [ms]')
        ax[y].set_yticks(ynew_dist)
        ax[y].set_yticklabels(ylabel)
        ax[y] = plt.gca()
        ax[y].invert_yaxis()
    fig.colorbar(im,ax=ax[2])

def plot_banks_norm(fbanks_norm):        
    fig, ax = plt.subplots(nrows=1, ncols=3, sharex=False,
                             sharey=True, figsize=(15,5))
    fig.suptitle('Filter Banks Mean Normalization', size=16)

    x_tick, y_tick = fbanks_norm['Open'].shape  
    xnew_dist, xlabel = rescale_axis(max_old_total=x_tick, max_new_scale=800,
                                     max_new_total=config.len_ms(), n_ticks=9)
    
    ynew_dist, ylabel = rescale_axis(max_old_total=y_tick,max_new_scale=10,
                                     max_new_total=10, n_ticks=11)

    ax[0].set_ylabel('Frequenz [kHz]')
    
    for y in range(3):
        ax[y].set_title(list(fbanks_norm.keys())[y])
        im = ax[y].imshow(list(fbanks_norm.values())[y].T,
                cmap=plt.cm.jet, aspect='auto' )
        ax[y].set_ylim(y_tick, 0)
        ax[y].set_xlim(0, x_tick-1)
        ax[y].set_xticks(xnew_dist)
        ax[y].set_xticklabels(xlabel)
        ax[y].set_xlabel('Zeit [ms]') 
        ax[y].set_yticks(ynew_dist)
        ax[y].set_yticklabels(ylabel)  # Range from 0 - 10.000 Hz
        ax[y] = plt.gca()
        ax[y].invert_yaxis()
    fig.colorbar(im,ax=ax[2])

def plot_mfcc(mfccs):
    fig, ax = plt.subplots(nrows=1, ncols=3, sharex=False, sharey=True,
                           figsize=(15,5))
    
    #fig.suptitle('MFCC', size=16)
    
    fig.tight_layout()
    x_tick, y_tick = mfccs['Open'].shape  
    xnew_dist, xlabel = rescale_axis(max_old_total=x_tick, max_new_scale=800,
                                 max_new_total=config.len_ms(), n_ticks=9)
    ax[0].set_ylabel('MFCC Koeffizienten')
    
    for y in range(3):
        ax[y].set_title(list(mfccs.keys())[y])
        im = ax[y].imshow(list(mfccs.values())[y].T, cmap=plt.cm.jet, 
                          aspect='auto',vmin=-400,vmax=425)
        # ax[y].set_ylim(y_tick-0.5, 0)
        ax[y].set_xlim(0, x_tick-0.5)
        ax[y].set_xticks(xnew_dist)
        ax[y].set_xticklabels(xlabel)
        ax[y].set_xlabel('Zeit [ms]')
        ax[y] = plt.gca()
        ax[y].invert_yaxis()
    fig.colorbar(im,ax=ax[2])        
        
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
ffts_hamming = {}
stfts = {}
fbanks = {}
fbanks_norm = {}
mfccs = {}
dict_status = {0:'Open', 1:'Close', 2:'Error'} 

#  Calculation
for c in classes:   
    file = df[df.label==c].iloc[3,0]
    sample_rate, emphasized_signal = sp.read_wav(file)  # Read & 1. processing
    
    Y, freq = sp.calc_fft(emphasized_signal, sample_rate)  # FFT
    Y_h, freq_h= sp.calc_fft(emphasized_signal*np.hamming(len(emphasized_signal)), sample_rate)
    
    frames = sp.framing(sample_rate, emphasized_signal)  # Framing       
    pow_frames, mag_frames = sp.calc_stft(frames)  # Power and FFT      
    filter_banks = sp.calc_fbanks(sample_rate, pow_frames)  # Filter Banks  
    fbnorm = filter_banks - (np.mean(filter_banks, axis=0) + 1e-8) # Mean Normalization      
    mfcc = sp.calc_mfcc(filter_banks)  # MFCC
    #  Store in dict
    c = dict_status[c]
    signals[c] = emphasized_signal
    ffts[c] = Y, freq
    ffts_hamming[c] = Y_h, freq_h
    framed[c] = frames
    stfts[c] = mag_frames
    fbanks[c] = filter_banks
    mfccs[c] = mfcc      
    fbanks_norm[c]=fbnorm

#  Definitions of the font sizes of the plots
plt.rc('font', size =14)
plt.rc('axes', titlesize=16)
plt.rc('axes', labelsize=14)
# plt.rc('legend', fontsize=16)

#  Plotting    
# p1=plot_wav(signals)
# p2=plot_fft(ffts)
# p3=plot_fft_hamming(ffts_hamming)
p4=plot_stft(stfts)
# p5=plot_banks(fbanks)
# p6=plot_banks_norm(fbanks_norm)
p7=plot_mfcc(mfccs)

