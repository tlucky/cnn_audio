"""
Do the caluclation stuff, from wave file to MFCC.
"""

import numpy as np
from scipy.fftpack import dct

import config

config = config.Config()

def resample(arr):
    """
    Compresses the singal to the parameter "new_len" which is mandatory for 
    the X input to the CNN.
    """
    old_len = len(arr)
    diff = old_len-config.new_len
    index_rand = np.random.permutation(diff)  # Random indices which are getting deleted
    new_arr = np.delete(arr, index_rand)
    return new_arr

# https://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html    

def framing(sample_rate, signal):
    # Framing
    """
    Framing the singnal
    Definition of the overlapping of 50%
    Definition of the step size
    ...
    Using the hamming window
    """
    frame_stride = config.frame_size/2  # Overlap 50%        
    frame_length = config.frame_size * sample_rate  # Convert from seconds to samples
    frame_step = frame_stride * sample_rate  # Convert from seconds to samples
    frame_length = int(round(frame_length))
    frame_step = int(round(frame_step))   
        
    signal_length = len(signal)
    num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) 
                             / frame_step))  # Make sure that we have at least 1 frame
    
    pad_signal_length = num_frames * frame_step + frame_length
    z = np.zeros((pad_signal_length - signal_length))
    
    # Pad Signal to make sure that all frames have equal number of samples 
    # without truncating any samples from the original signal
    pad_signal = np.append(signal, z) 
    
    indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
    
    frames = pad_signal[indices.astype(np.int32, copy=False)]
    # frames = pad_signal[indices.astype(np.int16, copy=False)]
    frames *= np.hamming(frame_length)  # Hamming Window
    return frames

def calc_fft(signal, rate):
    """
    Calculates the FFT
    """
    n = len(signal)
    freq = np.fft.rfftfreq(n, d=1/rate)
    Y = abs(np.fft.rfft(signal) / n)
    return (Y, freq)
