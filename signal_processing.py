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

def calc_stft(frames):
    """
    Calculates the STFT and Power Spectrum.
    """
    mag_frames = np.absolute(np.fft.rfft(frames, config.nfft))  # Magnitude of the FFT
    pow_frames = ((1.0 / config.nfft) * ((mag_frames) ** 2))  # Power Spectrum
    return pow_frames, mag_frames
    
def calc_fbanks(sample_rate, pow_frames):
    """
    Calculates the Filter Banks based on the smaple rate and the power frames.
    """
    high_freq_mel = (2595 * np.log10(1 + (sample_rate / 2) / 700))  # Convert Hz to Mel
    mel_points = np.linspace(config.low_freq_mel, high_freq_mel, config.nfilt+2)  # Equally spaced in Mel scale
    hz_points = (700 * (10**(mel_points / 2595) - 1))  # Convert Mel to Hz
    bin = np.floor((config.nfft + 1) * hz_points / sample_rate)
    fbank = np.zeros((config.nfilt, int(np.floor(config.nfft / 2 + 1))))
    for m in range(1, config.nfilt + 1):
        f_m_minus = int(bin[m - 1])   # left
        f_m = int(bin[m])             # center
        f_m_plus = int(bin[m + 1])    # right 
        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
    filter_banks = np.dot(pow_frames, fbank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)  # Numerical Stability
    filter_banks = 20 * np.log10(filter_banks)  # dB
    return filter_banks

def calc_mfcc(filter_banks, num_ceps = config.num_ceps, cep_lifter = config.cep_lifter):
    """
    Calculates the MFCC based on the Filter Bank
    """
    mfcc = dct(filter_banks, type=2, axis=1, norm='ortho')[:, 1 : (num_ceps + 1)] # Keep 2-13
    (nframes, ncoeff) = mfcc.shape
    n = np.arange(ncoeff)
    lift = 1 + (cep_lifter / 2) * np.sin(np.pi * n / cep_lifter)
    mfcc *= lift  #*
    # mfcc -= (np.mean(mfcc, axis=0) + 1e-8)
    return mfcc

