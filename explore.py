import numpy as np
import pandas as pd
import os
from tqdm import tqdm

import pickle

from keras.utils import to_categorical
from scipy.io import wavfile
from scipy.fftpack import dct

from keras.layers import Conv2D, MaxPool2D, Flatten
from keras.layers import LeakyReLU, MaxPooling2D
from keras.layers import Dropout, Dense, TimeDistributed
from keras.models import Sequential
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint

class Config:
    # nfft=500
    def __init__(self, mode='conv', nfilt=40, nfeat=1, nfft=512, sample_rate=16000, 
                 low_freq_mel = 0, pre_emphasis = 0.97, frame_size = 0.025):
        self.mode = mode
        self.nfilt = nfilt
        self.nfeat = nfeat
        self.nfft = nfft
        self.sample_rate = sample_rate
        self.low_freq_mel = low_freq_mel
        self.pre_emphasis = pre_emphasis
        self.frame_size = frame_size
        #self.step = int(rate/10)  # hier evtl das auch ändern
        self.model_path = os.path.join('models', mode + '.model')
        self.p_path = os.path.join('pickles', mode + '.p')
        
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

def resample(arr, new_len=13856):
    """
    Compresses the singal to the parameter "new_len" which is mandatory for 
    the X input to the CNN.

    """
    old_len = len(arr)
    diff = old_len-new_len
    index_rand = np.random.permutation(diff)  # Random indices which are getting deleted
    new_arr = np.delete(arr, index_rand)
    return new_arr

    
def calc_fft(frames):
    """
    Calculates the FFT and Power Spectrum.
    """
    mag_frames = np.absolute(np.fft.rfft(frames, config.nfft))  # Magnitude of the FFT
    pow_frames = ((1.0 / config.nfft) * ((mag_frames) ** 2))  # Power Spectrum
    return pow_frames, mag_frames
    
def calc_fbanks(sample_rate, pow_frames):
    """
    Calculates the Filter Banks based on the smaple rate and the power frames.
    """
    high_freq_mel = (2595 * np.log10(1 + (sample_rate / 2) / 700))  # Convert Hz to Mel
    mel_points = np.linspace(config.low_freq_mel, high_freq_mel, config.nfilt + 2)  # Equally spaced in Mel scale
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

def calc_mfcc(filter_banks):
    """
    Calculates the MFCC based on the Filter Bank
    """
    num_ceps = 12
    cep_lifter = 22
    mfcc = dct(filter_banks, type=2, axis=1, norm='ortho')[:, 1 : (num_ceps + 1)] # Keep 2-13
    (nframes, ncoeff) = mfcc.shape
    n = np.arange(ncoeff)
    lift = 1 + (cep_lifter / 2) * np.sin(np.pi * n / cep_lifter)
    mfcc *= lift  #*
    return mfcc

def read_wav(file):
    """
    Reads in the wave file. Add a offset (+0.5) to the signal.
    Emphasises the signal.
    Compresses the signal to one length (which is the smallest file size)
    """
    sample_rate, signal = wavfile.read('clean/'+file)
    signal = signal + 0.5
    # Conpression of the signal
    comp_signal = resample(signal, 13856)
    # Pre-Emphasis
    emphasized_signal = np.append(comp_signal[0], comp_signal[1:] - config.pre_emphasis * comp_signal[:-1])
    return sample_rate, emphasized_signal

def framing(sample_rate, emphasized_signal):
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
        
    signal_length = len(emphasized_signal)
    num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step))  # Make sure that we have at least 1 frame
    
    pad_signal_length = num_frames * frame_step + frame_length
    z = np.zeros((pad_signal_length - signal_length))
    pad_signal = np.append(emphasized_signal, z) # Pad Signal to make sure that all frames have equal number of samples without truncating any samples from the original signal
    
    indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
    frames = pad_signal[indices.astype(np.int32, copy=False)]
    
    # Window
    frames *= np.hamming(frame_length)  # Hamming Window
    return frames

def build_X():
    """
    Building X and y for the input and output of the CNN

    """
    X = []
    _min, _max = float('inf'), -float('inf')
    for index, file in tqdm(enumerate(df['fname'])):
        # Read the File and first processing
        sample_rate, emphasized_signal = read_wav(file)     
        
        # Framing
        frames = framing(sample_rate, emphasized_signal)        
        # Power and FFT
        pow_frames, mag_frames = calc_fft(frames)       
        # Filter Banks
        filter_banks = calc_fbanks(sample_rate, pow_frames)        
        # Mel-frequency Cepstral Coefficients (MFCCs)
        mfcc = calc_mfcc(filter_banks)
        
        _min = min(np.amin(mfcc), _min)
        _max = max(np.amin(mfcc), _max)

        X.append(mfcc)
    config.min = _min
    config.max = _max
    X = np.array(X)
    X = (X - _min) / (_max - _min)
    X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
    config.data = (X)
    return X

def get_conv_model():
    model = Sequential()
    model.add(Conv2D(16, (3,3), activation='relu', strides=(1, 1),
                      padding='same',input_shape=input_shape))
    model.add(Conv2D(32, (3, 3), activation='relu', strides=(1,1),
                      padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu', strides=(1,1),
                      padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu', strides=(1,1),
                      padding='same'))
    model.add(MaxPool2D((2, 2)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(3, activation='softmax'))
    model.summary()
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['acc'])
    return model

#  Program
df = pd.DataFrame(columns=['fname', 'label', 'length'],)  
df['fname'] = os.listdir('./clean/')
for index, row in df.iterrows():
    row['label'] = work_status(row['fname'])    
    rate, signal = wavfile.read('clean/'+row['fname'])
    row['length'] = signal.shape[0] / rate
#df.set_index('fname', inplace = True)
#  count the different labels and their distribution
classes = list(np.unique(df.label))
class_dist = df.groupby(['label'])['label'].count()/len(df)
prob_dist = class_dist / class_dist.sum()

config = Config()
file = df['fname']

X = build_X()
y = []
y = to_categorical(df.label, num_classes=3)
y_flat = np.argmax(y, axis=1)
input_shape = (X.shape[1], X.shape[2], 1)

model = get_conv_model()

class_weight = compute_class_weight('balanced', np.unique(y_flat), y_flat)

#  Only saves a new model when the predicition increases
checkpoint = ModelCheckpoint(config.model_path, monitor='val_acc', verbose=1, 
                              mode='max', save_best_only=True, 
                              save_weights_only=False, period=1)

model.fit(X, y, epochs=10, batch_size=32, shuffle=True, validation_split=0.1,
          callbacks=[checkpoint])
          #class_weight=class_weight)

model.save(config.model_path)

#model.save(config.model_path)

# # Prints
# plt.plot(emphasized_signal)
# plt.title('emphasized_signal 2')
# plt.show()

# plt.imshow(mag_frames.T, cmap=plt.cm.jet, aspect='auto')
# ax = plt.gca()
# ax.invert_yaxis()
# plt.title('FFT 3')
# plt.show()

# plt.imshow(filter_banks.T, cmap=plt.cm.jet, aspect='auto')
# # plt.xticks(np.arange(0, (filter_banks.T).shape[1],
# # int((filter_banks.T).shape[1] / 4)),
# # ['0s', '0.5s', '1s', '1.5s','2.5s','3s','3.5'])
# ax = plt.gca()
# ax.invert_yaxis()
# plt.title('Filter Banks 4')
# plt.show()
# # Mean Normalization
# filter_banks -= (np.mean(filter_banks, axis=0) + 1e-8)

# plt.imshow(filter_banks.T, cmap=plt.cm.jet, aspect='auto')
# # plt.xticks(np.arange(0, (filter_banks.T).shape[1],
# # int((filter_banks.T).shape[1] / 4)),
# # ['0s', '0.5s', '1s', '1.5s','2.5s','3s','3.5'])
# ax = plt.gca()
# ax.invert_yaxis()
# plt.title('Filter Banks Mean Normalization 5')
# plt.show()

# plt.imshow(X.T, cmap=plt.cm.jet, aspect='auto')
# # plt.xticks(np.arange(0, (filter_banks.T).shape[1],
# # int((filter_banks.T).shape[1] / 4)),
# # ['0s', '0.5s', '1s', '1.5s','2.5s','3s','3.5'])
# ax = plt.gca()
# ax.invert_yaxis()
# plt.title('mfcc 6')
# plt.show()


