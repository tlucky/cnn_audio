import os
import numpy as np

class Config:
    def __init__(self, mode='conv', n_mels=32, n_fft=440, window='hann', 
                 sample_rate=16000, num_classes = 3, new_len=14079):
        self.mode = mode
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = int(n_fft / 2)        
        self.window = window
        self.sample_rate = sample_rate       
        self.num_classes = num_classes
        self.new_len = new_len  # Length of the      
        self.len_ms = new_len/sample_rate * 1000  # Frame lenght in ms  
        self.classes = np.array(['open', 'close', 'noise'])        
        self.threshold = np.array([0.8, 0.8])  # Threshold for the classes
        self.overlap = 0.5
        self.signal_old = np.zeros(int(new_len*0.25), dtype=float) 
        self.model_path = os.path.join('models', mode + '.model')
        self.p_path = os.path.join('pickles', mode + '.p')
        