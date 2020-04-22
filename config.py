import os

class Config:
    def __init__(self, mode='conv', nfilt=40, nfeat=1, nfft=512, sample_rate=16000, 
                 low_freq_mel = 0, frame_size = 0.025, 
                 new_len=13000, num_ceps = 12, cep_lifter = 22, freq_fft=10000):
        self.mode = mode
        self.nfilt = nfilt
        self.nfeat = nfeat
        self.nfft = nfft
        self.sample_rate = sample_rate
        self.low_freq_mel = low_freq_mel
        self.frame_size = frame_size  # in ms
        self.num_ceps = num_ceps
        self.cep_lifter = cep_lifter  # 
        self.freq_fft = freq_fft  # Frequenz of the FFT
        
        self.new_len = new_len  # Length of the 
        self.len_ms = new_len/sample_rate * 1000  # Frame lenght in ms
        
        self.model_path = os.path.join('models', mode + '.model')
        self.p_path = os.path.join('pickles', mode + '.p')
        
