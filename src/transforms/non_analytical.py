from scipy.fft import fft, fftfreq
from scipy.signal import spectrogram, get_window

class NonAnalyticalAnalyser():
    def __init__(self, signal):
        self.signal = signal

    def fast_fourier(self, Fs = 100):
        N = len(self.signal)                  
        X = fft(self.signal)                  
        freqs = fftfreq(N, 1/Fs)              
        return freqs, X
    
    def spectrogram(self, window_type='boxcar', nperseg=256, noverlap=128):
        window = get_window(window_type, nperseg)
        f, t, Sxx = spectrogram(self.signal, window=window, nperseg=nperseg, noverlap=noverlap)
        return f, t, Sxx
    
    def short_time_fourier(self):
        pass
    
    def wavelet_transform(self):
        pass
    
    