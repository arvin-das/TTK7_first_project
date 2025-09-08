import numpy as np
from tftb.processing.linear import ShortTimeFourierTransform
import pywt
from scipy.signal import spectrogram, get_window, stft

class NonAnalyticalAnalyser:
    def __init__(self, signal, sampling_period):
        self.signal = signal
        self.sampling_period = sampling_period
    
    def fast_fourier(self):
        return np.fft.rfft(self.signal)
    
    def short_time_fourier(self):
        analysis = ShortTimeFourierTransform(self.signal)
        analysis.run()
        return analysis
    
    def spectrogram(self, window_type='hann', nperseg=256, noverlap=128):
        window = get_window(window_type, nperseg, fftbins=True)
        f, t, Sxx = spectrogram(self.signal, window=window, nperseg=nperseg, noverlap=noverlap)
        return f, t, Sxx

    def compute_stft(self, window_type='hann', nperseg=256, noverlap=128, Fs = 100):
        window = get_window(window_type, nperseg, fftbins=True)

        f, t, Zxx = stft(
            self.signal, fs=Fs, window=window, nperseg=nperseg, noverlap=noverlap, detrend=False, 
            return_onesided=True, padded=False
        )
        return f, t, Zxx
    
    def wavelet_transform(self):
        # Wavelet type
        wavelet = "cmor1.5-1.0"
        # logarithmic scale for scales, as suggested by Torrence & Compo (A paper):
        widths = np.geomspace(1, 1024, num=100)
        cwtmatr, freqs = pywt.cwt(self.signal, widths, wavelet, sampling_period=self.sampling_period)
        # absolute take absolute value of complex result
        cwtmatr = np.abs(cwtmatr[:-1, :-1])
        return cwtmatr, freqs
        
    
    