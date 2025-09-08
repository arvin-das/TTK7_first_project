import numpy as np
from tftb.processing.linear import ShortTimeFourierTransform
import pywt
from scipy.signal import spectrogram, get_window, stft

class NonAnalyticalAnalyser:
    def __init__(self, signal, sampling_period):
        self.signal = signal
        self.sampling_period = sampling_period
        self.Fs = 1/sampling_period
    
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

    def compute_stft(self, window_type='hann', nperseg=256, noverlap=128, Fs = 100, nfft = 1024):
        window = get_window(window_type, nperseg, fftbins=True)

        f, t, Zxx = stft(
            self.signal, fs=Fs, window=window, nperseg=nperseg, noverlap=noverlap, detrend=False, 
            return_onesided=True, padded=False, nfft= nfft
        )
        return f, t, Zxx
    
    def wavelet_transform(self, wavelet_type="cmor1.5-1.0", num_freqs=100, method ="conv", fmax = 50):
        # Wavelet type
        wavelet = wavelet_type
        # logarithmic scale for scales, as suggested by Torrence & Compo (A paper):

        fmin = 0.5
        freqs  = np.geomspace(fmax, fmin, num=num_freqs)

        cf = pywt.central_frequency(wavelet)                 # wavelet center freq (cycles/sample)
        scales = (cf * self.Fs) / freqs 

        cwtmatr, out_freqs  = pywt.cwt(self.signal, scales, wavelet, sampling_period=self.sampling_period, method=method)

        # absolute take absolute value of complex result
        cwtmatr = np.abs(cwtmatr)

        t = np.arange(self.signal.size) / self.Fs
        return cwtmatr, out_freqs, t
        
    
    