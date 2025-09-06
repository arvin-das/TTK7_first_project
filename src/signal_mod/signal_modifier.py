import numpy as np

class SignalModifier:
    def __init__(self, signal: np.ndarray, sampling_freq: int):
        self._original_signal = signal
        self._modified_signal = signal
        self._time = np.linspace(0, len(signal)/sampling_freq, len(signal))
        
    def add_offset(self, offset: float):
        self._modified_signal = np.add(self._modified_signal, offset)
        
    def add_noise(self, SNR_dB: int):
        signal_mean_power = np.mean(np.square(self._original_signal))
        signal_mean_power_dB = 10 * np.log10(signal_mean_power)
        noise_power = 10**((signal_mean_power_dB - SNR_dB)/10)
        
        noise = np.random.normal(0, np.sqrt(noise_power), self._original_signal.size)
        self._modified_signal = np.add(self._modified_signal, noise)

    def add_component(self, k: int):
        frequencies = k*self._time
        amplitude = np.max(self._original_signal)/5
        chirp_component = amplitude * np.sin(2*np.pi*frequencies*self._time)
        
        self._modified_signal = np.add(self._modified_signal, chirp_component)
    
    def get_modified_signal(self):
        return self._modified_signal
    
    def get_original_signal(self):
        return self._original_signal