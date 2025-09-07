import numpy as np
from src.signal_mod.signal_modifier import SignalModifier
from src.transforms.analytical import AnalyticalAnalyser
from src.transforms.non_analytical import NonAnalyticalAnalyser
import matplotlib.pyplot as plt

SAMPLING_FREQUENCY = 100 # found in signal generation script

def read_signal(path):
    with open(path, 'r') as eeg_csv:
        eeg = np.array(eeg_csv.read().strip().split(','), dtype=float)
        
    return eeg
        

raw_signal = read_signal('Signal2_2018.csv')

modifier = SignalModifier(raw_signal, sampling_freq=SAMPLING_FREQUENCY)

modifier.add_offset(1e-6)
modifier.add_noise(5)
modifier.add_component(0.3)

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
time_axis = np.linspace(0, len(raw_signal)/SAMPLING_FREQUENCY, len(raw_signal))
ax1.plot(time_axis, modifier.get_original_signal())
ax2.plot(time_axis, modifier.get_modified_signal())
ax2.set_xlabel("Time [s]")
plt.show()

analytical_analyser = AnalyticalAnalyser(modifier.get_original_signal())
wv = analytical_analyser.wigner_ville_distribution()

wv.plot(kind='contour', show_tf=True)

non_analytical_analyser = NonAnalyticalAnalyser(modifier.get_original_signal())

fft_axis, fft_values = non_analytical_analyser.fast_fourier()

plt.figure()
plt.plot(fft_axis,  np.abs(fft_values))
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.title("FFT of the signal")
plt.grid()
plt.show()  


f, t, Sxx = non_analytical_analyser.spectrogram(window_type='boxcar', nperseg=256, noverlap=128)
Sxx_dB = 10*np.log10(Sxx + 1e-12)

plt.figure(figsize=(8, 4))
plt.pcolormesh(t, f, Sxx_dB, shading='gouraud')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [s]')
cbar = plt.colorbar()
cbar.set_label('Power spectral density [dB/Hz]')
plt.tight_layout()
plt.show()
