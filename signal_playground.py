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
sampling_period = np.diff(time_axis).mean()
ax1.plot(time_axis, modifier.get_original_signal())
ax2.plot(time_axis, modifier.get_modified_signal())
ax2.set_xlabel("Time [s]")
plt.show()


analytical_analyser = AnalyticalAnalyser(modifier.get_original_signal())
wv = analytical_analyser.wigner_ville_distribution()
wv.plot(kind='contour', show_tf=True)


non_analytical_analyser = NonAnalyticalAnalyser(modifier.get_original_signal(), sampling_period)
fft = non_analytical_analyser.fast_fourier()
x_fft = np.fft.rfftfreq(len(modifier.get_original_signal()), sampling_period)
plt.plot(x_fft, fft)
plt.title('FFT')
plt.ylabel('Energy')
plt.xlabel('Frequency (Hz)')
plt.grid()
plt.show()


stft = non_analytical_analyser.short_time_fourier()
stft.plot()


cwtmatr, freqs = non_analytical_analyser.wavelet_transform()
fig, axs = plt.subplots(1, 1)
pcm = axs.pcolormesh(time_axis, freqs, cwtmatr)
axs.set_yscale("log")
axs.set_xlabel("Time (s)")
axs.set_ylabel("Frequency (Hz)")
axs.set_title("Continuous Wavelet Transform (Scaleogram)")
fig.colorbar(pcm, ax=axs)
plt.show()