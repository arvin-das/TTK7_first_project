import numpy as np
from src.signal_mod.signal_modifier import SignalModifier
from src.transforms.analytical import AnalyticalAnalyser
from src.transforms.non_analytical import NonAnalyticalAnalyser
import matplotlib.pyplot as plt

SAMPLING_FREQUENCY = 100 # found in signal generation script
EPSILON = 1e-12

def read_signal(path):
    with open(path, 'r') as eeg_csv:
        eeg = np.array(eeg_csv.read().strip().split(','), dtype=float)
        
    return eeg
        

raw_signal = read_signal('Signal2_2018.csv')

modifier = SignalModifier(raw_signal, sampling_freq=SAMPLING_FREQUENCY)

modifier.add_offset(max(modifier.get_original_signal())/2)
# modifier.add_noise(5)
# modifier.add_component(0.3)

# fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
time_axis = np.linspace(0, len(raw_signal)/SAMPLING_FREQUENCY, len(raw_signal))
# sampling_period = np.diff(time_axis).mean()
# ax1.plot(time_axis, modifier.get_original_signal())
# ax2.plot(time_axis, modifier.get_modified_signal())
# ax2.set_xlabel("Time [s]")
# ax1.set_ylabel('Amplitude [V]')
# ax2.set_ylabel('Amplitude [V]')
# ax1.grid()
# ax2.grid()
# plt.title('Original vs Offsetted Signal')
# plt.show()

# plt.plot(time_axis, modifier.get_modified_signal())
# plt.title('Brain Signal')
# plt.xlabel('Time [s]')
# plt.ylabel('Amplitude [V]')
# plt.grid()
# plt.show()

analytical_analyser = AnalyticalAnalyser(modifier.get_modified_signal())

hilbert_signal = analytical_analyser.hilbert_transform()
hilbert_array = np.asarray(hilbert_signal)
envelope = np.abs(hilbert_array).astype(float)

plt.figure()
plt.plot(time_axis, envelope)
plt.plot(time_axis, modifier.get_modified_signal())
plt.title('Hilbert')
plt.xlabel('Time [s]')
plt.ylabel('Voltage [V]')
plt.grid()
plt.show()

hilbert_real = np.real(hilbert_array)
hilbert_imaginary = np.imag(hilbert_array)

plt.figure()
plt.plot(hilbert_real, hilbert_imaginary)
plt.title('Hilbert - Signal with Offset')
plt.ylabel('Imaginary Part (Hilbert Transform)')
plt.xlabel('Real Part (EEG Signal)')
plt.grid()
plt.show()


# wv = analytical_analyser.wigner_ville_distribution()
# wv.plot(kind='cmap', show_tf=True)




sampling_period = 1/SAMPLING_FREQUENCY

non_analytical_analyser = NonAnalyticalAnalyser(modifier.get_modified_signal(), sampling_period)
# fft = non_analytical_analyser.fast_fourier()
# x_fft = np.fft.rfftfreq(len(modifier.get_modified_signal()), sampling_period)
# plt.plot(x_fft, np.abs(fft))
# plt.title('FFT')
# plt.ylabel('Energy')
# plt.xlabel('Frequency (Hz)')
# plt.grid()
# plt.show()

# f, t, Sxx = non_analytical_analyser.spectrogram(window_type='boxcar', nperseg=20, noverlap=10)
# Sxx_dB = 10*np.log10(Sxx + EPSILON)
# plt.figure()
# plt.pcolormesh(t, f, Sxx_dB, shading='gouraud')
# plt.ylabel('Frequency [Hz]')
# plt.xlabel('Time [s]')
# cbar = plt.colorbar()
# cbar.set_label('Power spectral density [dB/Hz]')
# plt.tight_layout()
# plt.show()

# f, t, Zxx = non_analytical_analyser.compute_stft(window_type='boxcar', nperseg=20, 
#                                                  noverlap=10, Fs = SAMPLING_FREQUENCY, nfft=512)
# Zxx_db = 20*np.log10(np.abs(Zxx) + EPSILON)

# plt.pcolormesh(t, f, Zxx_db, shading="gouraud")
# plt.xlabel("Time [s]"); plt.ylabel("Frequency [Hz]")
# plt.colorbar(label="Magnitude [dB]")
# plt.title("STFT (magnitude)")
# plt.tight_layout(); plt.show()


# stft = non_analytical_analyser.short_time_fourier()
# stft.plot()


# wavelet_list = ["cmor0.5-1.0","cgau3", "shan2-1.0", "fbsp2-1.0-1.5", "morl", "gaus2", "mexh"]
# # list over diffrent wavelets we can use. The last 3 are real-valued wavelets while the first 4 are complex-valued wavelets.

# # Fmax should be changed according to the mother wavlet. some wavlet allow much higher frequencies than others.
# cwtmatr, freqs, time = non_analytical_analyser.wavelet_transform(wavelet_type=wavelet_list[0], fmax=51, 
#                                                                  num_freqs=100, method ="conv")    
# fig, axs = plt.subplots(1, 1)
# pcm = axs.pcolormesh(time, freqs, cwtmatr)
# axs.set_yscale("log")
# axs.set_xlabel("Time (s)")
# axs.set_ylabel("Frequency (Hz)")
# axs.set_title("Continuous Wavelet Transform (Scaleogram) - cmor0.5-1.0")
# fig.colorbar(pcm, ax=axs)
# plt.show()


# cwtmatr, freqs, time = non_analytical_analyser.wavelet_transform(wavelet_type=wavelet_list[3], fmax=51, 
#                                                                  num_freqs=100, method ="conv")    
# fig, axs = plt.subplots(1, 1)
# pcm = axs.pcolormesh(time, freqs, cwtmatr)
# axs.set_yscale("log")
# axs.set_xlabel("Time (s)")
# axs.set_ylabel("Frequency (Hz)")
# axs.set_title("Continuous Wavelet Transform (Scaleogram) - fbsp2-1.0-1.5")
# fig.colorbar(pcm, ax=axs)
# plt.show()


