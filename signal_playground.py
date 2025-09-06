import numpy as np
from src.signal_mod.signal_modifier import SignalModifier
from src.transforms.analytical import AnalyticalAnalyser
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

