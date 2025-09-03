import os
import matplotlib.pyplot as plt
from transforms import fft
import numpy as np

Fs = 100 



file_name = "Signal2_2018.csv"
file_path = os.path.relpath(file_name)
print(f"Relative path: {file_path}")

with open(file_path, "r") as csv_file:
    data = csv_file.read().strip().split(",")
    data = [float(x) for x in data]

time_axis = np.linspace(0, len(data)/Fs, len(data))

plt.plot(time_axis, data)
plt.xlabel("Time (s)")
plt.ylabel("Signal amplitude")
plt.title("Raw signal")
plt.grid()
plt.show()

fft_data = fft.fft_transform(data)
fft_axis = np.fft.fftfreq(len(data), 1/Fs)

plt.figure()
plt.plot(fft_axis, fft_data)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.title("FFT of the signal")
plt.grid()
plt.show()
