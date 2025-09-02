
import os
import matplotlib.pyplot as plt

from transforms import fft

file_name = "Signal2_2018.csv"
file_path = os.path.relpath(file_name)
print(f"Relative path: {file_path}")

with open(file_path, "r") as csv_file:
    data = csv_file.read().strip().split(",")
    data = [float(x) for x in data]

print(fft.test())

plt.plot(data)
plt.xlabel("Sample")
plt.ylabel("Signal amplitude")
plt.title("Raw signal")
plt.grid()
plt.show()
