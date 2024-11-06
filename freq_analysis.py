import matplotlib.pyplot as plt
from scipy.signal import spectrogram
import numpy as np

# Define the paths for each signal
signal_paths = ['path/to/signal1', 'path/to/signal2', 'path/to/signal3']

# Load the signals from the paths
signals = [np.load(path) for path in signal_paths]
labels = ['Signal 1', 'Signal 2', 'Signal 3']
colors = ['viridis', 'plasma', 'inferno']

plt.figure(figsize=(10, 6))

for i, signal in enumerate(signals):
    f, t, Sxx = spectrogram(signal, RATE)
    plt.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud', cmap=colors[i], alpha=0.5, label=labels[i])

plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.title('Spectrogram of Multiple Signals')
plt.legend()
plt.colorbar(label='Intensity [dB]')
plt.show()