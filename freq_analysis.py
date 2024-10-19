import matplotlib.pyplot as plt
from scipy.signal import spectrogram

f, t, Sxx = spectrogram(test_signal, RATE)
plt.pcolormesh(t, f, 10 * np.log10(Sxx))
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.title('Spectrogram of Test Signal')
plt.show()
