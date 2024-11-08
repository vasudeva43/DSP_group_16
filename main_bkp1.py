import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, lfilter

class HMDANC:
    def __init__(self, num_harmonics=40, step_size=0.01, filter_length=1024, leakage=0.9999):
        self.num_harmonics = num_harmonics
        self.step_size = step_size
        self.filter_length = filter_length
        self.weights = np.zeros(num_harmonics)
        self.leakage = leakage

    def preprocess_signal(self, signal, fs):
        # Normalize
        signal = signal / np.max(np.abs(signal))
        
        # Apply bandpass filter (50-2000 Hz)
        nyq = fs / 2
        b, a = butter(4, [50/nyq, 2000/nyq], btype='band')
        return lfilter(b, a, signal)

    def estimate_f0(self, signal, fs, frame_length=1024):
        frames = np.array([signal[i:i+frame_length] for i in range(0, len(signal)-frame_length, frame_length//2)])
        f0s = []
        for frame in frames:
            corr = np.correlate(frame, frame, mode='full')
            corr = corr[len(corr)//2:]
            peak = np.argmax(corr[20:]) + 20  # Avoid first 20 samples to skip first peak
            f0 = fs / peak
            f0s.append(np.clip(f0, 50, 500))  # Limit to reasonable range
        return np.array(f0s)

    def generate_harmonics(self, signal, f0s, fs):
        t = np.arange(len(signal)) / fs
        harmonics = np.zeros((self.num_harmonics, len(signal)))
        for i, f0 in enumerate(f0s):
            start = i * len(signal) // len(f0s)
            end = (i + 1) * len(signal) // len(f0s)
            for n in range(1, self.num_harmonics + 1):
                harmonics[n-1, start:end] = np.cos(2 * np.pi * n * f0 * t[start:end])
        return harmonics

    def adaptive_filter(self, harmonics, error):
        anti_noise = np.zeros_like(error)
        for n in range(len(error)):
            x = harmonics[:, n]
            y = np.dot(self.weights, x)
            anti_noise[n] = y
            
            # Normalize step size
            power = np.dot(x, x)
            normalized_step = self.step_size / (power + 1e-6)
            
            # Update weights with leakage
            self.weights = self.leakage * self.weights + normalized_step * error[n] * x
            
            # Constrain weights
            self.weights = np.clip(self.weights, -1, 1)
        
        # Constrain anti-noise amplitude
        anti_noise = np.tanh(anti_noise)
        return anti_noise

    def process(self, input_wav, error_wav, output_wav):
        fs, input_signal = wavfile.read(input_wav)
        _, error_signal = wavfile.read(error_wav)

        min_len = min(len(input_signal), len(error_signal))
        input_signal = input_signal[:min_len]
        error_signal = error_signal[:min_len]

        input_signal = self.preprocess_signal(input_signal, fs)
        error_signal = self.preprocess_signal(error_signal, fs)

        f0s = self.estimate_f0(input_signal, fs)
        harmonics = self.generate_harmonics(input_signal, f0s, fs)

        anti_noise = self.adaptive_filter(harmonics, error_signal)

        anti_noise = anti_noise * 32767
        wavfile.write(output_wav, fs, anti_noise.astype(np.int16))

# Usage
anc = HMDANC()
anc.process('./sample_data/modified/main_signal.wav', './sample_data/modified/error_signal.wav', './sample_data/modified/output_anti_noise.wav')