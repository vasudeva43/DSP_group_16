import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt

# Function to add lag (in seconds) and reduce power
def process_signal(signal, sample_rate, lag_seconds=0.05, power_reduction_db=-20):
    # Convert dB reduction to linear scale
    power_reduction = 10 ** (power_reduction_db / 20)
    
    # Apply power reduction
    processed_signal = signal * power_reduction

    # Add lag by padding with zeros
    lag_samples = int(lag_seconds * sample_rate)
    processed_signal = np.pad(processed_signal, (lag_samples, 0), mode='constant')
    
    return processed_signal

# Function to mix two signals
def mix_signals(primary_signal, secondary_signal):
    # Ensure both signals have the same length by padding the shorter one with zeros
    if len(primary_signal) > len(secondary_signal):
        secondary_signal = np.pad(secondary_signal, (0, len(primary_signal) - len(secondary_signal)), mode='constant')
    else:
        primary_signal = np.pad(primary_signal, (0, len(secondary_signal) - len(primary_signal)), mode='constant')
    
    # Mix both signals
    mixed_signal = primary_signal + secondary_signal
    return mixed_signal

# Function to load and save .wav files
def load_wav(file_path):
    sample_rate, data = wavfile.read(file_path)
    return sample_rate, data

def save_wav(file_path, sample_rate, data):
    wavfile.write(file_path, sample_rate, data.astype(np.int16))

# Function to simulate the test signal by adding lag, reducing power, and mixing additional speech
def simulate_test_signal(test_signal_path, additional_speech_path, output_path, lag_seconds=0.05, power_reduction_db=-6):
    # Load primary test signal (to be filtered)
    sample_rate, test_signal = load_wav(test_signal_path)

    # Load additional speech signal (the noise we want to filter out)
    _, additional_speech = load_wav(additional_speech_path)

    # Add lag and reduce power for the test signal
    processed_test_signal = process_signal(test_signal, sample_rate, lag_seconds, power_reduction_db)

    # Mix the test signal with the additional speech
    final_signal = mix_signals(processed_test_signal, additional_speech)

    # Save the mixed signal as a .wav file
    save_wav(output_path, sample_rate, final_signal)
    print(f"Processed and mixed signal saved to {output_path}")




# Test the function
simulate_test_signal('test_signal.wav', 'additional_speech.wav', 'output_test_signal.wav', lag_seconds=0.05, power_reduction_db=-6)

