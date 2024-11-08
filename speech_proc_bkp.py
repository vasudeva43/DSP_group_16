import os
import random
import numpy as np
from scipy.io import wavfile

# Function to add lag (in seconds) and reduce power
def process_signal(signal, sample_rate, lag_seconds=0.05, power_reduction_db=-20):
    power_reduction = 10 ** (power_reduction_db / 20)
    processed_signal = signal * power_reduction
    lag_samples = int(lag_seconds * sample_rate)
    processed_signal = np.pad(processed_signal, (lag_samples, 0), mode='constant')
    return processed_signal

# Function to mix multiple signals
def mix_signals(signals):
    max_length = max([len(signal) for signal in signals])
    mixed_signal = np.zeros(max_length)
    for signal in signals:
        reduction_factor = random.uniform(0.2, 0.4)
        padded_signal = np.pad(signal, (0, max_length - len(signal)), mode='constant')
        mixed_signal += padded_signal*reduction_factor
    return mixed_signal

# Function to load .wav files
def load_wav(file_path):
    sample_rate, data = wavfile.read(file_path)
    return sample_rate, data

def save_wav(file_path, sample_rate, data):
    wavfile.write(file_path, sample_rate, data.astype(np.int16))

# Function to load multiple .wav files from a folder
def load_wav_files_from_folder(folder_path, limit=None):
    wav_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.wav')]
    loaded_signals = []
    for file in wav_files[:limit]:
        sample_rate, data = load_wav(file)
        loaded_signals.append(data)
    return sample_rate, loaded_signals

# Function to simulate the test signal by adding lag, reducing power, and mixing multiple additional speeches
def simulate_test_signal(folder_path, output_path, main_path, lag_seconds=0.05, power_reduction_db=-6, n_amp = 1):
    # List all .wav files in the folder
    wav_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.wav')]

    #sample_rate, test_noise = load_wav(tnoise_path)
    #sample_rate, error_noise = load_wav(enoise_path)

    # Randomly select a file as the test signal
    test_signal_file = random.choice(wav_files)
    remaining_files = [file for file in wav_files if file != test_signal_file]

    # Load primary test signal (to be filtered)
    sample_rate, test_signal = load_wav(test_signal_file)
    print(f"Test Signal: {test_signal_file}")

    # Load additional speech signals from the remaining files
    sample_rate, additional_speeches = load_wav_files_from_folder(folder_path, limit=20)

    # Add lag and reduce power for the test signal
    processed_test_signal = process_signal(test_signal, sample_rate, lag_seconds, power_reduction_db)

    # Mix all the additional speech signals
    additional_speech_mixed = mix_signals(additional_speeches)

    length = len(test_signal)

    test_noise = 2 * n_amp * (np.random.random(length) - 0.5)
    error_noise = 2 * n_amp * (np.random.random(length) - 0.5)

    # Mix the processed test signal with the additional speech noise
    final_signal = mix_signals([processed_test_signal, additional_speech_mixed, error_noise])
    test_signal1 = mix_signals([test_signal, test_noise, additional_speech_mixed])

    # Save the mixed signal as a .wav file
    save_wav(output_path, sample_rate, final_signal)
    save_wav(main_path, sample_rate, test_signal1)
    save_wav(original_path, sample_rate, test_signal)
    print(f"Processed and mixed signal saved to {output_path}")

# Relative paths to the folder and output file
folder_path = './sample_data/original'
#tnoise_path = './sample_data/modified/test_noise.wav'
#enoise_path = './sample_data/modified/error_noise.wav'
output_path = './sample_data/modified/error_signal.wav'
main_path = './sample_data/modified/main_signal.wav'
original_path = './sample_data/modified/orig_signal.wav'

# Simulate the test signal
simulate_test_signal(folder_path, output_path, main_path, lag_seconds=0.05, power_reduction_db=-6, n_amp = 1)