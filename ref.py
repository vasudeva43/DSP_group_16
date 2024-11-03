import numpy as np
from scipy.io import wavfile
from scipy.signal import lfilter

# Constants
RATE = 48000  # Sampling rate in Hz
CHUNK = 1024
STEP_SIZE = 0.005

# Function to load .wav file
def load_wav(file_path):
    sample_rate, data = wavfile.read(file_path)
    return sample_rate, data.astype(np.float32)

# Function to save .wav file
def save_wav(file_path, sample_rate, data):
    wavfile.write(file_path, sample_rate, data.astype(np.int16))

# Reference ANC - Filtered-X LMS function
def fxlms_anc(test_signal, error_signal, step_size=STEP_SIZE):
    filter_length = 256  # Set filter length for FXLMS
    adaptive_filter = np.zeros(filter_length)
    output_signal = np.zeros_like(test_signal)

    for i in range(filter_length, len(test_signal)):
        # Extract the current frame
        x_frame = test_signal[i-filter_length:i]
        
        # Estimate the error
        estimated_error = error_signal[i] - np.dot(adaptive_filter, x_frame)

        # Update the adaptive filter coefficients
        adaptive_filter += step_size * estimated_error * x_frame

        # Generate the anti-noise signal
        output_signal[i] = -np.dot(adaptive_filter, x_frame)

    return output_signal

# Performance Comparison Function
def compare_anc_performance(test_signal_path, error_signal_path, output_ref_path, output_hmd_path):
    # Load signals
    sample_rate, test_signal = load_wav(test_signal_path)
    _, error_signal = load_wav(error_signal_path)

    # Ensure signals are the same length
    min_len = min(len(test_signal), len(error_signal))
    test_signal = test_signal[:min_len]
    error_signal = error_signal[:min_len]

    # Reference ANC - Filtered-X LMS
    ref_anc_output = fxlms_anc(test_signal, error_signal)

    # Load HMD-ANC result from speech_proc's final output (to compare results)
    _, hmd_anc_output = load_wav(output_hmd_path)

    # Ensure the HMD-ANC output is of the same length for fair comparison
    hmd_anc_output = hmd_anc_output[:min_len]

    # Save reference ANC output to .wav
    save_wav(output_ref_path, sample_rate, ref_anc_output)

    # Compute Residual Errors
    ref_residual_error = np.mean((error_signal - ref_anc_output) ** 2)
    hmd_residual_error = np.mean((error_signal - hmd_anc_output) ** 2)

    print("Performance Comparison:")
    print(f"FXLMS Residual Error: {ref_residual_error}")
    print(f"HMD-ANC Residual Error: {hmd_residual_error}")

# Paths to input and output files
test_signal_path = './modified/main_signal.wav'
error_signal_path = './modified/error_signal.wav'
output_ref_path = './modified/output_ref_anc.wav'
output_hmd_path = './modified/output_anti_noise.wav'

# Run the comparison
compare_anc_performance(test_signal_path, error_signal_path, output_ref_path, output_hmd_path)
