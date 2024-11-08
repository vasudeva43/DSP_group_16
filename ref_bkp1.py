import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, filtfilt

# Constants
RATE = 48000  # Sampling rate in Hz
CHUNK = 1024
STEP_SIZE = 0.01

def load_wav(file_path):
    sample_rate, data = wavfile.read(file_path)
    data = data.astype(np.float32)
    if np.max(np.abs(data)) > 0:
        data = data / np.max(np.abs(data))
    return sample_rate, data

def save_wav(file_path, sample_rate, data):
    max_val = np.max(np.abs(data))
    if max_val > 0:
        data = data * (32767 / max_val)
    wavfile.write(file_path, sample_rate, data.astype(np.int16))

def fxlms_anc(test_signal, error_signal, step_size=STEP_SIZE):
    filter_length = 512
    delay = 50
    
    test_signal = test_signal - np.mean(test_signal)
    error_signal = error_signal - np.mean(error_signal)
    
    b, a = butter(4, [50/24000, 2000/24000], btype='band')
    test_signal = filtfilt(b, a, test_signal)
    
    w = np.zeros(filter_length)
    x_buffer = np.zeros(filter_length)
    output_signal = np.zeros_like(test_signal)
    
    power = np.zeros_like(test_signal)
    beta = 0.999
    
    for i in range(delay, len(test_signal)):
        x_buffer = np.roll(x_buffer, 1)
        x_buffer[0] = test_signal[i]
        
        power[i] = beta * power[i-1] + (1-beta) * x_buffer[0]**2
        mu = step_size / (power[i] + 1e-6)
        
        output_signal[i] = -np.dot(w, x_buffer)
        
        e = error_signal[i] + output_signal[i]
        
        w = w + mu * e * x_buffer
        w = np.clip(w, -1, 1)
    
    return output_signal

def compare_anc_performance(test_signal_path, error_signal_path, output_ref_path, output_hmd_path):
    # Load signals
    sample_rate, test_signal = load_wav(test_signal_path)
    _, error_signal = load_wav(error_signal_path)
    
    # Ensure signals are the same length
    min_len = min(len(test_signal), len(error_signal))
    test_signal = test_signal[:min_len]
    error_signal = error_signal[:min_len]
    
    # Reference ANC - FXLMS
    ref_anc_output = fxlms_anc(test_signal, error_signal)
    
    # Load HMD-ANC result
    _, hmd_anc_output = load_wav(output_hmd_path)
    hmd_anc_output = hmd_anc_output[:min_len]
    
    # Save reference ANC output
    save_wav(output_ref_path, sample_rate, ref_anc_output)
    
    # Compute error reduction in dB
    original_power = np.mean(error_signal**2)
    ref_residual_power = np.mean((error_signal - ref_anc_output)**2)
    hmd_residual_power = np.mean((error_signal - hmd_anc_output)**2)
    
    ref_reduction_db = 10 * np.log10(original_power / ref_residual_power)
    hmd_reduction_db = 10 * np.log10(original_power / hmd_residual_power)
    
    print("Performance Comparison:")
    print(f"FXLMS Noise Reduction: {ref_reduction_db:.2f} dB")
    print(f"HMD-ANC Noise Reduction: {hmd_reduction_db:.2f} dB")

# Paths to input and output files
test_signal_path = './sample_data/modified/main_signal.wav'
error_signal_path = './sample_data/modified/error_signal.wav'
output_ref_path = './sample_data/modified/output_ref_anc.wav'
output_hmd_path = './sample_data/modified/output_anti_noise.wav'

# Run the comparison
compare_anc_performance(test_signal_path, error_signal_path, output_ref_path, output_hmd_path)