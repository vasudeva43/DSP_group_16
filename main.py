import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, filtfilt, hilbert, lfilter
from scipy.fft import fft, ifft

# Constants
NUM_HARMONICS = 60
STEP_SIZE = 0.01
WINDOW_SIZE = 1024

# ... (keep all the previous functions as they are) ...

def secondary_path_model(x, b):
    return lfilter(b, [1.0], x)

def hmd_anc(input_signal, error_signal, sample_rate, D=50, L=60, mu=0.01):
    input_signal = input_signal / np.max(np.abs(input_signal))
    
    b, a = butter(4, [50/24000, 2000/24000], btype='band')
    input_signal = filtfilt(b, a, input_signal)
    
    # Estimate fundamental frequency
    f0 = autocorrelation(input_signal[:WINDOW_SIZE], sample_rate)
    
    # Harmonic decomposition
    harmonics = harmonic_decomposition(input_signal, f0, sample_rate, L)
    quadrature = quadrature_harmonics(harmonics)
    
    e = np.zeros_like(input_signal, dtype=np.float32)
    y = np.zeros_like(input_signal, dtype=np.float32)
    w_in = np.zeros((L, 1), dtype=np.float32)
    w_quad = np.zeros((L, 1), dtype=np.float32)
    
    power = np.zeros(L, dtype=np.float32)
    beta = 0.995
    leak = 0.9999
    
    # Simple secondary path model (can be improved with actual measurements)
    b_secondary = np.ones(10) / 10
    
    for n in range(D, len(input_signal)):
        power = beta * power + (1-beta) * (harmonics[:, n-D]**2 + quadrature[:, n-D]**2)
        
        mu_n = mu / (power + 1e-6)
        
        x_in = harmonics[:, n-D].reshape(-1, 1)
        x_quad = quadrature[:, n-D].reshape(-1, 1)
        
        y[n] = np.sum(w_in * x_in + w_quad * x_quad)
        
        filtered_y = secondary_path_model(y[:n+1], b_secondary)[-1]
        e[n] = error_signal[n] - filtered_y
        
        w_in = leak * w_in + mu_n.reshape(-1, 1) * e[n] * x_in
        w_quad = leak * w_quad + mu_n.reshape(-1, 1) * e[n] * x_quad
        
        np.clip(w_in, -1, 1, out=w_in)
        np.clip(w_quad, -1, 1, out=w_quad)
    
    return y

# Paths to input and output files
input_signal_path = './sample_data/modified/main_signal.wav'
error_signal_path = './sample_data/modified/error_signal.wav'
output_hmd_path = './sample_data/modified/output_hmd_anc.wav'

# Load signals
sample_rate, input_signal = read_wav(input_signal_path)
_, error_signal = read_wav(error_signal_path)

# Ensure both signals have the same length
min_length = min(len(input_signal), len(error_signal))
input_signal = input_signal[:min_length]
error_signal = error_signal[:min_length]

# Run HMD-ANC
hmd_anc_output = hmd_anc(input_signal, error_signal, sample_rate)

# Save HMD-ANC output
save_wav(output_hmd_path, sample_rate, hmd_anc_output)