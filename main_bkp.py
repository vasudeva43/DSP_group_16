import numpy as np
from scipy.io import wavfile
from scipy.signal import lfilter

# Constants
NUM_HARMONICS = 40  # Number of harmonics to decompose
STEP_SIZE = 0.005  # FXNLMS step size
WINDOW_SIZE = 960  # Window size for F0 estimation (20ms window)

# Function to read .wav file
def read_wav(file_path):
    sample_rate, data = wavfile.read(file_path)
    return sample_rate, data.astype(np.float32)

# Function to save .wav file
def save_wav(file_path, sample_rate, data):
    wavfile.write(file_path, sample_rate, data.astype(np.int16))

def autocorrelation(signal, sample_rate):
    """Estimate fundamental frequency using autocorrelation method."""
    corr = np.correlate(signal, signal, mode='full')
    corr = corr[len(corr)//2:]  # Use positive lags
    d = np.diff(corr)
    start = np.where(d > 0)[0][0]  # First positive difference
    peak = np.argmax(corr[start:]) + start
    f0 = sample_rate / peak
    return f0

def harmonic_decomposition(signal, f0, sampling_rate, num_harmonics):
    """Decompose signal into harmonic components."""
    harmonics = []
    omega_0 = 2 * np.pi * f0 / sampling_rate
    for l in range(1, num_harmonics + 1):
        harmonic = np.cos(l * omega_0 * np.arange(len(signal)))
        harmonics.append(harmonic)
    return np.array(harmonics)

def quadrature_harmonics(harmonics):
    """Generate quadrature harmonics (90-degree phase shift)."""
    quadrature = np.sin(np.angle(harmonics))
    return quadrature

def fxnlms_update(harmonics, quadrature, error_signal, weights, step_size):
    """Update FXNLMS weights for harmonics and their quadrature components."""
    num_harmonics = harmonics.shape[0]
    for l in range(num_harmonics):
        # Update for both harmonic and quadrature components
        # Applying element-wise multiplication with error_signal
        weights[0, l] += step_size * np.sum(error_signal * harmonics[l])
        weights[1, l] += step_size * np.sum(error_signal * quadrature[l])
    return weights

def generate_anti_noise(harmonics, quadrature, weights, length):
    """Generate anti-noise signal using weighted harmonics and quadrature."""
    anti_noise = np.zeros(length)
    for l in range(harmonics.shape[0]):
        anti_noise += weights[0, l] * harmonics[l] + weights[1, l] * quadrature[l]
    return anti_noise

# Initialize weights for the FXNLMS adaptive filter
#weights = np.zeros((2, NUM_HARMONICS))  # Separate weights for harmonics and quadrature components

def process_wav_files(input_wav, error_wav, output_wav):

    weights = np.zeros((2, NUM_HARMONICS))  # Separate weights for harmonics and quadrature components
    # Load input and error .wav files
    sample_rate, input_signal = read_wav(input_wav)
    _, error_signal = read_wav(error_wav)

    # Ensure both signals are the same length
    min_len = min(len(input_signal), len(error_signal))
    input_signal = input_signal[:min_len]
    error_signal = error_signal[:min_len]
    #print(f"Input signal: {len(input_signal)} samples")

    # Estimate fundamental frequency using autocorrelation on a 20ms window
    f0 = autocorrelation(input_signal[:WINDOW_SIZE], sample_rate)

    # Harmonic decomposition
    harmonics = harmonic_decomposition(input_signal, f0, sample_rate, NUM_HARMONICS)
    quadrature = quadrature_harmonics(harmonics)

    # FXNLMS adaptive filtering update
    weights = fxnlms_update(harmonics, quadrature, error_signal, weights, STEP_SIZE)

    # Generate anti-noise signal
    anti_noise = generate_anti_noise(harmonics, quadrature, weights, len(input_signal))
    #print(f"Anti-noise signal: {len(anti_noise)} samples")

    # Save the anti-noise signal to output .wav
    save_wav(output_wav, sample_rate, anti_noise)

# Example usage with .wav files
process_wav_files('./sample_data/modified/main_signal.wav', './sample_data/modified/error_signal.wav', './sample_data/modified/output_anti_noise.wav')
