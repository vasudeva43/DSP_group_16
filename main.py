import numpy as np
import pyaudio
from scipy.signal import lfilter

# Constants
RATE = 48000  # Sampling rate in Hz
CHUNK = 1024  # Audio chunk size
NUM_HARMONICS = 40  # Number of harmonics to decompose
STEP_SIZE = 0.005  # FXNLMS step size
DELAY_SAMPLES = 6  # Delay for prediction (samples)
WINDOW_SIZE = 960  # Window size for F0 estimation (20ms window)

# Initialize PyAudio
audio = pyaudio.PyAudio()

# Open audio input and output streams
input_stream = audio.open(format=pyaudio.paFloat32, channels=1, rate=RATE, input=True, frames_per_buffer=CHUNK)
output_stream = audio.open(format=pyaudio.paFloat32, channels=1, rate=RATE, output=True, frames_per_buffer=CHUNK)

def autocorrelation(signal):
    """Estimate fundamental frequency using autocorrelation method."""
    corr = np.correlate(signal, signal, mode='full')
    corr = corr[len(corr)//2:]  # Use positive lags
    d = np.diff(corr)
    start = np.where(d > 0)[0][0]  # First positive difference
    peak = np.argmax(corr[start:]) + start
    f0 = RATE / peak
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
        weights[0, l] += step_size * error_signal * harmonics[l]
        weights[1, l] += step_size * error_signal * quadrature[l]
    return weights

def generate_anti_noise(harmonics, quadrature, weights):
    """Generate anti-noise signal using weighted harmonics and quadrature."""
    anti_noise = np.sum(weights[0] * harmonics + weights[1] * quadrature, axis=0)
    return anti_noise

# Initialize weights for the FXNLMS adaptive filter
weights = np.zeros((2, NUM_HARMONICS))  # Separate weights for harmonics and quadrature components

try:
    while True:
        # Read audio input
        data = np.frombuffer(input_stream.read(CHUNK), dtype=np.float32)

        # Estimate fundamental frequency using autocorrelation on a 20ms window
        f0 = autocorrelation(data[:WINDOW_SIZE])

        # Harmonic decomposition
        harmonics = harmonic_decomposition(data, f0, RATE, NUM_HARMONICS)
        quadrature = quadrature_harmonics(harmonics)

        # Simulate error signal (e.g., residual noise at error microphone)
        error_signal = data  # In practice, this would come from an error microphone

        # FXNLMS adaptive filtering update
        weights = fxnlms_update(harmonics, quadrature, error_signal, weights, STEP_SIZE)

        # Generate anti-noise signal
        anti_noise = generate_anti_noise(harmonics, quadrature, weights)

        # Output the anti-noise signal to the headphones
        output_stream.write(anti_noise.tobytes())

except KeyboardInterrupt:
    print("ANC stopped.")

finally:
    # Close the streams and terminate PyAudio
    input_stream.stop_stream()
    input_stream.close()
    output_stream.stop_stream()
    output_stream.close()
    audio.terminate()


# This is the code