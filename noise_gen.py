import numpy as np
from scipy.io.wavfile import write

# Constants
length = 100000
amp = 2
RATE = 8000  # Add a sample rate definition (e.g., 8kHz)

# Function to generate random noise
def noise_gen(amp, length, Rate):
    test_noise = 2 * amp * (np.random.random(length) - 0.5)
    error_noise = 2 * amp * (np.random.random(length) - 0.5)

    write("./modified/test_noise.wav", Rate, test_noise.astype(np.float32))
    write("./modified/error_noise.wav", Rate, error_noise.astype(np.float32))

# Call the function to generate noise
noise_gen(amp, lengthRATE)
