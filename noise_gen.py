import numpy as np
from scipy.io.wavfile import write

RATE = 48000  # Sample rate
DURATION = 10  # 10 seconds
FREQUENCY = 400  # 400 Hz test tone

t = np.linspace(0, DURATION, int(RATE * DURATION), endpoint=False)
test_signal = 0.5 * np.sin(2 * np.pi * FREQUENCY * t)

write("test_noise.wav", RATE, test_signal.astype(np.float32))

#for noise

#Record Error Signal:
#Place a microphone at the error position (near the ear) and record the residual noise after ANC is applied