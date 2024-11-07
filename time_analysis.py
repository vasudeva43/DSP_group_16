import matplotlib.pyplot as plt
from scipy.io import wavfile
import numpy as np

def analyze_signals(ref_path, hmd_path, orig_path, test_signal_path, sample_rate):
    """
    Analyze and compare different ANC outputs with proper normalization and error calculation
    """
    # Load signals and convert to float32 for consistent processing
    signals = []
    for path in [ref_path, hmd_path, orig_path]:
        rate, signal = wavfile.read(path)
        signal = signal.astype(np.float32)
        # Normalize to [-1, 1] range
        if np.max(np.abs(signal)) > 0:
            signal = signal / np.max(np.abs(signal))
        signals.append(signal)
    
    # Ensure all signals are of the same length
    min_len = min(len(s) for s in signals)
    signals = [s[:min_len] for s in signals]
    
    # Plot time domain comparison
    plt.figure(figsize=(12, 8))
    time = np.linspace(0, min_len / sample_rate, min_len)
    labels = ['Reference ANC', 'HMD ANC', 'Original Signal']
    
    for signal, label in zip(signals, labels):
        plt.plot(time, signal, label=label, alpha=0.7)
    
    plt.ylabel('Normalized Amplitude')
    plt.xlabel('Time [sec]')
    plt.title('Time Domain Signal Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    # Calculate proper MSNE (Mean Square Normalized Error)
    # MSNE = E[(e(n))^2] / E[(d(n))^2]
    # where e(n) is the error signal and d(n) is the desired (original) signal
    orig_power = np.mean(signals[2] ** 2)
    
    msne_ref = np.mean((signals[2] - signals[0]) ** 2) / (orig_power + 1e-10)
    msne_hmd = np.mean((signals[2] - signals[1]) ** 2) / (orig_power + 1e-10)
    
    # Calculate noise reduction in dB
    nr_ref = -10 * np.log10(msne_ref + 1e-10)
    nr_hmd = -10 * np.log10(msne_hmd + 1e-10)
    
    print("\nPerformance Metrics:")
    print(f"Mean Square Normalized Error (Reference ANC): {msne_ref:.4f}")
    print(f"Mean Square Normalized Error (HMD ANC): {msne_hmd:.4f}")
    print(f"Noise Reduction (Reference ANC): {nr_ref:.2f} dB")
    print(f"Noise Reduction (HMD ANC): {nr_hmd:.2f} dB")
    
    # Plot error signals
    plt.figure(figsize=(12, 8))
    error_ref = signals[2] - signals[0]
    error_hmd = signals[2] - signals[1]
    
    plt.subplot(2, 1, 1)
    plt.plot(time, error_ref)
    plt.title('Error Signal - Reference ANC')
    plt.ylabel('Amplitude')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 1, 2)
    plt.plot(time, error_hmd)
    plt.title('Error Signal - HMD ANC')
    plt.xlabel('Time [sec]')
    plt.ylabel('Amplitude')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# Example usage
RATE = 48000  # Your sampling rate
signal_paths = {
    'ref': './sample_data/modified/output_ref_anc.wav',
    'hmd': './sample_data/modified/output_anti_noise.wav',
    'orig': './sample_data/modified/orig_signal.wav',
    'test' : './sample_data/modified/main_signal.wav'
}

analyze_signals(signal_paths['ref'], signal_paths['hmd'], signal_paths['orig'], signal_paths['test'], RATE)