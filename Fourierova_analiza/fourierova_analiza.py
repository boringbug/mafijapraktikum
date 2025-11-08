import numpy as np
import matplotlib.pyplot as plt

def dft_slow(x):
    """
    Compute the Discrete Fourier Transform of the 1D array x
    """
    N = len(x)
    X = np.zeros(N, dtype=complex)
    
    for k in range(N):  # Frequency index
        for n in range(N):  # Time index
            X[k] += x[n] * np.exp(2j * np.pi * k * n / N)
    
    return X

# Example usage
if __name__ == "__main__":
    # Create a test signal
    fs = 10000  # Sampling frequency
    t = np.linspace(0, 1, fs, endpoint=False)
    f1, f2 = 50, 120  # Frequencies
    signal = 0.7 * np.sin(2 * np.pi * f1 * t) + np.sin(2 * np.pi * f2 * t)
    
    # Compute DFT
    X = dft_slow(signal[:640])  # Use only first 64 points for speed
    
    # Plot results
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Time domain
    ax1.plot(t[:640], signal[:640])
    ax1.set_title('Time Domain Signal')
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('Amplitude')
    
    # Frequency domain (magnitude)
    freqs = np.arange(640) * fs / 640
    ax2.plot(freqs[:320], np.abs(X[:320]))  # Only plot first half (Nyquist)
    ax2.set_title('Frequency Domain')
    ax2.set_xlabel('Frequency [Hz]')
    ax2.set_ylabel('Magnitude')
    
    plt.tight_layout()
    plt.show()
