import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as la

# Definition of dft and constants
def dft(signal):
    '''
    Function to calculate the DFT of a 1D signal
    '''
    N = len(signal)
    n = np.arange(N)
    k = n.reshape((N, 1))
    e = np.exp(2j * np.pi * k * n / N)

    output = np.dot(e, signal)

    return output

def inverse_dft(signal):
    '''
    Function to calculate the inverse of DFT of a 1D signal
    '''
    N = len(signal)
    n = np.arange(N)
    k = n.reshape((N, 1))
    e = np.exp(-2j * np.pi * k * n / N)

    output = np.dot(e, signal)

    return output

SAMPLE_RATE = 2e4
SAMPLE_INTERVAL = 1.0/SAMPLE_RATE

time = np.arange(0, 1, SAMPLE_INTERVAL)
N = len(time)

# Generating a sin sample

frequency_list = [10]
amplitude_list = [1]

signal = np.zeros_like(time)
for f, A in zip(frequency_list, amplitude_list):
    signal += A*np.sin(2*np.pi*f*time)

# Ploting the original signal and dft singal

plt.plot(time, signal)
plt.xlabel('Time')
plt.ylabel('Amplitude')

plt.show()


dft_signal = dft(signal)

n = np.arange(N)
TIME_INTERVAL = N/SAMPLE_RATE
freq = n/TIME_INTERVAL
NYQUIST_FREQUENCY = 1/(2*SAMPLE_INTERVAL)

plt.plot(freq, 2*SAMPLE_INTERVAL*abs(dft_signal))
plt.xlabel('Frequency')
plt.ylabel('Amplitude')

plt.show()

# Parsvel equality 
norm_signal = la.norm(signal)
norm_dft_signal = la.norm(abs(dft_signal))
diff = norm_signal - norm_dft_signal/np.sqrt(N)


inverse_dft_signal = inverse_dft(dft_signal)

plt.plot(time, abs(signal) - abs(inverse_dft_signal)/N)
plt.xlabel('Time')
plt.ylabel('Diff')

plt.show()

print("The differece between the signal and dft/N: ", diff)
