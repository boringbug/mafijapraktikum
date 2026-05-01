import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as la
import gzip
import csv

# Definition of dft and constants
def dft(signal):
    '''
    Function to calculate the DFT of a 1D signal
    '''
    N = len(signal)
    n_bins = N//2
    output = np.zeros(n_bins, dtype=complex)

    for k in range(n_bins):
        sum = 0j

        for n in range(N):
            sum += signal[n] * np.exp(2j*np.pi*k*n/N)
        output[k] = sum

    return output

sampling_rates = [882, 1378, 2756, 5512, 11025, 44100]
#sampling_rates = [882, 1378, 2756, 5512]
FILE_NAME = '/home/flopa/Physics/mafijapraktikum/Fourierova_analiza/Bach.'
SUFFIX = '.txt.gz'

for rate in sampling_rates:
    with gzip.open(FILE_NAME+str(rate)+SUFFIX, 'r') as f:
        signal = [float(line.strip()) for line in f]

    dft_signal = dft(signal)

    signal_length = len(signal)
    n = np.arange(len(dft_signal))
    time_interval = signal_length/rate
    freq = n/time_interval

    magnitude = np.abs(dft_signal)
    magnitude[1:-1] *= 2
    magnitude = magnitude / rate

    #plt.plot(freq, magnitude, label=str(rate))
    #plt.legend()
    #plt.show()

    # Write to CSV file
    csv_filename = f'fourier_data_{rate}Hz.csv'
    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Write header
        writer.writerow(['Frequency (Hz)', 'Magnitude'])
        # Write data
        for f, m in zip(freq, magnitude):
            writer.writerow([f, m])
    
    print(f"Saved data to {csv_filename}")
