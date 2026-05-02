import time
import csv
import matplotlib.pyplot as plt
import numpy as np

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

sampling_rates = np.arange(100, 1e4, 100)[::-1]
dft_times = []

for rate in sampling_rates:
    x = np.arange(0, 1, 1/rate)
    signal = np.exp(-x**2) # Gaussian signal

    start_time = time.time_ns()
    dft_signal = dft(signal)
    end_time = time.time_ns()

    dft_times.append(end_time - start_time)
    print(rate, " done")

# Write to CSV file
csv_filename = 'time_complexity.csv'

with open(csv_filename, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Sampling rate', 'Time(ns)'])
    for s, t in zip(sampling_rates, dft_times):
        writer.writerow([s, t])
