import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sampling_rates = [882, 1378, 2756, 5512, 11025, 44100]

file_dir = "/home/flopa/Physics/mafijapraktikum/Fourierova_analiza/"

# Create figure with two parts: individual subplots and comparison
fig = plt.figure(figsize=(16, 12))

# 1. Individual subplots (2 rows, 3 columns) with Nyquist lines
for i, rate in enumerate(sampling_rates):
    fourier_data = pd.read_csv(file_dir + "fourier_data_" + str(rate) + "Hz.csv")

    magnitude = fourier_data['Magnitude']
    frequency = fourier_data['Frequency (Hz)']

    ax = plt.subplot(2, 3, i+1)
    ax.plot(frequency, magnitude, 'b-', linewidth=0.8)
    
    # Calculate and add Nyquist frequency line
    nyquist_freq = rate / 2
    ax.axvline(x=nyquist_freq, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label=f'Nyquist: {nyquist_freq} Hz')
    
    ax.set_title(f"{rate} Hz", fontsize=10)
    ax.set_xlabel("Frekvenca (Hz)", fontsize=8)
    ax.set_ylabel("Amplituda", fontsize=8)
    ax.set_xlim(0, min(7000, nyquist_freq * 1.1))  # Adjust xlim to show Nyquist
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=6)

plt.suptitle("Fourierova analiza pri različnih vzorčenjih", fontsize=14)
plt.tight_layout()
plt.savefig("seperate_dft_polots.pdf", dpi=1200)
plt.show()

# 2. Comparison plot with vertical shifts (all on same axes, stacked vertically)
plt.figure(figsize=(14, 10))

# Define vertical shifts (increasing offsets for higher sampling rates)
vertical_shifts = [0, 0.5, 1.0, 1.5, 2.0, 2.5]
colors = ['blue', 'green', 'orange', 'red', 'purple', 'brown']

for i, rate in enumerate(sampling_rates):
    fourier_data = pd.read_csv(file_dir + "fourier_data_" + str(rate) + "Hz.csv")
    magnitude = fourier_data['Magnitude']
    frequency = fourier_data['Frequency (Hz)']
    
    # Shift the magnitude vertically for better visibility
    shifted_magnitude = magnitude + vertical_shifts[i]
    
    # Plot the shifted spectrum
    plt.plot(frequency, shifted_magnitude, color=colors[i], label=f'{rate} Hz', linewidth=1.2, alpha=0.8)
    
    # Add Nyquist frequency line for this sampling rate
    nyquist_freq = rate / 2
    plt.axvline(x=nyquist_freq, color=colors[i], linestyle='--', linewidth=1, alpha=0.5, 
                ymin=vertical_shifts[i]/max(vertical_shifts+[shifted_magnitude.max()]), 
                ymax=(vertical_shifts[i]+magnitude.max())/max(vertical_shifts+[shifted_magnitude.max()]))
    
    # Add text label for Nyquist frequency
    plt.text(nyquist_freq, vertical_shifts[i] + magnitude.max() * 0.8, 
             f'{nyquist_freq:.0f} Hz', 
             fontsize=8, rotation=90, alpha=0.7, color=colors[i])

plt.xlabel("Frequency (Hz)", fontsize=12)
plt.ylabel("Amplitude (vertically shifted)", fontsize=12)
plt.title("Comparison of Fourier Spectra at Different Sampling Rates (Stacked)", fontsize=14)
plt.xlim(0, 7000)
plt.ylim(-0.5, max(vertical_shifts) + 1.5)  # Adjust ylim to fit all shifted spectra
plt.legend(loc='upper right', fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 3. Alternative comparison plot with transparency and normalization
plt.figure(figsize=(14, 8))

for i, rate in enumerate(sampling_rates):
    fourier_data = pd.read_csv(file_dir + "fourier_data_" + str(rate) + "Hz.csv")
    magnitude = fourier_data['Magnitude']
    frequency = fourier_data['Frequency (Hz)']
    
    # Normalize magnitude for better comparison
    normalized_magnitude = magnitude / magnitude.max()
    
    # Add small vertical shift for separation
    shift = i * 0.15
    shifted_normalized = normalized_magnitude + shift
    
    plt.plot(frequency, shifted_normalized, color=colors[i], label=f'{rate} Hz', linewidth=1.2, alpha=0.8)
    
    # Add Nyquist frequency line
    nyquist_freq = rate / 2
    plt.axvline(x=nyquist_freq, color=colors[i], linestyle='--', linewidth=1, alpha=0.4)

plt.xlabel("Frekvenca (Hz)", fontsize=12)
plt.ylabel("Normirana amplituda", fontsize=12)
plt.title("Normiran Fourierov spekter", fontsize=14)
plt.xlim(0, 7000)
plt.ylim(-0.1, len(sampling_rates) * 0.15 + 0.2)
plt.legend(loc='upper right', fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("dft_together_plot.pdf", dpi=1200)
plt.show()

# 4. Log scale plot with vertical shifts
plt.figure(figsize=(14, 10))

for i, rate in enumerate(sampling_rates):
    fourier_data = pd.read_csv(file_dir + "fourier_data_" + str(rate) + "Hz.csv")
    magnitude = fourier_data['Magnitude']
    frequency = fourier_data['Frequency (Hz)']
    
    # Add small constant to avoid log(0)
    magnitude_shifted = magnitude + 1e-10
    
    # Shift for log scale (additive shift in log space is multiplicative in linear)
    shift_factor = 10 ** (i * 0.5)  # Exponential shifts for log scale
    shifted_magnitude = magnitude_shifted * shift_factor
    
    plt.semilogy(frequency, shifted_magnitude, color=colors[i], label=f'{rate} Hz', linewidth=1.2, alpha=0.8)
    
    # Add Nyquist frequency line
    nyquist_freq = rate / 2
    plt.axvline(x=nyquist_freq, color=colors[i], linestyle='--', linewidth=1, alpha=0.4)

plt.xlabel("Frequency (Hz)", fontsize=12)
plt.ylabel("Amplitude (log scale, shifted)", fontsize=12)
plt.title("Log Scale Fourier Spectra with Vertical Separation", fontsize=14)
plt.xlim(0, 7000)
plt.legend(loc='upper right', fontsize=10)
plt.grid(True, alpha=0.3, which='both')
plt.tight_layout()
plt.show()

# 5. Waterfall plot (3D-like view)
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')

for i, rate in enumerate(sampling_rates):
    fourier_data = pd.read_csv(file_dir + "fourier_data_" + str(rate) + "Hz.csv")
    magnitude = fourier_data['Magnitude']
    frequency = fourier_data['Frequency (Hz)']
    
    # Only plot up to 7000 Hz for better visualization
    mask = frequency <= 7000
    freq_trimmed = frequency[mask]
    mag_trimmed = magnitude[mask]
    
    # Create y-position (constant for each sampling rate)
    y_pos = np.ones_like(freq_trimmed) * i
    
    ax.plot(freq_trimmed, y_pos, mag_trimmed, color=colors[i], linewidth=1, alpha=0.8, label=f'{rate} Hz')
    
    # Add Nyquist frequency as vertical line in 3D
    nyquist_freq = rate / 2
    if nyquist_freq <= 7000:
        ax.plot([nyquist_freq, nyquist_freq], [i, i], [0, mag_trimmed.max()], 
                color=colors[i], linestyle='--', linewidth=1, alpha=0.5)

ax.set_xlabel('Frekvenca (Hz)', fontsize=10)
ax.set_ylabel('Vzorčenje', fontsize=10)
ax.set_zlabel('Amplituda', fontsize=10)
ax.set_title('Fourier Spectra', fontsize=14)
ax.set_yticks(range(len(sampling_rates)))
ax.set_yticklabels([f'{rate}' for rate in sampling_rates])
ax.view_init(elev=20, azim=-60)
plt.tight_layout()
plt.savefig("3D_dft_plots.pdf", dpi=1200)
plt.show()

# 6. Print summary statistics with Nyquist info
print("\n" + "="*70)
print("SUMMARY STATISTICS WITH NYQUIST FREQUENCIES")
print("="*70)

for rate in sampling_rates:
    fourier_data = pd.read_csv(file_dir + "fourier_data_" + str(rate) + "Hz.csv")
    magnitude = fourier_data['Magnitude']
    frequency = fourier_data['Frequency (Hz)']
    
    nyquist_freq = rate / 2
    
    # Find peak frequency (excluding DC component)
    peak_idx = magnitude[1:].idxmax() if len(magnitude) > 1 else 0
    peak_freq = frequency.iloc[peak_idx]
    peak_amp = magnitude.iloc[peak_idx]
    
    # Find energy above Nyquist (should be zero for properly sampled signals)
    above_nyquist = frequency > nyquist_freq
    energy_above = magnitude[above_nyquist].sum()
    total_energy = magnitude.sum()
    energy_ratio = (energy_above / total_energy * 100) if total_energy > 0 else 0
    
    print(f"\n{rate:5d} Hz:")
    print(f"  - Nyquist frequency: {nyquist_freq:.1f} Hz")
    print(f"  - Frequency bins: {len(frequency)}")
    print(f"  - Max frequency in data: {frequency.iloc[-1]:.1f} Hz")
    print(f"  - Peak amplitude: {peak_amp:.4f} at {peak_freq:.1f} Hz")
    print(f"  - Energy above Nyquist: {energy_ratio:.3f}%")
    
    if energy_ratio < 0.01:
        print(f"  ✓ Signal is properly sampled (negligible aliasing)")
    elif energy_ratio < 1:
        print(f"  ⚠ Small amount of aliasing detected")
    else:
        print(f"  ✗ Significant aliasing - increase sampling rate")

# 7. Create a separate plot focusing on Nyquist frequencies
fig, ax = plt.subplots(figsize=(12, 6))

for i, rate in enumerate(sampling_rates):
    fourier_data = pd.read_csv(file_dir + "fourier_data_" + str(rate) + "Hz.csv")
    magnitude = fourier_data['Magnitude']
    frequency = fourier_data['Frequency (Hz)']
    
    nyquist_freq = rate / 2
    
    # Only plot up to Nyquist frequency
    mask = frequency <= nyquist_freq
    plt.plot(frequency[mask], magnitude[mask], color=colors[i], label=f'{rate} Hz (Nyquist: {nyquist_freq:.0f} Hz)', linewidth=1.2, alpha=0.7)
    
    # Mark Nyquist point
    if nyquist_freq in frequency.values:
        nyquist_idx = np.abs(frequency - nyquist_freq).argmin()
        plt.plot(nyquist_freq, magnitude.iloc[nyquist_idx], 'o', color=colors[i], markersize=4)

plt.xlabel("Frequency (Hz)", fontsize=12)
plt.ylabel("Amplitude", fontsize=12)
plt.title("Fourier Spectra (Only up to Nyquist Frequency)", fontsize=14)
plt.xlim(0, max([r/2 for r in sampling_rates]))
plt.legend(loc='upper right', fontsize=9)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
