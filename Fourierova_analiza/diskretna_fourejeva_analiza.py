import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as la
import time
from scipy.optimize import curve_fit
from numpy.fft import fft, fftfreq

# ============================================================================
# Custom DFT and inverse DFT implementations
# ============================================================================

def dft(signal):
    '''Calculate the DFT of a 1D signal using O(N^2) algorithm'''
    N = len(signal)
    n = np.arange(N)
    k = n.reshape((N, 1))
    e = np.exp(2j * np.pi * k * n / N)
    return np.dot(e, signal)

def inverse_dft(signal):
    '''Calculate the inverse DFT of a 1D signal'''
    N = len(signal)
    n = np.arange(N)
    k = n.reshape((N, 1))
    e = np.exp(-2j * np.pi * k * n / N)
    return np.dot(e, signal)

# ============================================================================
# 1. PERIODIC SIGNAL - spectrum with exactly located peaks
# ============================================================================
print("Generating periodic_spectrum.pdf ...")

SAMPLE_RATE = 10000
SAMPLE_INTERVAL = 1.0 / SAMPLE_RATE
T_duration = 1.0
t_periodic = np.arange(0, T_duration, SAMPLE_INTERVAL)
N = len(t_periodic)

frequency_list = [5, 20]
amplitude_list = [1.0, 0.5]
signal_periodic = np.zeros_like(t_periodic)
for f, A in zip(frequency_list, amplitude_list):
    signal_periodic += A * np.sin(2 * np.pi * f * t_periodic)

dft_signal_periodic = dft(signal_periodic)
n = np.arange(N)
freq = n / T_duration
positive_mask = freq <= SAMPLE_RATE / 2
freq_pos = freq[positive_mask]
magnitude_pos = 2 * SAMPLE_INTERVAL * np.abs(dft_signal_periodic[positive_mask])

plt.figure(figsize=(12, 5))
plt.plot(freq_pos, magnitude_pos, 'b-', linewidth=1.5, label='lastna DFT')
plt.xlabel('Frekvenca (Hz)', fontsize=12)
plt.ylabel('Amplituda', fontsize=12)
plt.title('Periodični signal – frekvenci sta večkratnika osnovne frekvence (f₀ = 1 Hz)', fontsize=14)
plt.xlim(0, 30)
plt.grid(True, alpha=0.3)
plt.axvline(x=5, color='red', linestyle='--', alpha=0.7, label='5 Hz')
plt.axvline(x=20, color='green', linestyle='--', alpha=0.7, label='20 Hz')
plt.legend()
plt.tight_layout()
plt.savefig('periodic_spectrum.pdf', dpi=300, format='pdf')
plt.close()

# ============================================================================
# 2. NON-PERIODIC SIGNAL - spectral leakage
# ============================================================================
print("Generating nonperiodic_spectrum.pdf ...")

frequency_list_nonperiodic = [5.3, 20.7]
amplitude_list_nonperiodic = [1.0, 0.5]
signal_nonperiodic = np.zeros_like(t_periodic)
for f, A in zip(frequency_list_nonperiodic, amplitude_list_nonperiodic):
    signal_nonperiodic += A * np.sin(2 * np.pi * f * t_periodic)

dft_signal_nonperiodic = dft(signal_nonperiodic)
magnitude_nonperiodic = 2 * SAMPLE_INTERVAL * np.abs(dft_signal_nonperiodic[positive_mask])

plt.figure(figsize=(12, 5))
plt.plot(freq_pos, magnitude_nonperiodic, 'b-', linewidth=1.5, label='lastna DFT')
plt.xlabel('Frekvenca (Hz)', fontsize=12)
plt.ylabel('Amplituda', fontsize=12)
plt.title('Neperiodični signal – razlivanje spektra (leakage)', fontsize=14)
plt.xlim(0, 30)
plt.grid(True, alpha=0.3)
plt.axvline(x=5.3, color='red', linestyle='--', alpha=0.7, label='prava frekvenca: 5.3 Hz')
plt.axvline(x=20.7, color='green', linestyle='--', alpha=0.7, label='prava frekvenca: 20.7 Hz')
plt.legend()
plt.tight_layout()
plt.savefig('nonperiodic_spectrum.pdf', dpi=300, format='pdf')
plt.close()

# ============================================================================
# 3. ALIASING - synthetic signal (frequency above Nyquist)
# ============================================================================
print("Generating aliasing_synthetic.pdf ...")

SAMPLE_RATE_LOW = 100
SAMPLE_INTERVAL_LOW = 1.0 / SAMPLE_RATE_LOW
T_long = 4.0
t_low = np.arange(0, T_long, SAMPLE_INTERVAL_LOW)
N_low = len(t_low)

f1 = 10
f2 = 80
signal_aliasing = np.sin(2 * np.pi * f1 * t_low) + np.sin(2 * np.pi * f2 * t_low)

dft_signal_aliasing = dft(signal_aliasing)
n_low = np.arange(N_low)
freq_low = n_low / T_long
positive_mask_low = freq_low <= SAMPLE_RATE_LOW / 2
freq_pos_low = freq_low[positive_mask_low]
magnitude_aliasing = 2 * SAMPLE_INTERVAL_LOW * np.abs(dft_signal_aliasing[positive_mask_low])

nyquist_low = SAMPLE_RATE_LOW / 2
aliased_freq = abs(f2 - 2 * nyquist_low)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
mask_t = t_low < 1.0
ax1.plot(t_low[mask_t], signal_aliasing[mask_t], 'b-', linewidth=1.5)
ax1.set_xlabel('Čas (s)', fontsize=12)
ax1.set_ylabel('Amplituda', fontsize=12)
ax1.set_title('Signal s frekvenco nad Nyquistovo mejo (f₂ = 80 Hz, f_N = 50 Hz)', fontsize=14)
ax1.grid(True, alpha=0.3)

ax2.plot(freq_pos_low, magnitude_aliasing, 'b-', linewidth=1.5)
ax2.set_xlabel('Frekvenca (Hz)', fontsize=12)
ax2.set_ylabel('Amplituda', fontsize=12)
ax2.set_title('Frekvenčni spekter – opazen pojav aliasinga (80 Hz → ~20 Hz)', fontsize=14)
ax2.set_xlim(0, 60)
ax2.grid(True, alpha=0.3)
ax2.axvline(x=f1, color='green', linestyle='--', alpha=0.7, label=f'originalna: {f1} Hz')
ax2.axvline(x=f2, color='red', linestyle='--', alpha=0.7, label=f'originalna: {f2} Hz (nad Nyquistom)')
ax2.axvline(x=nyquist_low, color='purple', linestyle='-.', alpha=0.7, label=f'Nyquist: {nyquist_low} Hz')
ax2.axvline(x=aliased_freq, color='orange', linestyle='--', alpha=0.7, linewidth=2, label=f'aliasing: {aliased_freq:.0f} Hz')
ax2.legend()
plt.tight_layout()
plt.savefig('aliasing_synthetic.pdf', dpi=300, format='pdf')
plt.close()

# ============================================================================
# 4. NUMERICAL ACCURACY COMPARISON (DFT vs FFT)
# ============================================================================
print("Generating dft_accuracy_comparison.pdf ...")

N_values = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
relative_errors = []

for N_test in N_values:
    t_test = np.linspace(0, 1, N_test, endpoint=False)
    np.random.seed(42)
    freqs_test = np.random.uniform(5, 100, 5)
    amps_test = np.random.uniform(0.5, 2, 5)
    signal_test = np.zeros(N_test)
    for f, A in zip(freqs_test, amps_test):
        signal_test += A * np.sin(2 * np.pi * f * t_test)
    
    H_our = dft(signal_test)
    H_np = fft(signal_test)
    error = np.linalg.norm(H_our - H_np) / np.linalg.norm(H_np)
    relative_errors.append(error)

def error_trend(N, a, b):
    return a * N**b

N_fit = np.array(N_values[4:])
errors_fit = np.array(relative_errors[4:])
popt_error, _ = curve_fit(error_trend, N_fit, errors_fit)

N_smooth = np.logspace(np.log10(2), np.log10(4096), 200)
error_smooth = error_trend(N_smooth, *popt_error)

plt.figure(figsize=(12, 7))
plt.loglog(N_values, relative_errors, 'bo-', linewidth=2, markersize=8, label='Izmerjena relativna napaka')
plt.loglog(N_smooth, error_smooth, 'r--', linewidth=1.5, alpha=0.7,
          label=f'Fit: ε = {popt_error[0]:.2e}·N^{popt_error[1]:.4f}')
plt.xlabel('Dolžina signala N', fontsize=12)
plt.ylabel('Relativna napaka ε', fontsize=12)
plt.title('Numerična natančnost lastne DFT v primerjavi z numpy.fft', fontsize=14)
plt.grid(True, alpha=0.3, which='both')
plt.legend(fontsize=11)
plt.tight_layout()
plt.savefig('dft_accuracy_comparison.pdf', dpi=300, format='pdf')
plt.close()

# ============================================================================
# 5. INVERSE DFT RECONSTRUCTION ERROR
# ============================================================================
print("Generating inverse_dft_error.pdf ...")

recon_errors = []
for N_test in N_values:
    t_test = np.linspace(0, 1, N_test, endpoint=False)
    np.random.seed(42)
    freqs_test = np.random.uniform(5, 100, 3)
    amps_test = np.random.uniform(0.5, 2, 3)
    signal_test = np.zeros(N_test)
    for f, A in zip(freqs_test, amps_test):
        signal_test += A * np.sin(2 * np.pi * f * t_test)
    
    H = dft(signal_test)
    signal_recon = inverse_dft(H) / N_test
    recon_errors.append(np.max(np.abs(signal_test - signal_recon)))

plt.figure(figsize=(12, 7))
plt.loglog(N_values, recon_errors, 'go-', linewidth=2, markersize=8)
plt.xlabel('Dolžina signala N', fontsize=12)
plt.ylabel('Maksimalna napaka rekonstrukcije', fontsize=12)
plt.title('Natančnost inverzne DFT – rekonstrukcija originalnega signala', fontsize=14)
plt.grid(True, alpha=0.3, which='both')
plt.tight_layout()
plt.savefig('inverse_dft_error.pdf', dpi=300, format='pdf')
plt.close()

# ============================================================================
# 6. COMPUTATIONAL COMPLEXITY ANALYSIS
# ============================================================================
print("Generating dft_complexity_analysis.pdf ...")

N_complexity = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
times_dft = []
times_fft = []

for N_test in N_complexity:
    t_test = np.linspace(0, 1, N_test, endpoint=False)
    signal_test = np.sin(2 * np.pi * 10 * t_test) + 0.5 * np.sin(2 * np.pi * 25 * t_test)
    
    start = time.perf_counter()
    H_our = dft(signal_test)
    times_dft.append(time.perf_counter() - start)
    
    start = time.perf_counter()
    H_np = fft(signal_test)
    times_fft.append(time.perf_counter() - start)

def power_law(N, a, b):
    return a * N**b

def n_log_n(N, a, c):
    return a * N * np.log(N) + c

popt_dft, _ = curve_fit(power_law, N_complexity, times_dft)
popt_fft, _ = curve_fit(n_log_n, N_complexity, times_fft)

N_smooth_comp = np.logspace(np.log10(2), np.log10(4096), 200)
t_dft_fit = power_law(N_smooth_comp, *popt_dft)
t_fft_fit = n_log_n(N_smooth_comp, *popt_fft)

plt.figure(figsize=(12, 7))
plt.loglog(N_complexity, times_dft, 'ro-', linewidth=2, markersize=8, label='Lastna DFT (O(N²))')
plt.loglog(N_smooth_comp, t_dft_fit, 'r--', linewidth=1.5, alpha=0.7,
          label=f'Fit: t = {popt_dft[0]:.2e}·N^{popt_dft[1]:.2f}')
plt.loglog(N_complexity, times_fft, 'bo-', linewidth=2, markersize=8, label='numpy FFT (O(N log N))')
plt.loglog(N_smooth_comp, t_fft_fit, 'b--', linewidth=1.5, alpha=0.7,
          label=f'Fit: t = {popt_fft[0]:.2e}·N·log(N)')
plt.xlabel('Dolžina signala N', fontsize=12)
plt.ylabel('Čas izvajanja (s)', fontsize=12)
plt.title('Računska kompleksnost: lastna DFT (O(N²)) vs numpy FFT (O(N log N))', fontsize=14)
plt.grid(True, alpha=0.3, which='both')
plt.legend(fontsize=11)
plt.tight_layout()
plt.savefig('dft_complexity_analysis.pdf', dpi=300, format='pdf')
plt.close()

# ============================================================================
# 7. PARSEVAL'S THEOREM AND RECONSTRUCTION VERIFICATION
# ============================================================================
print("\n" + "="*60)
print("PARSEVALOV IZREK IN NATANČNOST REKONSTRUKCIJE")
print("="*60)

norm_signal_sq = la.norm(signal_periodic)**2
norm_dft_sq = la.norm(np.abs(dft_signal_periodic))**2
parseval_diff = norm_signal_sq - norm_dft_sq / N

print(f"||h||² = {norm_signal_sq:.6f}")
print(f"(1/N)||H||² = {norm_dft_sq/N:.6f}")
print(f"Razlika (Parsevalov izrek): {parseval_diff:.2e}")

inverse_reconstructed = inverse_dft(dft_signal_periodic) / N
max_recon_error = np.max(np.abs(signal_periodic - inverse_reconstructed))
print(f"Maksimalna napaka rekonstrukcije (inverzna DFT): {max_recon_error:.2e}")

print("\nGENERIRANE SLIKE (PDF):")
print(" 1. periodic_spectrum.pdf")
print(" 2. nonperiodic_spectrum.pdf")
print(" 3. aliasing_synthetic.pdf")
print(" 4. dft_accuracy_comparison.pdf")
print(" 5. inverse_dft_error.pdf")
print(" 6. dft_complexity_analysis.pdf")
print(" 7. bach_all_rates_comparison.pdf")
print(" 8. resonator_signals.pdf")
