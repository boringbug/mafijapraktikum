import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftfreq
import gzip
from pathlib import Path
from typing import Dict, Tuple, List, Optional, Any, Union
from dataclasses import dataclass
from scipy.signal import find_peaks

# Type aliases
SignalData = np.ndarray
AutocorrelationResult = np.ndarray
FrequencyPeak = Dict[str, Union[float, int]]
PeriodInfo = Dict[str, Union[int, float]]

@dataclass
class SignalStats:
    """Signal statistics"""
    mean: float
    variance: float
    length: int
    duration: float

@dataclass
class DominantFrequency:
    """Dominant frequency information"""
    frequency: float
    magnitude: float
    period_seconds: float

@dataclass
class PeriodPeak:
    """Period peak information from autocorrelation"""
    lag_samples: int
    lag_seconds: float
    frequency_hz: float
    strength: float

@dataclass
class IdentificationResult:
    """Result of owl identification"""
    identified_as: str
    confidence: float
    similarity_to_bubomono: float
    similarity_to_bubo2mono: float
    periods: List[PeriodPeak]

def load_audio_from_txt_gz(filepath: Path) -> Optional[SignalData]:
    """
    Load audio data from a .txt.gz file with one number per line.
    
    Args:
        filepath: Path to the .txt.gz file
        
    Returns:
        Normalized signal data as numpy array, or None if loading fails
    """
    try:
        with gzip.open(filepath, 'rt') as f:
            # Read all lines and convert to float
            data: List[float] = []
            for line in f:
                line = line.strip()
                if line:  # Skip empty lines
                    try:
                        value = float(line)
                        data.append(value)
                    except ValueError:
                        # Skip any non-numeric lines
                        continue
        
        if not data:
            print(f"Warning: No data found in {filepath}")
            return None
            
        signal: SignalData = np.array(data, dtype=np.float64)
        
        # Center the data (remove DC offset)
        signal = signal - np.mean(signal)
        
        # Normalize to range [-1, 1] for better numerical stability
        max_val: float = np.max(np.abs(signal))
        if max_val > 0:
            signal = signal / max_val
        
        print(f"  Loaded {len(signal)} samples from {filepath.name}")
        print(f"  Data range: [{np.min(signal):.3f}, {np.max(signal):.3f}]")
        return signal
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def autocorrelation_fft(signal: SignalData) -> AutocorrelationResult:
    """
    Compute autocorrelation using FFT method:
    C(h,h)_n = (1/(N-n)) * Σ h_{k+n} h_k for n = 0,...,N-1
    
    Using FFT: C = F^{-1}[|Fh|^2]
    
    Args:
        signal: Input signal array
        
    Returns:
        Unbiased autocorrelation array
    """
    N: int = len(signal)
    
    # Zero-pad to length 2N for linear convolution (avoid circular wrap-around)
    signal_padded: SignalData = np.zeros(2*N, dtype=np.float64)
    signal_padded[:N] = signal
    
    # Compute FFT
    S = fft(signal_padded)
    
    # Compute power spectrum |Fh|^2
    power_spectrum = S * np.conj(S)
    
    # Inverse FFT to get autocorrelation
    autocorr_circular: SignalData = np.real(ifft(power_spectrum))
    
    # Take first N points (for lags 0 to N-1)
    autocorr_linear: SignalData = autocorr_circular[:N]
    
    # Apply unbiased estimate: divide by (N - n) for each lag n
    unbiased_autocorr: SignalData = np.zeros(N, dtype=np.float64)
    for n in range(N):
        unbiased_autocorr[n] = autocorr_linear[n] / (N - n)
    
    return unbiased_autocorr

def normalized_autocorrelation(signal: SignalData) -> Tuple[AutocorrelationResult, SignalStats]:
    """
    Compute normalized autocorrelation as defined:
    \tilde{C}(h, h)_n = (C(h, h)_n - ⟨h⟩²) / (C(h, h)_0 - ⟨h⟩²)
    where ⟨h⟩ = (1/N) Σ h_k
    
    Args:
        signal: Input signal array
        
    Returns:
        Tuple of (normalized autocorrelation array, signal statistics)
    """
    N: int = len(signal)
    
    # Compute mean ⟨h⟩
    mean_signal: float = np.mean(signal)
    
    # Compute autocorrelation C(h, h)_n
    autocorr: AutocorrelationResult = autocorrelation_fft(signal)
    
    # Compute variance σ² = C(h, h)_0 - ⟨h⟩²
    variance: float = autocorr[0] - mean_signal**2
    
    # Normalize autocorrelation
    if variance > 1e-10:  # Avoid division by zero
        normalized_autocorr: AutocorrelationResult = (autocorr - mean_signal**2) / variance
    else:
        normalized_autocorr = np.zeros_like(autocorr)
        normalized_autocorr[0] = 1.0
    
    stats: SignalStats = SignalStats(
        mean=mean_signal,
        variance=variance,
        length=N,
        duration=N / 44100.0  # Will be updated with actual samplerate
    )
    
    return normalized_autocorr, stats

def compute_fft_magnitude(signal: SignalData, samplerate: int) -> Tuple[SignalData, SignalData]:
    """
    Compute FFT magnitude spectrum of a signal.
    
    Args:
        signal: Input signal array
        samplerate: Sampling rate in Hz
        
    Returns:
        Tuple of (frequencies array, magnitude spectrum array)
    """
    N: int = len(signal)
    
    # Compute FFT
    fft_vals = fft(signal)
    
    # Get magnitude spectrum (only positive frequencies)
    magnitude: SignalData = np.abs(fft_vals[:N//2])
    
    # Frequency axis
    freqs: SignalData = fftfreq(N, 1/samplerate)[:N//2]
    
    return freqs, magnitude

def estimate_samplerate(signal_length: int, typical_rates: List[int] = None) -> int:
    """
    Estimate samplerate based on signal length and typical audio rates.
    
    Args:
        signal_length: Number of samples in the signal
        typical_rates: List of typical samplerates to check
        
    Returns:
        Estimated samplerate in Hz
    """
    if typical_rates is None:
        typical_rates = [44100, 22050, 48000, 16000, 8000]
    
    for rate in typical_rates:
        duration: float = signal_length / rate
        if 1 < duration < 30:  # Reasonable duration for an owl recording
            return rate
    # Default to 44100 Hz if no match found
    return 44100

def find_dominant_frequencies(signal: SignalData, samplerate: int, num_peaks: int = 5) -> List[DominantFrequency]:
    """
    Find dominant frequencies using FFT.
    
    Args:
        signal: Input signal array
        samplerate: Sampling rate in Hz
        num_peaks: Number of dominant frequencies to find
        
    Returns:
        List of dominant frequencies with their magnitudes
    """
    N: int = len(signal)
    
    # Compute FFT
    fft_vals = fft(signal)
    fft_magnitude: SignalData = np.abs(fft_vals[:N//2])
    
    # Frequency axis
    freqs: SignalData = np.fft.fftfreq(N, 1/samplerate)[:N//2]
    
    # Find peaks (excluding very low frequencies)
    min_freq: int = 50  # Owl calls are typically above 50 Hz
    max_freq: int = 5000  # Upper limit for owl calls
    start_idx: int = np.searchsorted(freqs, min_freq)
    end_idx: int = np.searchsorted(freqs, max_freq)
    
    peaks, properties = find_peaks(fft_magnitude[start_idx:end_idx], 
                                   height=np.max(fft_magnitude[start_idx:end_idx]) * 0.1,
                                   distance=10)
    peaks = peaks + start_idx
    
    # Sort by magnitude and take top peaks
    peak_mags: SignalData = fft_magnitude[peaks]
    sorted_indices: np.ndarray = np.argsort(peak_mags)[::-1]
    
    dominant: List[DominantFrequency] = []
    for i in sorted_indices[:num_peaks]:
        freq: float = freqs[peaks[i]]
        dominant.append(DominantFrequency(
            frequency=freq,
            magnitude=float(peak_mags[i]),
            period_seconds=1.0 / freq if freq > 0 else 0.0
        ))
    
    return dominant

def find_autocorrelation_peaks(autocorr: AutocorrelationResult, samplerate: int, 
                               num_peaks: int = 5, min_lag_samples: int = 50) -> List[PeriodPeak]:
    """
    Find prominent peaks in autocorrelation function.
    
    Args:
        autocorr: Normalized autocorrelation array
        samplerate: Sampling rate in Hz
        num_peaks: Number of peaks to find
        min_lag_samples: Minimum lag in samples to consider
        
    Returns:
        List of period peaks with their properties
    """
    # Start from minimum lag to avoid initial peak
    start_idx: int = min_lag_samples
    
    # Find peaks
    peaks: List[Tuple[int, float]] = []
    for i in range(start_idx, len(autocorr)-1):
        if autocorr[i] > 0.15 and autocorr[i] > autocorr[i-1] and autocorr[i] > autocorr[i+1]:
            # Check if peak is distinct
            is_new: bool = True
            for p in peaks:
                if abs(i - p[0]) < samplerate/50:  # Within 20ms
                    is_new = False
                    break
            if is_new:
                peaks.append((i, autocorr[i]))
    
    # Sort by autocorrelation strength
    peaks.sort(key=lambda x: x[1], reverse=True)
    
    # Convert to periods
    periods: List[PeriodPeak] = []
    for lag, strength in peaks[:num_peaks]:
        periods.append(PeriodPeak(
            lag_samples=lag,
            lag_seconds=lag / samplerate,
            frequency_hz=samplerate / lag if lag > 0 else 0.0,
            strength=strength
        ))
    
    return periods

def similarity_score(periods_test: List[PeriodPeak], periods_ref: List[PeriodPeak]) -> float:
    """
    Calculate similarity between two sets of periods.
    
    Args:
        periods_test: Periods from test signal
        periods_ref: Periods from reference signal
        
    Returns:
        Similarity score between 0 and 1
    """
    if not periods_test or not periods_ref:
        return 0.0
    
    score: float = 0.0
    n_compare: int = min(3, len(periods_test), len(periods_ref))
    
    total_weight: float = 0.0
    for i in range(n_compare):
        # Compare frequencies
        freq_test: float = periods_test[i].frequency_hz
        freq_ref: float = periods_ref[i].frequency_hz
        
        if freq_ref > 0:
            # Relative difference
            diff: float = abs(freq_test - freq_ref) / freq_ref
            # Convert to similarity (1 = perfect match, 0 = >100% difference)
            sim: float = max(0.0, 1.0 - diff)
            # Weight by strength
            weight: float = periods_test[i].strength * periods_ref[i].strength
            score += sim * weight
            total_weight += weight
    
    if total_weight > 0:
        score = score / total_weight
    else:
        score = score / n_compare if n_compare > 0 else 0.0
    
    return score

def analyze_signals(signals: Dict[str, SignalData], samplerate: int) -> Tuple[Dict[str, AutocorrelationResult], Dict[str, IdentificationResult]]:
    """
    Analyze all signals and identify the owl in mixed signals.
    
    Args:
        signals: Dictionary of signal names to signal data
        samplerate: Sampling rate in Hz
        
    Returns:
        Tuple of (autocorrelations dictionary, identification results dictionary)
    """
    print("\n" + "="*70)
    print("AUTOCORRELATION ANALYSIS")
    print("="*70)
    
    # Compute autocorrelations for all signals
    autocorrs: Dict[str, AutocorrelationResult] = {}
    stats: Dict[str, SignalStats] = {}
    
    for name, signal in signals.items():
        print(f"\nProcessing {name}...")
        print(f"  Signal length: {len(signal)} samples ({len(signal)/samplerate:.2f} seconds)")
        
        # Remove mean (already done in load function, but ensure)
        signal_zm: SignalData = signal - np.mean(signal)
        
        # Compute normalized autocorrelation
        norm_autocorr, signal_stats = normalized_autocorrelation(signal_zm)
        # Update duration with actual samplerate
        signal_stats.duration = len(signal) / samplerate
        autocorrs[name] = norm_autocorr
        stats[name] = signal_stats
        
        print(f"  Signal mean: {signal_stats.mean:.6f}")
        print(f"  Signal variance: {signal_stats.variance:.6f}")
        
        # Find dominant frequencies
        dom_freqs: List[DominantFrequency] = find_dominant_frequencies(
            signal_zm[:min(len(signal_zm), samplerate*5)], samplerate
        )
        if dom_freqs:
            print(f"  Dominant frequencies:")
            for f in dom_freqs[:3]:
                print(f"    {f.frequency:.1f} Hz (period: {f.period_seconds:.3f}s, magnitude: {f.magnitude:.1f})")
    
    # Extract periods from autocorrelation for references
    print("\n" + "="*70)
    print("PERIODICITY ANALYSIS (from autocorrelation peaks)")
    print("="*70)
    
    ref_names: List[str] = ['bubomono', 'bubo2mono']
    ref_periods: Dict[str, List[PeriodPeak]] = {}
    
    for name in ref_names:
        if name in autocorrs:
            periods: List[PeriodPeak] = find_autocorrelation_peaks(autocorrs[name], samplerate)
            ref_periods[name] = periods
            print(f"\n{name} - Dominant periods from autocorrelation:")
            if periods:
                for p in periods[:5]:
                    print(f"  {p.lag_seconds:.4f}s ({p.frequency_hz:.1f}Hz) - strength: {p.strength:.3f}")
            else:
                print("  No significant periods found")
    
    # Identify mixed signals
    print("\n" + "="*70)
    print("IDENTIFICATION OF OWLS IN NOISY SIGNALS")
    print("="*70)
    
    results: Dict[str, IdentificationResult] = {}
    mixed_names: List[str] = [name for name in signals.keys() if name not in ref_names]
    
    for name in mixed_names:
        print(f"\n{name.upper()}:")
        
        # Find periods in test signal
        test_periods: List[PeriodPeak] = find_autocorrelation_peaks(autocorrs[name], samplerate)
        
        if test_periods:
            print(f"  Detected periods in {name}:")
            for p in test_periods[:3]:
                print(f"    {p.lag_seconds:.4f}s ({p.frequency_hz:.1f}Hz) - strength: {p.strength:.3f}")
        
        # Calculate similarity to each reference
        sim_to_bubo1: float = similarity_score(test_periods, ref_periods.get('bubomono', []))
        sim_to_bubo2: float = similarity_score(test_periods, ref_periods.get('bubo2mono', []))
        
        print(f"\n  Similarity to bubomono:  {sim_to_bubo1:.3f}")
        print(f"  Similarity to bubo2mono: {sim_to_bubo2:.3f}")
        
        if sim_to_bubo1 > sim_to_bubo2:
            identified: str = "bubomono"
            confidence: float = sim_to_bubo1 / (sim_to_bubo1 + sim_to_bubo2 + 0.001)
        else:
            identified = "bubo2mono"
            confidence = sim_to_bubo2 / (sim_to_bubo1 + sim_to_bubo2 + 0.001)
        
        print(f"\n  *** IDENTIFIED AS: {identified} (confidence: {confidence:.1%}) ***")
        
        results[name] = IdentificationResult(
            identified_as=identified,
            confidence=confidence,
            similarity_to_bubomono=sim_to_bubo1,
            similarity_to_bubo2mono=sim_to_bubo2,
            periods=test_periods
        )
    
    return autocorrs, results

def plot_fft_spectra(signals: Dict[str, SignalData], samplerate: int, 
                     results: Dict[str, IdentificationResult]) -> None:
    """
    Plot Fourier transform magnitude spectra for all signals with Slovene labels.
    
    Args:
        signals: Dictionary of signal names to signal data
        samplerate: Sampling rate in Hz
        results: Identification results dictionary
    """
    n_signals: int = len(signals)
    fig, axes = plt.subplots(n_signals, 1, figsize=(14, 3*n_signals))
    
    if n_signals == 1:
        axes = [axes]
    
    colors: List[str] = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    # Frequency range for plotting (0-5000 Hz, typical for owl calls)
    max_freq_plot: int = 5000
    
    for idx, (name, signal) in enumerate(signals.items()):
        color: str = colors[idx % len(colors)]
        
        # Compute FFT magnitude spectrum
        freqs, magnitude = compute_fft_magnitude(signal, samplerate)
        
        # Limit to frequency range of interest
        freq_limit_idx: int = np.searchsorted(freqs, max_freq_plot)
        freqs_plot = freqs[:freq_limit_idx]
        magnitude_plot = magnitude[:freq_limit_idx]
        
        # Convert to dB scale for better visualization
        magnitude_db: SignalData = 20 * np.log10(magnitude_plot + 1e-10)
        
        axes[idx].plot(freqs_plot, magnitude_db, color=color, linewidth=1)
        axes[idx].fill_between(freqs_plot, magnitude_db, alpha=0.3, color=color)
        
        # Add title with identification if it's a mixed signal
        if name in results:
            result: IdentificationResult = results[name]
            title: str = f"{name} - IDENTIFICIRANO: {result.identified_as} (zaupanje: {result.confidence:.1%})"
        else:
            title = f"{name} (Referenca)"
        
        axes[idx].set_title(title, fontsize=11, fontweight='bold')
        axes[idx].set_xlabel('Frekvenca (Hz)', fontsize=10)
        axes[idx].set_ylabel('Magnituda (dB)', fontsize=10)
        axes[idx].grid(True, alpha=0.3)
        axes[idx].set_xlim([0, max_freq_plot])
        
        # Mark dominant frequencies
        dom_freqs: List[DominantFrequency] = find_dominant_frequencies(signal, samplerate, num_peaks=5)
        for f in dom_freqs[:3]:
            if f.frequency <= max_freq_plot:
                axes[idx].axvline(x=f.frequency, color='red', linestyle=':', alpha=0.7, linewidth=1)
                axes[idx].text(f.frequency, np.max(magnitude_db) - 5, 
                             f"{f.frequency:.0f} Hz", 
                             fontsize=8, alpha=0.8, ha='center')
        
        # Annotate with fundamental frequency
        if dom_freqs:
            axes[idx].text(0.98, 0.98, f'Fundamentalna frekvenca:\n{dom_freqs[0].frequency:.1f} Hz', 
                          transform=axes[idx].transAxes, fontsize=9,
                          verticalalignment='top', horizontalalignment='right',
                          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.suptitle('Fourierovi spektri signalov (magnitude)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

def plot_fft_comparison(signals: Dict[str, SignalData], samplerate: int, 
                        mixed_name: str, results: Dict[str, IdentificationResult]) -> None:
    """
    Detailed FFT comparison plot for a specific mixed signal with Slovene labels.
    
    Args:
        signals: Dictionary of signal names to signal data
        samplerate: Sampling rate in Hz
        mixed_name: Name of the mixed signal to analyze
        results: Identification results dictionary
    """
    if mixed_name not in signals:
        print(f"Signal {mixed_name} not found")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Frequency range for plotting
    max_freq_plot: int = 5000
    
    # Compute FFT for all three signals
    signals_to_plot = ['bubomono', 'bubo2mono', mixed_name]
    
    for idx, name in enumerate(signals_to_plot):
        if name not in signals:
            continue
            
        row: int = idx // 2
        col: int = idx % 2
        
        signal = signals[name]
        freqs, magnitude = compute_fft_magnitude(signal, samplerate)
        
        # Limit to frequency range of interest
        freq_limit_idx: int = np.searchsorted(freqs, max_freq_plot)
        freqs_plot = freqs[:freq_limit_idx]
        magnitude_plot = magnitude[:freq_limit_idx]
        
        # Convert to dB scale
        magnitude_db: SignalData = 20 * np.log10(magnitude_plot + 1e-10)
        
        color = 'b' if name == 'bubomono' else ('g' if name == 'bubo2mono' else 'r')
        axes[row, col].plot(freqs_plot, magnitude_db, color=color, linewidth=1.5)
        axes[row, col].fill_between(freqs_plot, magnitude_db, alpha=0.3, color=color)
        
        title = 'Referenca: bubomono' if name == 'bubomono' else ('Referenca: bubo2mono' if name == 'bubo2mono' else f'Testni signal: {name}')
        axes[row, col].set_title(title, fontsize=11, fontweight='bold')
        axes[row, col].set_xlabel('Frekvenca (Hz)', fontsize=10)
        axes[row, col].set_ylabel('Magnituda (dB)', fontsize=10)
        axes[row, col].grid(True, alpha=0.3)
        axes[row, col].set_xlim([0, max_freq_plot])
        
        # Mark dominant frequencies
        dom_freqs: List[DominantFrequency] = find_dominant_frequencies(signal, samplerate, num_peaks=3)
        for f in dom_freqs:
            if f.frequency <= max_freq_plot:
                axes[row, col].axvline(x=f.frequency, color='red', linestyle=':', alpha=0.7, linewidth=1)
                axes[row, col].text(f.frequency, np.max(magnitude_db) - 3, 
                                   f"{f.frequency:.0f} Hz", 
                                   fontsize=8, alpha=0.8, ha='center')
    
    # Fourth subplot - overlay comparison
    ax_overlay = axes[1, 1]
    
    for name, color in [('bubomono', 'b'), ('bubo2mono', 'g'), (mixed_name, 'r')]:
        if name not in signals:
            continue
        signal = signals[name]
        freqs, magnitude = compute_fft_magnitude(signal, samplerate)
        
        freq_limit_idx: int = np.searchsorted(freqs, max_freq_plot)
        freqs_plot = freqs[:freq_limit_idx]
        magnitude_plot = magnitude[:freq_limit_idx]
        magnitude_db = 20 * np.log10(magnitude_plot + 1e-10)
        
        label = 'bubomono' if name == 'bubomono' else ('bubo2mono' if name == 'bubo2mono' else mixed_name)
        ax_overlay.plot(freqs_plot, magnitude_db, color=color, linewidth=1.5, alpha=0.7, label=label)
    
    ax_overlay.set_title('Primerjava Fourierovih spektrov', fontsize=11, fontweight='bold')
    ax_overlay.set_xlabel('Frekvenca (Hz)', fontsize=10)
    ax_overlay.set_ylabel('Magnituda (dB)', fontsize=10)
    ax_overlay.legend(loc='best')
    ax_overlay.grid(True, alpha=0.3)
    ax_overlay.set_xlim([0, max_freq_plot])
    
    # Add identification text
    if mixed_name in results:
        result: IdentificationResult = results[mixed_name]
        text: str = f"IDENTIFICIRANO: {result.identified_as}\nZaupanje: {result.confidence:.1%}\n"
        text += f"Podobnost z bubomono: {result.similarity_to_bubomono:.3f}\n"
        text += f"Podobnost z bubo2mono: {result.similarity_to_bubo2mono:.3f}"
        ax_overlay.text(0.02, 0.98, text, transform=ax_overlay.transAxes,
                       fontsize=10, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))
    
    plt.suptitle(f'Fourierova analiza signala: {mixed_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

def plot_autocorrelations(autocorrs: Dict[str, AutocorrelationResult], 
                         samplerate: int, 
                         results: Dict[str, IdentificationResult]) -> None:
    """
    Plot autocorrelation functions for all signals with Slovene labels.
    
    Args:
        autocorrs: Dictionary of signal names to autocorrelation arrays
        samplerate: Sampling rate in Hz
        results: Identification results dictionary
    """
    n_signals: int = len(autocorrs)
    fig, axes = plt.subplots(n_signals, 1, figsize=(12, 3*n_signals))
    
    if n_signals == 1:
        axes = [axes]
    
    colors: List[str] = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    # Determine max time to show (first 0.5 seconds or until autocorrelation decays significantly)
    max_time: float = min(0.5, len(list(autocorrs.values())[0]) / samplerate)
    max_samples: int = int(max_time * samplerate)
    
    for idx, (name, autocorr) in enumerate(autocorrs.items()):
        color: str = colors[idx % len(colors)]
        
        time_lag: SignalData = np.arange(min(len(autocorr), max_samples)) / samplerate
        autocorr_plot: SignalData = autocorr[:len(time_lag)]
        
        axes[idx].plot(time_lag, autocorr_plot, color=color, linewidth=1.5)
        axes[idx].axhline(y=0, color='k', linestyle='-', alpha=0.3, linewidth=0.5)
        axes[idx].axhline(y=0.2, color='r', linestyle='--', alpha=0.3, linewidth=0.5)
        
        # Add title with identification if it's a mixed signal (Slovene)
        if name in results:
            result: IdentificationResult = results[name]
            title: str = f"{name} - IDENTIFICIRANO: {result.identified_as} (zaupanje: {result.confidence:.1%})"
        else:
            title = f"{name} (Referenca)"
        
        axes[idx].set_title(title, fontsize=11, fontweight='bold')
        axes[idx].set_xlabel('Časovni zamik (sekunde)', fontsize=10)
        axes[idx].set_ylabel(r'Normalizirana avtokorelacija $\tilde{C}(h,h)_n$', fontsize=10)
        axes[idx].grid(True, alpha=0.3)
        axes[idx].set_ylim([-0.5, 1.1])
        
        # Mark peaks
        if name in results and results[name].periods:
            for period in results[name].periods[:3]:
                if period.lag_seconds <= max_time:
                    axes[idx].axvline(x=period.lag_seconds, color='red', 
                                    linestyle=':', alpha=0.5, linewidth=1)
                    axes[idx].text(period.lag_seconds, 0.9, 
                                 f"{period.lag_seconds:.3f}s", 
                                 rotation=45, fontsize=8, alpha=0.7)
    
    plt.suptitle('Normalizirane avtokorelacijske funkcije signalov', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

def plot_comparison(signals: Dict[str, SignalData], 
                   autocorrs: Dict[str, AutocorrelationResult], 
                   samplerate: int, 
                   results: Dict[str, IdentificationResult], 
                   mixed_name: str) -> None:
    """
    Detailed comparison plot for a specific mixed signal with Slovene labels.
    
    Args:
        signals: Dictionary of signal names to signal data
        autocorrs: Dictionary of signal names to autocorrelation arrays
        samplerate: Sampling rate in Hz
        results: Identification results dictionary
        mixed_name: Name of the mixed signal to analyze
    """
    if mixed_name not in signals or mixed_name not in autocorrs:
        print(f"Signal {mixed_name} not found")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot first 2 seconds of each signal
    duration: float = min(3.0, len(signals[mixed_name]) / samplerate)
    n_samples: int = int(duration * samplerate)
    time: SignalData = np.arange(n_samples) / samplerate
    
    # Reference 1: bubomono
    if 'bubomono' in signals:
        axes[0,0].plot(time, signals['bubomono'][:n_samples], 'b-', alpha=0.7, linewidth=0.8)
        axes[0,0].set_title('Referenca: bubomono', fontsize=11, fontweight='bold')
        axes[0,0].set_ylabel('Amplituda', fontsize=10)
        axes[0,0].grid(True, alpha=0.3)
        axes[0,0].set_xlim([0, duration])
    
    # Reference 2: bubo2mono
    if 'bubo2mono' in signals:
        axes[0,1].plot(time, signals['bubo2mono'][:n_samples], 'g-', alpha=0.7, linewidth=0.8)
        axes[0,1].set_title('Referenca: bubo2mono', fontsize=11, fontweight='bold')
        axes[0,1].grid(True, alpha=0.3)
        axes[0,1].set_xlim([0, duration])
    
    # Mixed signal
    axes[1,0].plot(time, signals[mixed_name][:n_samples], 'r-', alpha=0.7, linewidth=0.8)
    axes[1,0].set_title(f'Testni signal: {mixed_name}', fontsize=11, fontweight='bold')
    axes[1,0].set_xlabel('Čas (sekunde)', fontsize=10)
    axes[1,0].set_ylabel('Amplituda', fontsize=10)
    axes[1,0].grid(True, alpha=0.3)
    axes[1,0].set_xlim([0, duration])
    
    # Autocorrelation comparison
    max_time: float = min(0.5, len(autocorrs[mixed_name]) / samplerate)
    max_samples: int = int(max_time * samplerate)
    time_lag: SignalData = np.arange(max_samples) / samplerate
    
    axes[1,1].plot(time_lag, autocorrs['bubomono'][:max_samples], 'b-', 
                  label='bubomono', alpha=0.7, linewidth=1.5)
    axes[1,1].plot(time_lag, autocorrs['bubo2mono'][:max_samples], 'g-', 
                  label='bubo2mono', alpha=0.7, linewidth=1.5)
    axes[1,1].plot(time_lag, autocorrs[mixed_name][:max_samples], 'r-', 
                  label=mixed_name, alpha=0.8, linewidth=2)
    
    axes[1,1].set_title('Primerjava avtokorelacijskih funkcij', fontsize=11, fontweight='bold')
    axes[1,1].set_xlabel('Časovni zamik (sekunde)', fontsize=10)
    axes[1,1].set_ylabel(r'Normalizirana avtokorelacija $\tilde{C}(h,h)_n$', fontsize=10)
    axes[1,1].legend(loc='best')
    axes[1,1].grid(True, alpha=0.3)
    axes[1,1].set_ylim([-0.5, 1.1])
    axes[1,1].set_xlim([0, max_time])
    
    # Add identification text (Slovene)
    if mixed_name in results:
        result: IdentificationResult = results[mixed_name]
        text: str = f"IDENTIFICIRANO: {result.identified_as}\nZaupanje: {result.confidence:.1%}\n"
        text += f"Podobnost z bubomono: {result.similarity_to_bubomono:.3f}\n"
        text += f"Podobnost z bubo2mono: {result.similarity_to_bubo2mono:.3f}"
        axes[1,1].text(0.02, 0.98, text, transform=axes[1,1].transAxes,
                      fontsize=10, verticalalignment='top',
                      bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))
    
    plt.suptitle(f'Podrobna analiza signala: {mixed_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

def main() -> None:
    """Main function to run the owl call identification program."""
    print("="*70)
    print("OWL CALL IDENTIFICATION USING AUTOCORRELATION AND FFT")
    print("="*70)
    
    # File paths (adjust if files are in a different directory)
    base_path: Path = Path(".")
    
    file_names: List[str] = [
        'bubomono.txt.gz',
        'bubo2mono.txt.gz',
        'mix.txt.gz',
        'mix1.txt.gz',
        'mix2.txt.gz',
        'mix22.txt.gz'
    ]
    
    # Load all signals
    print("\nLoading audio files...")
    signals: Dict[str, SignalData] = {}
    
    for fname in file_names:
        filepath: Path = base_path / fname
        if filepath.exists():
            print(f"\nReading {fname}...")
            data: Optional[SignalData] = load_audio_from_txt_gz(filepath)
            if data is not None and len(data) > 0:
                name: str = fname.replace('.txt.gz', '')
                signals[name] = data
        else:
            print(f"\nWarning: {filepath} not found")
    
    if len(signals) < 2:
        print("\nERROR: Could not find required files!")
        print("Please ensure the following files are in the current directory:")
        for fname in file_names:
            print(f"  - {fname}")
        return
    
    print(f"\nSuccessfully loaded {len(signals)} files")
    
    # Estimate samplerate based on signal length
    longest_signal: int = max([len(s) for s in signals.values()])
    samplerate: int = estimate_samplerate(longest_signal)
    print(f"\nEstimated samplerate: {samplerate} Hz")
    print(f"Longest signal duration: {longest_signal/samplerate:.2f} seconds")
    
    # Analyze signals using autocorrelation
    autocorrs, results = analyze_signals(signals, samplerate)
    
    # Plot Fourier transform spectra
    print("\n" + "="*70)
    print("PLOTTING FOURIER TRANSFORM SPECTRA")
    print("="*70)
    plot_fft_spectra(signals, samplerate, results)
    
    # Plot FFT comparison for noisy signals
    for mixed_name in ['mix2', 'mix22']:
        if mixed_name in signals:
            plot_fft_comparison(signals, samplerate, mixed_name, results)
    
    # Plot all autocorrelations
    print("\n" + "="*70)
    print("PLOTTING AUTOCORRELATION FUNCTIONS")
    print("="*70)
    plot_autocorrelations(autocorrs, samplerate, results)
    
    # Detailed plots for noisy signals
    for mixed_name in ['mix2', 'mix22']:
        if mixed_name in signals:
            plot_comparison(signals, autocorrs, samplerate, results, mixed_name)
    
    # Summary
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    print("\nBased on autocorrelation analysis using FFT:")
    
    for name in ['mix2', 'mix22']:
        if name in results:
            result: IdentificationResult = results[name]
            print(f"\n  • {name}: Contains {result.identified_as} call")
            print(f"    (confidence: {result.confidence:.1%})")
    
    print("\n" + "="*70)
    print("Analysis complete!")
    print("="*70)

if __name__ == "__main__":
    main()
