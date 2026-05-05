import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftfreq
from scipy.signal import spectrogram
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

def plot_spectrogram(signal: SignalData, samplerate: int, title: str, filename: str) -> None:
    """
    Plot a spectrogram (waterfall plot) of the signal.
    
    Args:
        signal: Input signal array
        samplerate: Sampling rate in Hz
        title: Title for the plot
        filename: Filename to save the figure
    """
    # Compute spectrogram
    # Use window size of 2048 samples (~46ms at 44.1kHz), overlap of 50%
    nperseg = min(2048, len(signal) // 4)
    noverlap = nperseg // 2
    
    f, t, Sxx = spectrogram(signal, fs=samplerate, nperseg=nperseg, 
                            noverlap=noverlap, scaling='density')
    
    # Convert to dB scale
    Sxx_db = 10 * np.log10(Sxx + 1e-10)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Plot spectrogram
    im = ax.pcolormesh(t, f, Sxx_db, shading='gouraud', cmap='viridis')
    
    # Set labels and title
    ax.set_xlabel('Čas (sekunde)', fontsize=12)
    ax.set_ylabel('Frekvenca (Hz)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Set frequency limits (0-5000 Hz for owl calls)
    ax.set_ylim([0, min(5000, f[-1])])
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Magnituda (dB)', fontsize=11)
    
    # Add grid for better readability
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"  Saved: {filename}")
    plt.close()  # Close to free memory

def plot_all_spectrograms(signals: Dict[str, SignalData], samplerate: int, 
                          results: Dict[str, IdentificationResult]) -> None:
    """
    Plot and save spectrograms for all signals individually.
    
    Args:
        signals: Dictionary of signal names to signal data
        samplerate: Sampling rate in Hz
        results: Identification results dictionary
    """
    print("\n" + "="*70)
    print("PLOTTING SPECTROGRAMS (WATERFALL PLOTS)")
    print("="*70)
    
    for name, signal in signals.items():
        # Create title with identification if it's a mixed signal
        if name in results:
            result: IdentificationResult = results[name]
            title = f"Spektrogram signala: {name} - IDENTIFICIRANO: {result.identified_as} (zaupanje: {result.confidence:.1%})"
        else:
            title = f"Spektrogram signala: {name} (Referenca)"
        
        filename = f"{name}_spectrogram.pdf"
        
        print(f"\nGenerating spectrogram for {name}...")
        plot_spectrogram(signal, samplerate, title, filename)
    
    # Also create a combined comparison spectrogram for mix2 and mix22 with references
    for mixed_name in ['mix2', 'mix22']:
        if mixed_name in signals:
            # Create figure with 3 subplots (no colorbar to avoid tight_layout issues)
            fig, axes = plt.subplots(3, 1, figsize=(14, 12))
            
            # Plot spectrogram for bubomono
            nperseg = min(2048, len(signals['bubomono']) // 4)
            noverlap = nperseg // 2
            f1, t1, Sxx1 = spectrogram(signals['bubomono'], fs=samplerate, 
                                        nperseg=nperseg, noverlap=noverlap, scaling='density')
            Sxx1_db = 10 * np.log10(Sxx1 + 1e-10)
            im1 = axes[0].pcolormesh(t1, f1, Sxx1_db, shading='gouraud', cmap='viridis')
            axes[0].set_ylabel('Frekvenca (Hz)', fontsize=10)
            axes[0].set_title('Referenca: bubomono', fontsize=11, fontweight='bold')
            axes[0].set_ylim([0, 5000])
            axes[0].grid(True, alpha=0.3, linestyle='--')
            
            # Plot spectrogram for bubo2mono
            nperseg2 = min(2048, len(signals['bubo2mono']) // 4)
            noverlap2 = nperseg2 // 2
            f2, t2, Sxx2 = spectrogram(signals['bubo2mono'], fs=samplerate,
                                        nperseg=nperseg2, noverlap=noverlap2, scaling='density')
            Sxx2_db = 10 * np.log10(Sxx2 + 1e-10)
            im2 = axes[1].pcolormesh(t2, f2, Sxx2_db, shading='gouraud', cmap='viridis')
            axes[1].set_ylabel('Frekvenca (Hz)', fontsize=10)
            axes[1].set_title('Referenca: bubo2mono', fontsize=11, fontweight='bold')
            axes[1].set_ylim([0, 5000])
            axes[1].grid(True, alpha=0.3, linestyle='--')
            
            # Plot spectrogram for mixed signal
            nperseg3 = min(2048, len(signals[mixed_name]) // 4)
            noverlap3 = nperseg3 // 2
            f3, t3, Sxx3 = spectrogram(signals[mixed_name], fs=samplerate,
                                        nperseg=nperseg3, noverlap=noverlap3, scaling='density')
            Sxx3_db = 10 * np.log10(Sxx3 + 1e-10)
            im3 = axes[2].pcolormesh(t3, f3, Sxx3_db, shading='gouraud', cmap='viridis')
            axes[2].set_xlabel('Čas (sekunde)', fontsize=10)
            axes[2].set_ylabel('Frekvenca (Hz)', fontsize=10)
            
            if mixed_name in results:
                result = results[mixed_name]
                axes[2].set_title(f'Testni signal: {mixed_name} - IDENTIFICIRANO: {result.identified_as}', 
                                 fontsize=11, fontweight='bold')
            else:
                axes[2].set_title(f'Testni signal: {mixed_name}', fontsize=11, fontweight='bold')
            
            axes[2].set_ylim([0, 5000])
            axes[2].grid(True, alpha=0.3, linestyle='--')
            
            # Add a single colorbar for all subplots (placed at the bottom to avoid layout issues)
            plt.subplots_adjust(bottom=0.1)
            cbar_ax = fig.add_axes([0.15, 0.05, 0.7, 0.02])  # [left, bottom, width, height]
            cbar = fig.colorbar(im3, cax=cbar_ax, orientation='horizontal')
            cbar.set_label('Magnituda (dB)', fontsize=10)
            
            plt.suptitle(f'Primerjava spektrogramov: {mixed_name} z referenčnima signaloma', 
                        fontsize=14, fontweight='bold', y=0.98)
            
            comparison_filename = f"{mixed_name}_spectrograms_comparison.pdf"
            plt.savefig(comparison_filename, dpi=300, bbox_inches='tight')
            print(f"\n  Saved: {comparison_filename}")
            plt.close()

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
    plt.savefig("normalized_autocorrelations.pdf", dpi=300, bbox_inches='tight')
    print("  Saved: normalized_autocorrelations.pdf")
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
    plt.savefig(f"{mixed_name}_comparison.pdf", dpi=300, bbox_inches='tight')
    print(f"  Saved: {mixed_name}_comparison.pdf")
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
    
    # Plot spectrograms (waterfall plots) for all signals
    plot_all_spectrograms(signals, samplerate, results)
    
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
