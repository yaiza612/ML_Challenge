import os
import numpy as np
import pandas as pd
from scipy.signal import stft, welch, hilbert
from scipy.stats import entropy
from scipy.integrate import simpson
from src.data_loading.loader import load_data_as_windows
from src.feature_extraction.frequency_domain_features import bandpass_filter
from tqdm import tqdm
import pywt
from collections import defaultdict


def compute_wavelet_coherence(signal1, signal2, fs, wavelet_name='morl', band=None):
    """
    Compute wavelet coherence and return a single floating-point value.

    Parameters:
    - signal1: 1D array-like, first input signal
    - signal2: 1D array-like, second input signal (same length as signal1)
    - fs: int, sampling frequency
    - wavelet_name: str, type of wavelet to use (default='morl')
    - band: tuple, (min_freq, max_freq) to average coherence over a specific frequency band (optional)

    Returns:
    - avg_coherence: float, the average wavelet coherence as a single value
    """
    # Step 1: Compute the Continuous Wavelet Transform (CWT) of both signals
    scales = np.arange(1, 128)
    coeffs1, freqs1 = pywt.cwt(signal1, scales, wavelet_name, 1 / fs)
    coeffs2, freqs2 = pywt.cwt(signal2, scales, wavelet_name, 1 / fs)

    # Step 2: Calculate the Cross Wavelet Transform (XWT)
    cross_wavelet = coeffs1 * np.conj(coeffs2)

    # Step 3: Calculate the Wavelet Power of each signal
    power1 = np.abs(coeffs1) ** 2
    power2 = np.abs(coeffs2) ** 2

    # Step 4: Calculate Wavelet Coherence
    wavelet_coherence = np.abs(cross_wavelet) ** 2 / (power1 * power2)

    # Step 5: Average Wavelet Coherence over the entire time-frequency domain or within the given band
    if band:
        # Filter frequencies within the specified band
        band_idx = np.where((freqs1 >= band[0]) & (freqs1 <= band[1]))
        coherence_band = wavelet_coherence[band_idx]
        avg_coherence = np.mean(coherence_band)
    else:
        # Average over all time and frequencies
        avg_coherence = np.mean(wavelet_coherence)

    return avg_coherence


def phase_amplitude_coupling(low_freq_signal, high_freq_signal, fs, low_freq_band, high_freq_band, n_bins=18):
    """
    Calculate Phase-Amplitude Coupling using the Modulation Index (MI) method.

    Parameters:
    - signal: 1D array-like, the EEG signal
    - fs: int, the sampling frequency in Hz
    - low_freq_band: tuple, the frequency band for phase (e.g., (4, 8) for theta)
    - high_freq_band: tuple, the frequency band for amplitude (e.g., (30, 80) for gamma)
    - n_bins: int, the number of phase bins for computing the phase-amplitude distribution

    Returns:
    - modulation_index: float, the computed Modulation Index (MI) for PAC
    """
    # Step 3: Extract phase of low-frequency signal using the Hilbert transform
    low_freq_phase = np.angle(hilbert(low_freq_signal))

    # Step 4: Extract amplitude envelope of high-frequency signal using the Hilbert transform
    high_freq_amplitude = np.abs(hilbert(high_freq_signal))

    # Step 5: Calculate the phase-amplitude distribution
    phase_bins = np.linspace(-np.pi, np.pi, n_bins + 1)  # Create phase bins
    amplitude_means = np.zeros(n_bins)  # Initialize array for mean amplitude in each bin

    # Step 6: Bin the phase data and calculate the mean amplitude for each phase bin
    for i in range(n_bins):
        bin_mask = (low_freq_phase >= phase_bins[i]) & (low_freq_phase < phase_bins[i + 1])
        amplitude_means[i] = np.mean(high_freq_amplitude[bin_mask])

    # Normalize the amplitude distribution
    amplitude_means /= np.sum(amplitude_means)

    # Step 7: Compute entropy of the observed phase-amplitude distribution
    observed_entropy = -np.sum(amplitude_means * np.log(amplitude_means + 1e-8))  # Small constant to avoid log(0)

    # Step 8: Compute the Modulation Index (MI)
    uniform_entropy = np.log(n_bins)  # Entropy of a uniform distribution
    modulation_index = (uniform_entropy - observed_entropy) / uniform_entropy

    return modulation_index


def compute_wavelet_entropy(signal, wavelet="db4", level=4):
    """
    Calculate the wavelet entropy of the given signal.

    Parameters:
    - signal: 1D array-like, the EEG signal
    - wavelet: str, name of the wavelet (e.g., 'db4' for Daubechies)
    - level: int, the decomposition level

    Returns:
    - entropy: float, the wavelet entropy
    """
    # Step 1: Perform a multi-level wavelet decomposition
    coeffs = pywt.wavedec(signal, wavelet, level=level)

    # Step 2: Calculate the energy at each decomposition level
    energies = [np.sum(np.square(c)) for c in coeffs]

    # Step 3: Calculate the total energy
    total_energy = np.sum(energies)

    # Step 4: Calculate the normalized energy (probability distribution)
    normalized_energies = energies / total_energy

    # Step 5: Calculate the wavelet entropy
    entropy = -np.sum(normalized_energies * np.log(normalized_energies))

    return entropy


def compute_time_frequency_domain_features(segment_data, filtered_segment_data, fs=250, labels=None):
    """
    Extracts time-frequency domain features from EEG data for each segment using STFT.

    Parameters:
    - segment_data: np.ndarray - EEG data for one segment, shape (channels, samples).
    - fs: int - Sampling frequency (in Hz), default is 250.
    - labels: np.ndarray or list - Labels for each channel, shape (channels,).

    Returns:
    - pd.DataFrame containing time-frequency domain features for each segment.
    """
    # Frequency bands (in Hz)
    freq_bands = {'delta': (0.5, 4), 'theta': (4, 8), 'alpha': (8, 13), 'beta': (13, 30), 'gamma': (30, 45)}
    freq_bands_sorted = ['delta', 'theta', 'alpha', 'beta', 'gamma']

    # Feature storage
    feature_dict = defaultdict(list)

    # Process each channel in the segment
    for ch in range(segment_data.shape[0]):
        channel_data = segment_data[ch, :]

        # Short-Time Fourier Transform (STFT)
        f, t, Zxx = stft(channel_data, fs=fs, nperseg=fs, noverlap=fs // 2)
        power_spectrum = np.abs(Zxx) ** 2

        # Total power in STFT
        total_power = np.sum(power_spectrum)

        # Mean Frequency (across all time windows)
        mean_frequency = np.sum(f * np.mean(power_spectrum, axis=1)) / np.sum(np.mean(power_spectrum, axis=1))
        feature_dict['mean_frequency'].append(mean_frequency)

        # Peak Frequency (frequency with maximum average power over time)
        avg_power_spectrum = np.mean(power_spectrum, axis=1)
        peak_frequency = f[np.argmax(avg_power_spectrum)]
        feature_dict['peak_frequency'].append(peak_frequency)

        # Spectral Entropy over time
        spectral_entropy = entropy(avg_power_spectrum / np.sum(avg_power_spectrum))
        feature_dict['spectral_entropy'].append(spectral_entropy)

        # Wavelet Entropy
        wavelet_entropy = compute_wavelet_entropy(channel_data)
        feature_dict['wavelet_entropy'].append(wavelet_entropy)

        # Calculate average power and ratios for each frequency band
        band_powers = {}
        for band, (fmin, fmax) in freq_bands.items():
            band_indices = (f >= fmin) & (f <= fmax)
            band_power = simpson(np.mean(power_spectrum[band_indices, :], axis=1), dx=f[1] - f[0])
            band_powers[band] = band_power
            feature_dict[f'avg_power_{band}'].append(band_power)
        for low_freq_band_idx, low_freq_band_name in enumerate(freq_bands_sorted[:-1]):
            low_freq_band = freq_bands[low_freq_band_name]
            for high_freq_band_name in freq_bands_sorted[low_freq_band_idx + 1:]:
                high_freq_band = freq_bands[high_freq_band_name]
                phase_amplitude_coupled = phase_amplitude_coupling(
                    low_freq_signal=filtered_segment_data[low_freq_band_name][ch],
                    high_freq_signal=filtered_segment_data[high_freq_band_name][ch], fs=fs,
                    low_freq_band=low_freq_band,
                    high_freq_band=high_freq_band)
                feature_dict[f'phase_amplitude_coupled_{low_freq_band}_{high_freq_band}'].append(
                    phase_amplitude_coupled)
        # Band Ratios relative to total power
        for band in freq_bands:
            feature_dict[f'band_ratio_stft{band}'].append(band_powers[band] / total_power)
    #for channel_1_idx in range(5):
    #    channel_1 = segment_data[channel_1_idx, :]
    #    for channel_2_idx in range(channel_1_idx):
    #        channel_2 = segment_data[channel_2_idx, :]
    #        wavelet_coherence = compute_wavelet_coherence(signal1=channel_1, signal2=channel_2, fs=fs)
    #        feature_dict[f'wavelet_coherence_{channel_1_idx}_{channel_2_idx}'].extend([wavelet_coherence] * 5)
    #        for freq_band_name in freq_bands_sorted:
    #            freq_band = freq_bands[freq_band_name]
    #            wavelet_coherence_band = compute_wavelet_coherence(signal1=channel_1, signal2=channel_2,
    #                                                               fs=fs, band=freq_band)
    #            feature_dict[f'wavelet_coherence_{channel_1_idx}_{channel_2_idx}_{freq_band_name}']\
    #                .extend([wavelet_coherence_band] * 5)

    # Convert dictionary to DataFrame
    features_df = pd.DataFrame(feature_dict)
    return features_df


if __name__ == "__main__":

    path_for_features = "../../data/Engineered_features/"
    all_segment_features = []

    for i in range(6):
        filename = f'{path_for_features}eeg_time_frequency_domain_features_data_{i}.csv'
        if not os.path.isfile(filename):
            data, labels = load_data_as_windows(i)

            for idx in tqdm(range(data.shape[1]),
                            desc=f"Processing data of index {i} for frequency-time-domain features"):
                segment_data = data[:, idx, :]  # Shape: (5, 500)
                if labels is not None:
                    segment_labels = labels[:, idx]  # Shape: (5,)
                else:
                    segment_labels = None
                time_freq_features_df = compute_time_frequency_domain_features(segment_data, labels=segment_labels)

                all_segment_features.append(time_freq_features_df)

        final_features_df = pd.concat(all_segment_features, ignore_index=True)
        final_features_df.to_csv(filename, index=False)
        print(f"Time-frequency features and labels saved to {filename}")

# TODO wavelet based features time frequency spectral ratio, time frequency peaks.
