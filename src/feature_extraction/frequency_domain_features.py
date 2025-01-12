import os
import numpy as np
import pandas as pd
from scipy.signal import welch, find_peaks, butter, sosfilt, hilbert
from scipy.stats import entropy, linregress
from scipy.integrate import simpson
from src.data_loading.loader import load_data_as_windows
from tqdm import tqdm


def bandpass_filter(signal, fs, low, high, order=4):
    """
    Bandpass filter the signal to isolate the desired frequency band.

    Parameters:
    - signal: 1D array-like, the time-domain signal
    - fs: float, sampling frequency of the signal (in Hz)
    - lowcut: float, lower bound of the frequency band (in Hz)
    - highcut: float, upper bound of the frequency band (in Hz)
    - order: int, order of the Butterworth filter

    Returns:
    - filtered_signal: the bandpass filtered signal
    """
    sos = butter(order, [low, high], fs=fs, btype='band', output="sos")
    filtered_signal = sosfilt(sos, signal)
    return filtered_signal


def hilbert_transform_instantaneous_frequency(signal, fs):
    """
    Apply Hilbert Transform and calculate instantaneous frequency.

    Parameters:
    - signal: 1D array-like, the bandpass-filtered time-domain signal
    - fs: float, sampling frequency of the signal (in Hz)

    Returns:
    - instantaneous_frequency: instantaneous frequency of the signal
    - analytic_signal: the analytic signal from the Hilbert Transform
    - amplitude_envelope: the instantaneous amplitude envelope of the signal
    """
    # Step 1: Compute the analytic signal using Hilbert Transform
    analytic_signal = hilbert(signal)

    # Step 2: Compute the instantaneous phase
    instantaneous_phase = np.unwrap(np.angle(analytic_signal))

    # Step 3: Compute the instantaneous frequency as the derivative of phase
    instantaneous_frequency = np.diff(instantaneous_phase) / (2.0 * np.pi) * fs

    return instantaneous_frequency


def summarize_frequency_shifts(filtered_signal, fs, lowcut, highcut):
    """
    Summarize frequency shifts within a specific frequency band using mean and std of instantaneous frequency.

    Parameters:
    - filtered_signal: 1D array-like, the filtered time-domain signal
    - fs: float, sampling frequency of the signal (in Hz)
    - lowcut: float, lower bound of the frequency band (in Hz)
    - highcut: float, upper bound of the frequency band (in Hz)

    Returns:
    - mean_frequency: float, mean of the instantaneous frequency within the band
    - std_frequency: float, standard deviation of the instantaneous frequency (variability)
    """

    # Step 2: Apply the Hilbert Transform to get the instantaneous frequency
    instantaneous_frequency = hilbert_transform_instantaneous_frequency(filtered_signal, fs)

    # Step 3: Calculate mean and standard deviation of the instantaneous frequency
    mean_frequency = np.mean(instantaneous_frequency)
    std_frequency = np.std(instantaneous_frequency)
    frequency_range = np.max(instantaneous_frequency) - np.min(instantaneous_frequency)

    time = np.arange(len(instantaneous_frequency)) / fs  # time in seconds
    slope, intercept, r_value, p_value, std_err = linregress(time, instantaneous_frequency)

    return mean_frequency, std_frequency, frequency_range, slope


def compute_frequency_domain_features(data, labels, filtered_segment_data, fs=250):
    """
    Extracts frequency domain features from EEG data for each segment.

    Parameters:
    - data: np.ndarray - EEG data, shape (5, 15425, 500), channels x segments x samples.
    - fs: int - Sampling frequency (in Hz), default is 250.
    - labels: np.ndarray or list - Labels for each channel per segment, shape (5,).

    Returns:
    - pd.DataFrame containing frequency domain features and labels for each segment.
    """
    # Frequency bands (in Hz)
    freq_bands = {'delta': (0.5, 4), 'theta': (4, 8), 'alpha': (8, 13), 'beta': (13, 30), 'gamma': (30, 45)}

    # Feature storage
    feature_dict = {'psd_total': [], 'tonality_coefficient_thr_2': [], 'tonality_coefficient_thr_3': [],
                    'freq_domain_peak_frequency': [], 'freq_domain_spectral_entropy': [], 'median_frequency': [], 'sharp_spike': [],
                    'number_of_bursts': [], 'power_std': [], 'diffuse_slowing': [],
                    'low_signal_amplitude': [], 'spectral_centroid': []
    }
    for band in freq_bands:
        feature_dict[f'band_power_{band}'] = []
        feature_dict[f'band_ratio_{band}'] = []
        feature_dict[f'band_edge_low_{band}'] = []
        feature_dict[f'band_edge_high_{band}'] = []
        feature_dict[f'band_spectral_centroid_{band}'] = []
        feature_dict[f'band_mean_frequency_{band}'] = []
        feature_dict[f'band_std_frequency_{band}'] = []
        feature_dict[f'band_frequency_range_{band}'] = []
        feature_dict[f'band_slope_{band}'] = []

    # Process each channel and segment
    for ch in range(data.shape[0]):

        channel_data = data[ch, :]

        # Power Spectral Density (PSD) using Welch's method
        freqs, psd = welch(channel_data, fs, nperseg=fs*2)
        total_power = np.sum(psd)
        feature_dict['psd_total'].append(total_power)

        # Tonality coefficients for different thresholds
        peaks_2, _ = find_peaks(psd, height=np.mean(psd) * 2)
        tonal_power_2 = sum(psd[peaks_2])
        feature_dict['tonality_coefficient_thr_2'].append(tonal_power_2/total_power)

        peaks_3, _ = find_peaks(psd, height=np.mean(psd) * 3)
        tonal_power_3 = sum(psd[peaks_3])
        feature_dict['tonality_coefficient_thr_3'].append(tonal_power_3 / total_power)

        # Peak Frequency
        peak_freq = freqs[np.argmax(psd)]
        feature_dict['freq_domain_peak_frequency'].append(peak_freq)

        # Spectral Entropy
        spectral_entropy = entropy(psd / total_power)
        feature_dict['freq_domain_spectral_entropy'].append(spectral_entropy)

        # Median Frequency
        cumulative_sum = np.cumsum(psd)
        median_freq = freqs[np.argmax(cumulative_sum >= (total_power / 2))]
        feature_dict['median_frequency'].append(median_freq)

        # Sharp Spike (Peak-to-Mean Ratio)
        sharp_spike = np.max(channel_data) / np.mean(channel_data)
        feature_dict['sharp_spike'].append(sharp_spike)

        # Number of Bursts (Threshold-based)
        burst_threshold = 2 * np.std(channel_data)
        number_of_bursts = np.sum(channel_data > burst_threshold)
        feature_dict['number_of_bursts'].append(number_of_bursts)

        # Standard Deviation of Power (as power variability)
        feature_dict['power_std'].append(np.std(psd))

        # Diffuse Slowing (Ratio of delta band power to total power)
        delta_band = freq_bands['delta']
        delta_power = np.sum(psd[(freqs >= delta_band[0]) & (freqs <= delta_band[1])])
        diffuse_slowing = delta_power / total_power
        feature_dict['diffuse_slowing'].append(diffuse_slowing)

        # Low Signal Amplitude (Mean absolute amplitude)
        low_signal_amplitude = np.mean(np.abs(channel_data))
        feature_dict['low_signal_amplitude'].append(low_signal_amplitude)

        # Spectral Centroid (Center of Gravity)
        spectral_centroid = np.sum(freqs * psd) / np.sum(psd)
        feature_dict['spectral_centroid'].append(spectral_centroid)

        # Band Power and Band Ratios
        band_powers = {}
        for band, (fmin, fmax) in freq_bands.items():
            band_power = simpson(psd[(freqs >= fmin) & (freqs <= fmax)], dx=freqs[1] - freqs[0])
            band_powers[band] = band_power
            feature_dict[f'band_power_{band}'].append(band_power)
            band_indices = (freqs > fmin) & (freqs < fmax)
            power_threshold = 0.1 * max(psd[band_indices])
            frequencies = freqs[band_indices][psd[band_indices] > power_threshold]
            if len(frequencies) != 0:
                feature_dict[f'band_edge_low_{band}'].append(min(frequencies))
                feature_dict[f'band_edge_high_{band}'].append(max(frequencies))
            else:
                feature_dict[f'band_edge_low_{band}'].append(np.nan)
                feature_dict[f'band_edge_high_{band}'].append(np.nan)
            spectral_centroid_band = np.sum(freqs[band_indices] * psd[band_indices]) / np.sum(psd[band_indices])
            feature_dict[f'band_spectral_centroid_{band}'].append(spectral_centroid_band)
            mean_frequency, std_frequency, frequency_range, slope = \
                summarize_frequency_shifts(filtered_signal=filtered_segment_data[band][ch], fs=fs, lowcut=fmin, highcut=fmax)
            feature_dict[f'band_mean_frequency_{band}'].append(mean_frequency)
            feature_dict[f'band_std_frequency_{band}'].append(std_frequency)
            feature_dict[f'band_frequency_range_{band}'].append(frequency_range)
            feature_dict[f'band_slope_{band}'].append(slope)
        for band in freq_bands:
            feature_dict[f'band_ratio_{band}'].append(band_powers[band] / total_power)

    # Convert dictionary to DataFrame
    features_df = pd.DataFrame(feature_dict)

    return features_df


if __name__ == "__main__":

    path_for_features = "../../data/Engineered_features/"
    all_segment_features = []

    for i in range(6):
        filename = f'{path_for_features}eeg_frequency_domain_features{i}.csv'
        if not os.path.isfile(filename):

            data, labels = load_data_as_windows(i)

            for idx in tqdm(range(data.shape[1]), desc=f"Processing data of index {i} for frequency features"):
                segment_data = data[:, idx, :]  # Shape: (5, 500)
                if labels is not None:
                    segment_labels = labels[:, idx]  # Shape: (5,)
                else:
                    segment_labels = None
                freq_domain_features_df = compute_frequency_domain_features(segment_data, labels=segment_labels)

                all_segment_features.append(freq_domain_features_df)

            final_features_df = pd.concat(all_segment_features, ignore_index=True)
            final_features_df.to_csv(filename, index=False)
            print(f"Features and labels saved to {filename}")

# TODO calculate relative band_power (example: alpha power/ total power), harmonic ratios
