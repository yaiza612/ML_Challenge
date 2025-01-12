# Spatial features in EEG help to understand the relationships between different channels
import os.path

import numpy as np
from scipy.signal import coherence, hilbert
import networkx as nx
from src.data_loading.loader import load_data_as_windows
import pandas as pd
from tqdm import tqdm


def compute_spatial_features(data, labels=None):
    # Frequency bands for coherence calculation
    theta_band = (4, 7)
    alpha_band = (8, 13)
    beta_band = (14, 30)
    gamma_band = (30, 50)

    fs = 250  # Sampling frequency (250 Hz)

    features_list = []

    # Loop over each segment
    for idx in range(data.shape[0]):  # Assuming data shape is (5, segments, samples)
        channel_data = data[idx, :]  # Shape: (5, 500), 5 channels per segment

        # Mean amplitude for each channel
        mean_amplitudes = np.mean(channel_data, axis=1)  # Shape: (5,)

        # Standard deviation for each channel
        std_amplitudes = np.std(channel_data, axis=1)  # Shape: (5,)

        # Compute correlation matrix between channels
        correlation_matrix = np.corrcoef(channel_data)
        mean_correlation = np.mean(correlation_matrix[np.triu_indices(5, k=1)])  # Mean of upper triangle (non-diagonal)

        # Phase Locking Value (PLV) for spatial connectivity
        phase_data = np.angle(hilbert(channel_data, axis=1))  # Extract phases
        plv_matrix = np.zeros((5, 5))
        for i in range(5):
            for j in range(i + 1, 5):
                phase_diff = phase_data[i] - phase_data[j]
                plv_matrix[i, j] = np.abs(np.mean(np.exp(1j * phase_diff)))
        mean_plv = np.mean(plv_matrix[np.triu_indices(5, k=1)])

        # Graph-based feature (clustering coefficient)
        G = nx.from_numpy_array(plv_matrix)
        clustering_coeff = np.mean(list(nx.clustering(G).values()))  # Average clustering coefficient

        # Coherence calculations
        coherences = []
        for i in range(5):
            for j in range(i + 1, 5):
                freqs, Cxy = coherence(channel_data[i], channel_data[j], fs=fs, nperseg=256)

                # Calculate average coherence within specific frequency bands
                theta_coh = np.mean(Cxy[(freqs >= theta_band[0]) & (freqs <= theta_band[1])])
                alpha_coh = np.mean(Cxy[(freqs >= alpha_band[0]) & (freqs <= alpha_band[1])])
                beta_coh = np.mean(Cxy[(freqs >= beta_band[0]) & (freqs <= beta_band[1])])
                gamma_coh = np.mean(Cxy[(freqs >= gamma_band[0]) & (freqs <= gamma_band[1])])

                # Collect coherence features
                coherences.extend([theta_coh, alpha_coh, beta_coh, gamma_coh])

        # Calculate the mean coherence values for each band across all pairs
        mean_theta_coh = np.mean(coherences[0::4])
        mean_alpha_coh = np.mean(coherences[1::4])
        mean_beta_coh = np.mean(coherences[2::4])
        mean_gamma_coh = np.mean(coherences[3::4])

        # Aggregate features channel by channel
        for ch in range(5):
            features = {
                'channel': ch + 1,
                'mean_amplitude': mean_amplitudes[ch],
                'std_amplitude': std_amplitudes[ch],
                'mean_correlation': mean_correlation,  # Shared across channels
                'mean_plv': mean_plv,  # Shared across channels
                'clustering_coeff': clustering_coeff,  # Shared across channels
                'mean_theta_coh': mean_theta_coh,  # Shared across channels
                'mean_alpha_coh': mean_alpha_coh,  # Shared across channels
                'mean_beta_coh': mean_beta_coh,  # Shared across channels
                'mean_gamma_coh': mean_gamma_coh,  # Shared across channels
            }

            # Add label if provided
            if labels is not None:
                features['label'] = labels[idx]

            features_list.append(features)

    # Create DataFrame from features list
    df_spatial_features = pd.DataFrame(features_list)
    return df_spatial_features


if __name__ == "__main__":

    path_for_features = "../../data/Engineered_features/"
    all_segment_features = []

    for i in range(6):
        filename = f'{path_for_features}eeg_spatial_features_data_{i}.csv'
        if not os.path.isfile(filename):
            data, labels = load_data_as_windows(i)

            for idx in tqdm(range(data.shape[1]), desc=f"Processing data of index {i} for spatial features"):
                # Extract the 2-second segment for all 5 channels
                segment_data = data[:, idx, :]  # Shape: (5, 500)
                if labels is not None:
                    segment_labels = labels[:, idx]
                else:
                    segment_labels = None

                spatial_f = compute_spatial_features(segment_data, labels=segment_labels)
                all_segment_features.append(spatial_f)

            final_features_df = pd.concat(all_segment_features, ignore_index=True)
            final_features_df.to_csv(filename, index=False)
            print(f"Features and labels saved to {filename}")


# TODO add asymmetries indices, region-specific features, source location features.

# TODO add connectivity based features: granger causality, global field power
#  , mutual information, imaginary coherence, graph theoretic measures.

# TODO add EMG artifact power, eye blink/EOG features.
#  Rate of change over time of banc power and entropy can also make sense.


