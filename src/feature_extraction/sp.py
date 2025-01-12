import os
import numpy as np
import pandas as pd
import networkx as nx
from scipy.signal import hilbert, coherence
from src.data_loading.loader import load_data_as_windows
from tqdm import tqdm


def compute_spatial_features(segment_data, labels=None, filtered_segment_data=None):
    """
    Compute spatial features for a single segment of EEG data.

    Parameters:
    - segment_data: np.ndarray, shape (5, 500) - EEG data for a single segment (5 channels, 500 samples).
    - labels: list or np.ndarray, optional - Labels corresponding to this segment.

    Returns:
    - pd.DataFrame - A DataFrame containing spatial features for each channel in the segment.
    """
    # Frequency bands for coherence calculation
    delta_band = (0.5, 4)
    theta_band = (4, 8)
    alpha_band = (8, 13)
    beta_band = (13, 30)
    gamma_band = (30, 45)

    fs = 250  # Sampling frequency (250 Hz)

    features_list = []

    # Mean amplitude for each channel
    mean_amplitudes = np.mean(segment_data, axis=1)  # Shape: (5,)
    std_amplitudes = np.std(segment_data, axis=1)  # Shape: (5,)

    # Compute correlation matrix between channels
    correlation_matrix = np.corrcoef(segment_data)
    mean_correlation = np.mean(correlation_matrix[np.triu_indices(5, k=1)])  # Mean of upper triangle (non-diagonal)

    # Phase Locking Value (PLV) for spatial connectivity
    phase_data = np.angle(hilbert(segment_data, axis=1))  # Extract phases
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
            freqs, Cxy = coherence(segment_data[i], segment_data[j], fs=fs, nperseg=256)

            # Calculate average coherence within specific frequency bands
            delta_coh = np.mean(Cxy[(freqs >= delta_band[0]) & (freqs <= delta_band[1])])
            theta_coh = np.mean(Cxy[(freqs >= theta_band[0]) & (freqs <= theta_band[1])])
            alpha_coh = np.mean(Cxy[(freqs >= alpha_band[0]) & (freqs <= alpha_band[1])])
            beta_coh = np.mean(Cxy[(freqs >= beta_band[0]) & (freqs <= beta_band[1])])
            gamma_coh = np.mean(Cxy[(freqs >= gamma_band[0]) & (freqs <= gamma_band[1])])

            # Collect coherence features
            coherences.extend([delta_coh, theta_coh, alpha_coh, beta_coh, gamma_coh])

    # Calculate the mean coherence values for each band across all pairs
    mean_delta_coh = np.mean(coherences[0::5])
    mean_theta_coh = np.mean(coherences[1::5])
    mean_alpha_coh = np.mean(coherences[2::5])
    mean_beta_coh = np.mean(coherences[3::5])
    mean_gamma_coh = np.mean(coherences[4::5])

    # Aggregate features channel by channel
    for ch in range(5):
        features = {
            'channel': ch + 1,
            'mean_amplitude': mean_amplitudes[ch],
            'std_amplitude': std_amplitudes[ch],
            'mean_correlation': mean_correlation,  # Shared across channels
            'mean_plv': mean_plv,  # Shared across channels
            'clustering_coeff': clustering_coeff,  # Shared across channels
            'mean_delta_coh': mean_delta_coh,  # Shared across channels
            'mean_theta_coh': mean_theta_coh,  # Shared across channels
            'mean_alpha_coh': mean_alpha_coh,  # Shared across channels
            'mean_beta_coh': mean_beta_coh,  # Shared across channels
            'mean_gamma_coh': mean_gamma_coh,  # Shared across channels
        }

        features_list.append(features)

    # Create DataFrame from features list
    return pd.DataFrame(features_list)


if __name__ == "__main__":
    path_for_features = "../../data/Engineered_features/"

    for i in range(6):
        filename = f"{path_for_features}eeg_spatial_features_data_{i}.csv"
        if not os.path.isfile(filename):
            all_segment_features = []
            data, labels = load_data_as_windows(i)

            for idx in tqdm(range(data.shape[1]), desc=f"Processing train data of index {i} for spatial features"):
                segment_data = data[:, idx, :]  # Shape: (5, 500)
                if labels is not None:
                    segment_labels = labels[:, idx]  # Labels for this segment
                else:
                    segment_labels = None

                spatial_features = compute_spatial_features(segment_data, labels=segment_labels)
                all_segment_features.append(spatial_features)

            final_features_df = pd.concat(all_segment_features, ignore_index=True)
            final_features_df.to_csv(filename, index=False)
            print(f"Features and labels saved to {filename}")
