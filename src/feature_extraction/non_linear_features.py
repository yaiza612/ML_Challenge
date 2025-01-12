import os
import numpy as np
import pandas as pd
from scipy.stats import entropy
from scipy.spatial.distance import pdist, squareform
from statsmodels.tsa.ar_model import AutoReg
from sklearn.neighbors import NearestNeighbors
from src.data_loading.loader import load_data_as_windows
from tqdm import tqdm
from collections import Counter, defaultdict


def cross_entropy(signal_1, signal_2, bins=50):
    """
    Calculate the cross-entropy between two time series signals.

    Parameters:
    - signal_1: The first signal (1D numpy array).
    - signal_2: The second signal (1D numpy array).
    - bins: The number of bins to use for the histogram (default: 50).

    Returns:
    - cross_entropy: The cross-entropy between the two signals.
    """

    # Step 1: Compute the histogram for each signal
    hist_1, bin_edges_1 = np.histogram(signal_1, bins=bins, density=True)
    hist_2, bin_edges_2 = np.histogram(signal_2, bins=bins, density=True)

    # Step 2: Use a small epsilon to avoid log(0)
    epsilon = 1e-10
    hist_1 = hist_1 + epsilon
    hist_2 = hist_2 + epsilon

    # Step 3: Compute the cross-entropy (H(p, q) = -sum(p(x) * log(q(x)))
    cross_ent = np.sum(hist_1 * np.log(hist_2))

    return cross_ent


def permutation_entropy(signal, m=3, tau=1):
    """
    Calculate the Permutation Entropy of a time series.

    Parameters:
    - signal: The input time series (1D numpy array).
    - m: The embedding dimension (length of the embedding vectors).
    - tau: The time delay (usually 1 for regular time series).

    Returns:
    - perm_entropy: The permutation entropy of the time series.
    """
    n = len(signal)

    # Step 1: Create embedding vectors of length m
    # We use a sliding window of size m
    embedding_vectors = [signal[i:i + m] for i in range(0, n - m + 1, tau)]

    # Step 2: Find the ordinal pattern (permutation) for each vector
    permutations = []
    for vec in embedding_vectors:
        permutation = tuple(np.argsort(vec))  # Get the indices that would sort the vector
        permutations.append(permutation)

    # Step 3: Count the frequency of each permutation
    permutation_counts = Counter(permutations)

    # Step 4: Compute the probability of each permutation
    total_permutations = len(permutations)
    probabilities = [count / total_permutations for count in permutation_counts.values()]

    # Step 5: Calculate the Shannon entropy
    perm_entropy = -np.sum(np.array(probabilities) * np.log(probabilities))

    return perm_entropy


def poincare(signal):
    """
    Calculate Poincare plot features (SD1, SD2, and the ratio of SD2/SD1).

    Parameters:
    - signal: The input EEG signal (1D numpy array).

    Returns:
    - SD1: Standard deviation along the line of identity (short-term variability).
    - SD2: Standard deviation perpendicular to the line of identity (long-term variability).
    - ratio: The ratio of SD2 to SD1.
    """
    # Calculate the points for the Poincare plot: (x(t), x(t+1))
    x1 = signal[:-1]  # x(t)
    x2 = signal[1:]  # x(t+1)

    # Calculate SD1 and SD2
    diff = x2 - x1
    SD1 = np.std(diff) / np.sqrt(2)

    diff_perpendicular = np.sqrt(2 * np.std(x1 - np.mean(x1)) ** 2)  # Rough approximation of SD2
    SD2 = np.std(diff_perpendicular)

    # Compute ratio
    ratio = SD2 / SD1 if SD1 != 0 else np.nan

    return SD1, SD2, ratio


def time_delay_embedding(signal, dimension, delay):
    """
    Perform time-delay embedding of the signal.

    Parameters:
    - signal: 1D array-like, the input time series signal
    - dimension: int, embedding dimension (m)
    - delay: int, time delay (τ)

    Returns:
    - embedded: 2D array, the time-delay embedded signal
    """
    N = len(signal) - (dimension - 1) * delay
    embedded = np.empty((N, dimension))
    for i in range(dimension):
        embedded[:, i] = signal[i * delay:N + i * delay]
    return embedded


def calculate_largest_lyapunov_exponent(signal, fs, dimension=10, delay=1, max_iter=20):
    """
    Calculate the largest Lyapunov exponent using the Rosenstein method.

    Parameters:
    - signal: 1D array-like, the input time series signal
    - fs: int, sampling frequency (in Hz)
    - dimension: int, embedding dimension (default=10)
    - delay: int, time delay (default=1)
    - max_iter: int, maximum number of iterations (default=20)

    Returns:
    - lle: float, estimated largest Lyapunov exponent
    """
    # Step 1: Phase space reconstruction (time-delay embedding)
    embedded = time_delay_embedding(signal, dimension, delay)
    N = embedded.shape[0]

    # Step 2: Calculate the pairwise distances in phase space
    distances = squareform(pdist(embedded))

    # Step 3: Find nearest neighbors for each point
    nearest_neighbors = np.argmin(np.where(distances == 0, np.inf, distances), axis=1)

    # Step 4: Calculate divergence for each pair of neighbors over time
    divergences = np.zeros((N, max_iter))

    for i in range(N - max_iter):
        for j in range(max_iter):
            divergences[i, j] = np.linalg.norm(embedded[i + j] - embedded[nearest_neighbors[i] + j])

    # Step 5: Compute the logarithm of the average divergence
    log_divergence = np.mean(np.log(divergences + 1e-10), axis=0)  # Adding small constant to avoid log(0)

    # Step 6: Perform linear fit to estimate the slope (Lyapunov exponent)
    time = np.arange(max_iter) / fs
    slope, _ = np.polyfit(time, log_divergence, 1)

    return slope


# Hjorth Parameters
def compute_hjorth_params(signal):
    """Compute Hjorth mobility and complexity."""
    first_deriv = np.diff(signal)
    second_deriv = np.diff(first_deriv)
    var_zero = np.var(signal)
    var_diff1 = np.var(first_deriv)
    var_diff2 = np.var(second_deriv)
    mobility = np.sqrt(var_diff1 / var_zero)
    complexity = np.sqrt(var_diff2 / var_diff1) / mobility
    return mobility, complexity


# Shannon Entropy
def compute_shannon_entropy(signal):
    """Compute Shannon entropy of a signal."""
    hist, _ = np.histogram(signal, bins=10, density=True)
    return entropy(hist + np.finfo(float).eps)  # Add epsilon to avoid log(0)


# False Nearest Neighbors (FNN) method
def compute_false_nearest_neighbors(signal, dimension=2, tolerance=10):
    """Compute percentage of false nearest neighbors."""
    neighbors = NearestNeighbors(n_neighbors=dimension + 1).fit(signal.reshape(-1, 1))
    distances, indices = neighbors.kneighbors(signal.reshape(-1, 1))
    false_nearest = np.sum(distances[:, -1] > tolerance * distances[:, 1])
    return false_nearest / len(signal)


# ARMA Coefficients
def compute_arma_coefficients(signal, order=2):
    """Compute AR coefficients for a signal."""
    model = AutoReg(signal, lags=order).fit()
    return model.params[:order]  # Returns only the first 'order' coefficients


def compute_nonlinear_features(data, filtered_segment_data, fs=250, labels=None):
    """
    Computes non-linear features for each segment in EEG data.

    Parameters:
    - data: np.ndarray - EEG data, shape (5, 15425, 500), channels x segments x samples.
    - labels: np.ndarray - Optional, shape (5, 15425), labels for each segment and channel.

    Returns:
    - pd.DataFrame containing non-linear features for each segment.
    """
    feature_dict = defaultdict(list)

    # Process each channel and segment
    for ch in range(data.shape[0]):

        channel_data = data[ch, :]

        # Hjorth Parameters
        mobility, complexity = compute_hjorth_params(channel_data)
        feature_dict['hjorth_mobility'].append(mobility)
        feature_dict['hjorth_complexity'].append(complexity)

        # False Nearest Neighbors
        fnn = compute_false_nearest_neighbors(channel_data)
        feature_dict['false_nearest_neighbors'].append(fnn)

        # Shannon Entropy
        shannon_entropy = compute_shannon_entropy(channel_data)
        feature_dict['shannon_entropy'].append(shannon_entropy)

        # ARMA Coefficients
        arma_coeffs = compute_arma_coefficients(channel_data, order=2)
        feature_dict['arma_coef_1'].append(arma_coeffs[0])
        feature_dict['arma_coef_2'].append(arma_coeffs[1])

        # Lyapunov
        #lyapunov = calculate_largest_lyapunov_exponent(signal=channel_data, fs=fs)
        #feature_dict['lyapunov'].append(lyapunov)

        # Poincare features
        sd1, sd2, ratio = poincare(signal=channel_data)
        feature_dict['poincare_sd1'].append(sd1)
        feature_dict['poincare_sd2'].append(sd2)
        feature_dict['poincare_ratio'].append(ratio)

        # Permutation entropy
        perm_ent = permutation_entropy(signal=channel_data)
        feature_dict['permutation_entropy'].append(perm_ent)
    for channel_1_idx in range(5):
        channel_1 = data[channel_1_idx, :]
        for channel_2_idx in range(channel_1_idx):
            channel_2 = data[channel_2_idx, :]
            cross_ent = cross_entropy(signal_1=channel_1, signal_2=channel_2)
            feature_dict[f'cross_entropy_{channel_1_idx}_{channel_2_idx}'].extend([cross_ent] * 5)
    # Convert dictionary to DataFrame
    features_df = pd.DataFrame(feature_dict)
    return features_df


# Example usage
if __name__ == "__main__":
    path_for_features = "../../data/Engineered_features/"
    all_segment_features = []

    for i in (range(6)):
        filename = f'{path_for_features}eeg_nonlinear_features_data_{i}.csv'
        if not os.path.isfile(filename):
            data, labels = load_data_as_windows(i)
            if labels is not None:
                print(f"Labels shape: {labels.shape}")
            print(f"Data shape: {data.shape}")

            for idx in tqdm(range(data.shape[1]), desc=f"Processing data of index {i} for non-linear features"):
                segment_data = data[:, idx, :]  # Shape: (5, 500)

                if labels is not None:
                    segment_labels = labels[:, idx]  # Shape: (5,)
                else:
                    segment_labels = None

                nonlinear_features_df = compute_nonlinear_features(segment_data, labels=segment_labels)
                all_segment_features.append(nonlinear_features_df)

        final_features_df = pd.concat(all_segment_features, ignore_index=True)
        final_features_df.to_csv(filename, index=False)
        print(f"Non-linear features and labels saved to {filename}")
