import os
import numpy as np
from scipy.stats import skew, kurtosis
from src.data_loading.loader import load_data_as_windows
import pandas as pd
from tqdm import tqdm


def compute_time_domain_features(segment_data, filtered_segment_data, labels=None):
    """
    Computes time-domain features for each channel in the EEG data.

    Parameters:
    - eeg_data: np.array - EEG data with shape (channels, time_points).
    - labels: list - List of labels for each channel, or None if no labels provided.

    Returns:
    - features_dict: dict - A dictionary containing time-domain features for each channel.
    """
    features_dict = {'mean': [], 'std_dv': [], 'peak_to_peak': [], 'rms': [], 'skewness': [], 'kurt': []}

    # Iterate through each channel
    for ch in range(segment_data.shape[0]):
        channel_data = segment_data[ch, :]

        # Calculate time-domain features
        mean = np.mean(channel_data)
        features_dict['mean'].append(mean)
        std_dev = np.std(channel_data)
        features_dict['std_dv'].append(std_dev)
        peak_to_peak = np.ptp(channel_data)
        features_dict['peak_to_peak'].append(peak_to_peak)
        rms = np.sqrt(np.mean(channel_data ** 2))
        features_dict['rms'].append(rms)
        skewness = skew(channel_data)
        features_dict['skewness'].append(skewness)
        kurt = kurtosis(channel_data)
        features_dict['kurt'].append(kurt)

    # Convert the dictionary to a pandas DataFrame for easy saving
    features_df = pd.DataFrame(features_dict)
    return features_df


if __name__ == "__main__":
    path_for_features = "../../data/Engineered_features/"
    # Loop through the data and extract features
    for i in range(6):
        filename = f'{path_for_features}eeg_time_domain_features_data_{i}.csv'
        if not os.path.isfile(filename):
            all_segment_features = []
            # Load data and labels for segment `i`
            data, labels = load_data_as_windows(i)

            # Loop over each segment (assumed: data.shape = (5, 500, num_segments))
            for idx in tqdm(range(data.shape[1]), desc=f"Processing data of index {i} for time domain features"):
                # Extract the 2-second segment for all 5 channels
                segment_data = data[:, idx, :]  # Shape: (5, 500)
                # Get corresponding labels for the segment (assumes labels are per segment)
                if labels is not None:
                    segment_labels = labels[:, idx]  # Shape: (5,)
                else:
                    segment_labels = None
                # Compute time-domain features for this segment
                time_domain_features_df = compute_time_domain_features(segment_data, labels=segment_labels)
                all_segment_features.append(time_domain_features_df)

        # Concatenate all the features from different segments into one dataframe
            final_features_df = pd.concat(all_segment_features, ignore_index=True)
            final_features_df.to_csv(filename, index=False)
            print(f"Time-frequency features and labels saved to {filename}")


# TODO higher order statistics, energy features, envelop features, bursts dynamics in specific bands
#  , and event related features.