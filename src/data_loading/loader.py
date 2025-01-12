# First let's load the training data
from pathlib import Path
import numpy as np


def load_train_data(train_idx):
    root_path = Path("../../data/train")
    return np.load(root_path / f"data_{train_idx}.npy"), np.load(root_path / f"target_{train_idx}.npy")


def load_test_data(test_idx):
    root_path = Path("../../data/test")
    return np.load(root_path / f"data_{test_idx}.npy")


def load_data(idx):
    if idx <= 3:
        return load_train_data(idx)
    else:
        return load_test_data(idx), None


def reshape_array_into_windows(x, sample_rate, window_duration_in_seconds):
    """
    Reshape the data into an array of shape (C, T, window) where 'window' contains
    the points corresponding to 'window_duration' seconds of data.

    Parameters:
    x (numpy array): The input data array.
    sample_rate (int): The number of samples per second.
    Returns:
    reshaped_x (numpy array): The reshaped array with shape (C, T, window).
    """
    # Calculate the number of samples in one window
    window_size = int(window_duration_in_seconds * sample_rate)

    # Ensure the total length of x is a multiple of window_size
    total_samples = x.shape[-1]
    if total_samples % window_size != 0:
        # Truncate or pad x to make it divisible by window_size
        x = x[..., :total_samples - (total_samples % window_size)]
    # Reshape x into (C, T, window)
    reshaped_x = x.reshape(x.shape[0], -1, window_size)

    return reshaped_x


def load_train_data_as_windows(train_idx):
    non_windowed_train_data, labels = load_train_data(train_idx)
    return reshape_array_into_windows(x=non_windowed_train_data, sample_rate=250, window_duration_in_seconds=2), labels


def load_test_data_as_windows(test_idx):
    non_windowed_test_data = load_test_data(test_idx)
    return reshape_array_into_windows(x=non_windowed_test_data, sample_rate=250, window_duration_in_seconds=2), None


def load_data_as_windows(idx):
    if idx <= 3:
        return load_train_data_as_windows(train_idx=idx)
    else:
        return load_test_data_as_windows(test_idx=idx)


def load_data_single_channel(idx):
    features, labels = load_data_as_windows(idx)
    features_ = np.reshape(features, newshape=(features.shape[0] * features.shape[1], 1, features.shape[2]), order="F")
    if labels is not None:
        labels_ = np.reshape(labels, (labels.shape[0] * labels.shape[1]), order="F")
    else:
        labels_ = None
    # features shape: 5, 15425, 500; features_: 5*15425, 1, 500
    # labels shape: 5, 15425; labels_: 5*15425
    assert np.array_equal(features[0, 0], features_[0, 0])
    assert np.array_equal(features[0, 100], features_[5 * 100, 0])
    assert np.array_equal(features[2, 100], features_[5 * 100 + 2, 0])
    if labels is not None:
        assert np.array_equal(labels[0, 100], labels_[5 * 100])
        assert np.array_equal(labels[1, 100], labels_[5 * 100 + 1])
    return features_, labels_


if __name__ == "__main__":
    data = load_data_single_channel(0)