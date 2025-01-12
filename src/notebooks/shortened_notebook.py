# First let's load the training data
from pathlib import Path
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, sosfilt
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import cohen_kappa_score, f1_score
from sklearn.model_selection import LeaveOneGroupOut


do_plots = False
ROOT_PATH = Path("../../data/train")

# / 2**8, dtype=np.float16
#, dtype=np.float16
training_data = [(np.array(np.load(ROOT_PATH / f"data_{i}.npy")/ 2**8, dtype=np.float16)[:, :1000000],
                  np.array(np.load(ROOT_PATH / f"target_{i}.npy"), dtype=np.float16)[:, :1000000//250]) for i in range(4)]




# Let's have a look at the data duration
for i, (data, target) in enumerate(training_data):
    print()
    print(f"Recording {i}")
    print("Data shape", data.shape, target.shape)
    print("Data duration:", data.shape[1] / 250)
    print("Labels duration", target.shape[1] * 2)


def butter_bandpass(lowcut, highcut, fs, order=5):

    return butter(order, [lowcut, highcut], fs=fs, btype='band')


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    sos = butter(order, [lowcut, highcut], fs=fs, btype='band', output="sos")
    filtered_signal = sosfilt(sos, data)
    return filtered_signal



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


def concatenate_features(feature_list):
    return np.vstack(feature_list).T


def get_bandpass_features_from_training_data(train_data, low, high, order):
    all_data = []
    for data, target in train_data:
        filtered_data = butter_bandpass_filter(data, low, high, 250, order)
        reshaped_data = reshape_array_into_windows(filtered_data, 250, 2)
        reshaped_data = reshaped_data.reshape((-1, reshaped_data.shape[-1]))
        all_data.append(reshaped_data)
    all_data = np.concatenate(all_data)
    if np.any(np.isnan(all_data)):
        print("Nan detected")

    # We can now compute the mean, max and stdev over each 2 seconds segment to try to build features
    amplitude = (np.max(all_data, -1) - np.min(all_data, -1)).reshape(-1)
    variance = np.std(all_data, -1).reshape(-1)
    mean = np.mean(all_data, -1).reshape(-1)
    return amplitude, mean, variance

def get_labels_and_groups_from_training_data(train_data):
    all_targets = []
    groups = []
    for group_idx, (data, target) in enumerate(training_data):
        filtered_data = butter_bandpass_filter(data, 0.1, 18, 250, 4)
        reshaped_data = reshape_array_into_windows(filtered_data, 250, 2)
        targets_flatten = target[..., :len(reshaped_data[0])].reshape(-1)
        all_targets.append(targets_flatten)
        groups.append(np.ones_like(targets_flatten) * group_idx)
    all_targets = np.concatenate(all_targets)
    all_groups = np.concatenate(groups)
    return all_targets, all_groups
#{'delta': (0.5, 4), 'theta': (4, 8), 'alpha': (8, 13), 'beta': (13, 30), 'gamma': (30, 45)}
lows = [1, 3, 20]
highs = [20, 30, 35]

feature_list = []
for low, high in zip(lows, highs):
    amplitude, mean, variance = get_bandpass_features_from_training_data(training_data, low, high, 5)
    all_features = concatenate_features([amplitude, mean, variance])
    feature_list.append(all_features)

all_targets, all_groups = get_labels_and_groups_from_training_data(training_data)
example_features = feature_list[0]

logo = LeaveOneGroupOut()
logo.get_n_splits(X=example_features, y=all_targets, groups=all_groups)
cohens_kappas = []
for i, (train_index, test_index) in enumerate(logo.split(example_features, all_targets, all_groups)):
    train_features_for_knn = [f[train_index] for f in feature_list]
    train_labels = all_targets[train_index]
    val_features_for_knn = [f[test_index] for f in feature_list]
    val_labels = all_targets[test_index]
    neighbor_algs = [KNeighborsClassifier(n_neighbors=8) for _ in range(len(val_features_for_knn))]
    for neighbor_alg, train_features in zip(neighbor_algs, train_features_for_knn):
        neighbor_alg.fit(train_features, train_labels)
    predictions = [neighbor_alg.predict(val_features) for neighbor_alg, val_features in
                   zip(neighbor_algs, val_features_for_knn)]

    stacked_preds = np.vstack(predictions)
    z = np.sum(stacked_preds, axis=0)
    prediction = np.where(np.sum(stacked_preds, axis=0) > 1, 1, 0)

    cohens_kappas.append(cohen_kappa_score(prediction, val_labels))
print(np.mean(cohens_kappas))



"""
mean:
0.710298535437137
[1, 20, 5]
[0.32220941507072165, 0.7996311451586481, 0.8096352611603833, 0.909718320358795]

0.7313857361137066
[3, 30, 5]
[0.5346747603060276, 0.770582045258685, 0.7125305874021564, 0.9077555514879572]

0.6149990156876133
[20, 35, 5]
[0.5022297613943425, 0.6602943664880334, 0.5463276447984546, 0.7511442900696228]
"""
