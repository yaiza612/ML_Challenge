import numpy as np
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tslearn.metrics import dtw
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from sklearn.metrics import cohen_kappa_score, f1_score
from tqdm import tqdm
from tslearn.shapelets import LearningShapelets
from sklearn.svm import SVC


ROOT_PATH = Path("../../data/train")
# Sample time series data (X: 2D array where each row is a time series, y: labels
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

for odd_one_out in range(1, 4):

    training_data = [(np.load(ROOT_PATH / f"data_{i}.npy")[:, :5000],
                      np.load(ROOT_PATH / f"target_{i}.npy")[:, :5000//250]) for i in range(4)]
    all_data = []
    all_targets = []
    for (data, target) in tqdm(training_data):
        reshaped_data = reshape_array_into_windows(data, 250, 2)
        targets_flatten = target[..., :len(reshaped_data[0])].reshape(-1)
        #targets_flatten = target[channel_idx]
        reshaped_data = reshaped_data.reshape((-1, reshaped_data.shape[-1]))
        #reshaped_data = reshaped_data[channel_idx]
        all_data.append(reshaped_data)
        all_targets.append(targets_flatten)
    #all_data = np.concatenate(all_data)
    #all_targets = np.concatenate(all_targets)
    #assert all_data.shape[0] == all_targets.shape[0]
    #print(all_data.shape)


    # Train-test split
    X_train = np.concatenate([all_data[_] for _ in range(4) if _ != odd_one_out])
    X_test = all_data[(odd_one_out + 1) % 4]
    y_train = np.concatenate([all_targets[_] for _ in range(4) if _ != odd_one_out])
    y_test = all_targets[(odd_one_out + 1) % 4]
    #X_train, X_test, y_train, y_test = train_test_split(all_data, all_targets, test_size=0.2, random_state=42)

    # KNN classifier with DTW as distance metric
    clf = LearningShapelets(n_shapelets_per_size={4: 5, 10: 5, 20: 5}, max_iter=1000, verbose=0, scale=True,
                            optimizer="adam", weight_regularizer=0.05, batch_size=32)
    X_train = TimeSeriesScalerMeanVariance().fit_transform(X_train)
    X_test = TimeSeriesScalerMeanVariance().fit_transform(X_test)
    print("fitting")
    clf.fit(X_train, y_train)
    #print(test_distances)
    #print(test_distances.shape)
    #print(X_test.shape)
    preds_train = clf.predict(X_train)
    preds = clf.predict(X_test)
    print(cohen_kappa_score(preds, y_test))
    print(cohen_kappa_score(preds_train, y_train))
