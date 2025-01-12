import numpy as np
from tslearn.neighbors import KNeighborsTimeSeriesClassifier
from tslearn.metrics import dtw
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from sklearn.metrics import cohen_kappa_score, f1_score
from tqdm import tqdm


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

for odd_one_out in range(4):

    training_data = [(np.load(ROOT_PATH / f"data_{i}.npy")[:, :10000],
                      np.load(ROOT_PATH / f"target_{i}.npy")[:, :10000//250]) for i in range(4)]
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
    scaler = StandardScaler()
    X_train = np.concatenate([all_data[_] for _ in range(4) if _ != odd_one_out])
    X_train = np.concatenate([scaler.fit_transform(x_train.reshape(1, -1)) for x_train in X_train])
    X_test = all_data[odd_one_out + 1]
    X_test = np.concatenate([scaler.fit_transform(x_test.reshape(1, -1)) for x_test in X_test])
    y_train = np.concatenate([all_targets[_] for _ in range(4) if _ != odd_one_out])
    y_test = all_targets[odd_one_out + 1]
    #X_train, X_test, y_train, y_test = train_test_split(all_data, all_targets, test_size=0.2, random_state=42)

    # KNN classifier with DTW as distance metric
    knn_dtw = KNeighborsTimeSeriesClassifier(n_neighbors=5, metric="dtw")#,
                                             #metric_params={"global_constraint": "sakoe_chiba", "sakoe_chiba_radius": 1})
    print("fitting knn algorithm")
    knn_dtw.fit(X_train, y_train)

    # Test the classifier
    all_preds = []
    len_x_test = len(X_test)
    for sample in tqdm(X_test):
        all_preds.append(knn_dtw.predict(sample))
    print(cohen_kappa_score(all_preds, y_test))
