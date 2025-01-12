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
import ConfigSpace as CS
from ConfigSpace import Float, GreaterThanCondition, Integer, Categorical, ConfigurationSpace, \
    ForbiddenLessThanRelation, ForbiddenEqualsClause
import time
from sklearn.preprocessing import StandardScaler
from dehb import DEHB

from src.automl_scripts.run_analyzer import analyze_run


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


def create_search_space_rf():
    """
    Parameter space to be optimized --- contains the hyperparameters
    """
    band_1_low = Float('band_1_low', bounds=(0.5, 70))
    band_2_low = Float('band_2_low', bounds=(0.5, 70))
    band_3_low = Float('band_3_low', bounds=(0.5, 70))
    band_1_high = Float('band_1_high', bounds=(0.5, 70))
    band_2_high = Float('band_2_high', bounds=(0.5, 70))
    band_3_high = Float('band_3_high', bounds=(0.5, 70))
    order_1 = Integer("order_1", bounds=(3, 8))
    order_2 = Integer("order_2", bounds=(3, 8))
    order_3 = Integer("order_3", bounds=(3, 8))
    num_neighbors_1 = Integer("num_neighbors_1", bounds=(2, 80))
    num_neighbors_2 = Integer("num_neighbors_2", bounds=(2, 80))
    num_neighbors_3 = Integer("num_neighbors_3", bounds=(2, 80))
    weights_1 = Categorical('weights_1', ("uniform", "distance"))
    weights_2 = Categorical('weights_2', ("uniform", "distance"))
    weights_3 = Categorical('weights_3', ("uniform", "distance"))
    p_1 = Float("p_1", bounds=(1, 5))
    p_2 = Float("p_2", bounds=(1, 5))
    p_3 = Float("p_3", bounds=(1, 5))
    use_amplitude_1 = Categorical("use_amplitude_1", (True, False))
    use_amplitude_2 = Categorical("use_amplitude_2", (True, False))
    use_amplitude_3 = Categorical("use_amplitude_3", (True, False))
    use_mean_1 = Categorical("use_mean_1", (True, False))
    use_mean_2 = Categorical("use_mean_2", (True, False))
    use_mean_3 = Categorical("use_mean_3", (True, False))
    use_variance_1 = Categorical("use_variance_1", (True, False))
    use_variance_2 = Categorical("use_variance_2", (True, False))
    use_variance_3 = Categorical("use_variance_3", (True, False))
    cs = ConfigurationSpace(seed=123)
    cs.add([band_1_low, band_2_low, band_3_low, band_1_high, band_2_high, band_3_high,
            order_1, order_2, order_3,  num_neighbors_1, num_neighbors_2, num_neighbors_3,
            weights_1, weights_2, weights_3, p_1, p_2, p_3,
            use_amplitude_1, use_amplitude_2, use_amplitude_3,
            use_mean_1, use_mean_2, use_mean_3, use_variance_1, use_variance_2, use_variance_3])
    #forbidden_clause_band_1 = ForbiddenLessThanRelation(cs["band_1_high"], cs["band_1_low"])
    #forbidden_clause_band_2 = ForbiddenLessThanRelation(cs["band_2_high"], cs["band_2_low"])
    #forbidden_clause_band_3 = ForbiddenLessThanRelation(cs["band_3_high"], cs["band_3_low"])
    #forbidden_clause_band_1_2_low = ForbiddenLessThanRelation(cs['band_1_low'], cs['band_2_low'])
    #forbidden_clause_band_2_3_low = ForbiddenLessThanRelation(cs['band_2_low'], cs['band_3_low'])
    #forbidden_clause_band_1_2_high = ForbiddenLessThanRelation(cs['band_1_high'], cs['band_2_high'])
    #forbidden_clause_band_2_3_high = ForbiddenLessThanRelation(cs['band_2_high'], cs['band_3_high'])
    #forbidden_clause_use_1_feature_1 = CS.ForbiddenAndConjunction(
    #    CS.ForbiddenEqualsClause(cs['use_amplitude_1'], False),
    #    CS.ForbiddenEqualsClause(cs['use_mean_1'], False),
    #    CS.ForbiddenEqualsClause(cs['use_variance_1'], False)
    #)
    #forbidden_clause_use_1_feature_2 = CS.ForbiddenAndConjunction(
    #    CS.ForbiddenEqualsClause(cs['use_amplitude_2'], False),
    #    CS.ForbiddenEqualsClause(cs['use_mean_2'], False),
    #    CS.ForbiddenEqualsClause(cs['use_variance_2'], False)
    #)
    #forbidden_clause_use_1_feature_3 = CS.ForbiddenAndConjunction(
    #    CS.ForbiddenEqualsClause(cs['use_amplitude_3'], False),
    #    CS.ForbiddenEqualsClause(cs['use_mean_3'], False),
    #    CS.ForbiddenEqualsClause(cs['use_variance_3'], False)
    #)
    #cs.add([forbidden_clause_band_1, forbidden_clause_band_2, forbidden_clause_band_3,
    #        forbidden_clause_band_1_2_low, forbidden_clause_band_2_3_low,
    #                          forbidden_clause_band_1_2_high, forbidden_clause_band_2_3_high,
    #                          forbidden_clause_use_1_feature_1, forbidden_clause_use_1_feature_2,
    #                          forbidden_clause_use_1_feature_3])
    print(cs)
    return cs



def target_function(config, fidelity):
    subsample_ratio = fidelity / 27
    lows = [config["band_1_low"], config["band_2_low"], config["band_3_low"]]
    highs = [config["band_1_high"], config["band_2_high"], config["band_3_high"]]
    orders = [config["order_1"], config["order_2"], config["order_3"]]
    use_amplitudes = [config["use_amplitude_1"], config["use_amplitude_2"], config["use_amplitude_3"]]
    use_means = [config["use_mean_1"], config["use_mean_2"], config["use_mean_3"]]
    use_variances = [config["use_variance_1"], config["use_variance_2"], config["use_variance_3"]]
    if lows[0] > highs[0] or lows[1] > highs[1] or lows[2] > highs[2]:
        #print("Invalid Config generated")
        return {"fitness": 1, "cost": 0}
    if lows[0] < lows[1] or lows[1] < lows[2] or highs[0] < highs[1] or highs[1] < highs[2]:
        #print("Invalid Config generated")
        return {"fitness": 1, "cost": 0}
    if not use_amplitudes[0] and not use_means[0] and not use_variances[0]:
        #print("Invalid Config generated")
        return {"fitness": 1, "cost": 0}
    if not use_amplitudes[1] and not use_means[1] and not use_variances[1]:
        #print("Invalid Config generated")
        return {"fitness": 1, "cost": 0}
    if not use_amplitudes[2] and not use_means[2] and not use_variances[2]:
        #print("Invalid Config generated")
        return {"fitness": 1, "cost": 0}
    feature_list = []
    start_time = time.time()
    for low, high, order, use_amplitude, use_mean, use_variance in zip(lows, highs, orders,
                                                                       use_amplitudes, use_means, use_variances):
        amplitude, mean, variance = get_bandpass_features_from_training_data(training_data, low, high, order)
        features_in_list = []
        if use_amplitude:
            features_in_list.append(amplitude)
        if use_mean:
            features_in_list.append(mean)
        if use_variance:
            features_in_list.append(variance)
        all_features = concatenate_features(features_in_list)
        feature_list.append(all_features)

    all_targets, all_groups = get_labels_and_groups_from_training_data(training_data)
    example_features = feature_list[0]
    all_predictions = []
    logo = LeaveOneGroupOut()
    logo.get_n_splits(X=example_features, y=all_targets, groups=all_groups)
    cohens_kappas = []

    for i, (train_index, test_index) in enumerate(logo.split(example_features, all_targets, all_groups)):
        scalers = [StandardScaler(), StandardScaler(), StandardScaler()]
        train_features_for_knn = [f[train_index] for f in feature_list]
        train_labels = all_targets[train_index]
        n_samples = int(len(train_index) * subsample_ratio)
        subsample_indices = np.random.choice(len(train_index), n_samples, replace=False)
        train_features_for_knn = [f[subsample_indices] for f in train_features_for_knn]
        train_labels = train_labels[subsample_indices]
        train_features_for_knn = [scaler.fit_transform(f) for scaler, f in zip(scalers, train_features_for_knn)]

        val_features_for_knn = [f[test_index] for f in feature_list]
        val_features_for_knn = [scaler.transform(f) for scaler, f in zip(scalers, val_features_for_knn)]
        val_labels = all_targets[test_index]
        neighbor_algs = [KNeighborsClassifier(n_neighbors=config[f"num_neighbors_{_+1}"], weights=config[f"weights_{_+1}"],
                                              p=config[f"p_{_+1}"], n_jobs=-1) for _ in range(len(val_features_for_knn))]
        for neighbor_alg, train_features in zip(neighbor_algs, train_features_for_knn):
            neighbor_alg.fit(train_features, train_labels)
        predictions = [neighbor_alg.predict(val_features) for neighbor_alg, val_features in
                       zip(neighbor_algs, val_features_for_knn)]

        stacked_preds = np.vstack(predictions)
        prediction = np.where(np.sum(stacked_preds, axis=0) > 1, 1, 0)
        all_predictions.append(prediction)
        cohens_kappas.append(cohen_kappa_score(prediction, val_labels))
    total_preds = np.concatenate(all_predictions)
    cohens_kappas.append(cohen_kappa_score(total_preds, all_targets))
    end_time = time.time()
    cost = end_time - start_time
    intercept = -0.3893189175439966
    coefficients = [-0.66594295, - 0.91905975, - 0.38912936, - 0.13022785, 3.24499732]
    distance_from_optimal = 1 - np.array(cohens_kappas)
    fitness = float(np.linalg.norm(distance_from_optimal)) - 100
    estimated_kappa = np.dot(cohens_kappas, coefficients) + intercept
    print(f"estimated kappa: {estimated_kappa} at cost {cost} with fitness {fitness} for fidelity {fidelity}")
    print(f"cohens kappas: {cohens_kappas}")
    result = {
        "fitness": fitness,  # DE/DEHB minimizes
        "cost": cost,
        "info": {
            "data_0_score": cohens_kappas[0],
            "data_1_score": cohens_kappas[1],
            "data_2_score": cohens_kappas[2],
            "data_3_score": cohens_kappas[3],
            "data_all_score": cohens_kappas[4],
            "fidelity": fidelity
        }
    }
    return result

if __name__ == "__main__":
    ROOT_PATH = Path("../../data/train")
    training_data = [(np.array(np.load(ROOT_PATH / f"data_{i}.npy") / 2 ** 8, dtype=np.float16),
                      np.array(np.load(ROOT_PATH / f"target_{i}.npy"), dtype=np.float16)) for i in range(4)]
    get_labels_and_groups_from_training_data(training_data)
    cs = create_search_space_rf()
    dimensions = len(list(cs.values()))
    dehb = DEHB(
        f=target_function,
        cs=cs,
        dimensions=dimensions,
        min_fidelity=9,
        max_fidelity=27,
        n_workers=1,
        output_path="./knn_bandpass",
        resume=True
    )
    analyze_run(dehb, use_top_fidelities_only=True)
    analyze_run(dehb, use_top_fidelities_only=False)
    #trajectory, runtime, history = dehb.run(
    #    total_cost=3*3600,
    #    # parameters expected as **kwargs in target_function is passed here
    #)