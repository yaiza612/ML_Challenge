# First let's load the training data
from pathlib import Path
import numpy as np
from scipy.signal import butter, sosfilt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import cohen_kappa_score, f1_score
from sklearn.model_selection import LeaveOneGroupOut
import ConfigSpace as CS
from ConfigSpace import Float, GreaterThanCondition, Integer, Categorical, ConfigurationSpace
import time
from dehb import DEHB

from src.automl_scripts.run_analyzer import analyze_run
from src.data_loading.create_submission import create_submission_from_flattened_preds

do_plots = False
ROOT_PATH = Path("../../data/train")
TEST_PATH = Path("../../data/test")

training_data = [(np.array(np.load(ROOT_PATH / f"data_{i}.npy")/ 2**8, dtype=np.float16),
                  np.array(np.load(ROOT_PATH / f"target_{i}.npy"), dtype=np.float16)) for i in range(4)]

test_data = [(np.array(np.load(TEST_PATH / f"data_{i}.npy") / 2 ** 8, dtype=np.float16), None)
             for i in range(4, 6)]


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
    criterion = Categorical("criterion", ("gini", "entropy", "log_loss"))
    max_depth = Integer("max_depth", bounds=(2, 50), log=True)
    min_samples_leaf = Integer("min_samples_leaf", bounds=(1, 1024), log=True)
    max_features = Integer("max_features", bounds=(1, 9))
    class_weight = Categorical("class_weight", (None, "balanced", "balanced_subsample"))
    min_impurity_decrease = Float("min_impurity_decrease", bounds=(0.00001, 0.5), log=True)
    ccp_alpha = Float("ccp_alpha", bounds=(0.000001, 0.5), log=True)

    cs = ConfigurationSpace(seed=123)
    cs.add([band_1_low, band_2_low, band_3_low, band_1_high, band_2_high, band_3_high,
            order_1, order_2, order_3,  criterion, max_depth, min_samples_leaf, max_features, class_weight,
            min_impurity_decrease, ccp_alpha])
    print(cs)
    return cs


def target_function(config, fidelity):
    lows = [config["band_1_low"], config["band_2_low"], config["band_3_low"]]
    highs = [config["band_1_high"], config["band_2_high"], config["band_3_high"]]
    orders = [config["order_1"], config["order_2"], config["order_3"]]
    if lows[0] > highs[0] or lows[1] > highs[1] or lows[2] > highs[2]:
        return {"fitness": 1, "cost": 0}
    if lows[0] < lows[1] or lows[1] < lows[2] or highs[0] < highs[1] or highs[1] < highs[2]:
        return {"fitness": 1, "cost": 0}
    feature_list = []
    start_time = time.time()
    for low, high, order in zip(lows, highs, orders):
        amplitude, mean, variance = get_bandpass_features_from_training_data(training_data, low, high, order)
        feature_list.extend([amplitude, mean, variance])
    all_features = concatenate_features(feature_list)

    all_targets, all_groups = get_labels_and_groups_from_training_data(training_data)
    example_features = all_features
    all_predictions = []
    logo = LeaveOneGroupOut()
    logo.get_n_splits(X=example_features, y=all_targets, groups=all_groups)
    cohens_kappas = []

    for i, (train_index, test_index) in enumerate(logo.split(example_features, all_targets, all_groups)):
        train_features_for_knn = all_features[train_index]
        train_labels = all_targets[train_index]

        val_features_for_knn = all_features[test_index]
        val_labels = all_targets[test_index]
        clf = RandomForestClassifier(n_estimators=int(fidelity), criterion=config["criterion"],
                                     max_depth=config["max_depth"], min_samples_leaf=config["min_samples_leaf"],
                                     max_features=config["max_features"], n_jobs=-1,
                                     class_weight=config["class_weight"],
                                     min_impurity_decrease=config["min_impurity_decrease"],
                                     ccp_alpha=config["ccp_alpha"])
        clf.fit(train_features_for_knn, train_labels)
        prediction = clf.predict(val_features_for_knn)
        all_predictions.append(prediction)
        cohens_kappas.append(cohen_kappa_score(prediction, val_labels))
    total_preds = np.concatenate(all_predictions)
    cohens_kappas.append(cohen_kappa_score(total_preds, all_targets))
    end_time = time.time()
    cost = end_time - start_time
    intercept = -0.3893189175439966
    coefficients = [-0.66594295, - 0.91905975, - 0.38912936, - 0.13022785, 3.24499732]
    distance_from_optimal = 1 - np.array(cohens_kappas)
    fitness = float(np.linalg.norm(distance_from_optimal))
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


def make_test_prediction(config, fidelity):
    lows = [config["band_1_low"], config["band_2_low"], config["band_3_low"]]
    highs = [config["band_1_high"], config["band_2_high"], config["band_3_high"]]
    orders = [config["order_1"], config["order_2"], config["order_3"]]
    if lows[0] > highs[0] or lows[1] > highs[1] or lows[2] > highs[2]:
        return {"fitness": 1, "cost": 0}
    if lows[0] < lows[1] or lows[1] < lows[2] or highs[0] < highs[1] or highs[1] < highs[2]:
        return {"fitness": 1, "cost": 0}
    feature_list = []

    for low, high, order in zip(lows, highs, orders):
        amplitude, mean, variance = get_bandpass_features_from_training_data(training_data, low, high, order)
        feature_list.extend([amplitude, mean, variance])
    all_features = concatenate_features(feature_list)


    all_targets, all_groups = get_labels_and_groups_from_training_data(training_data)
    all_predictions = []
    for test_index in range(4, 6):
        test_feature_list = []
        for low, high, order in zip(lows, highs, orders):
            amplitude, mean, variance = get_bandpass_features_from_training_data([test_data[test_index - 4]], low, high, order)
            test_feature_list.extend([amplitude, mean, variance])
        all_features_test = concatenate_features(test_feature_list)
        train_features_for_knn = all_features
        train_labels = all_targets

        clf = RandomForestClassifier(n_estimators=int(fidelity), criterion=config["criterion"],
                                     max_depth=config["max_depth"], min_samples_leaf=config["min_samples_leaf"],
                                     max_features=config["max_features"], n_jobs=-1,
                                     class_weight=config["class_weight"],
                                     min_impurity_decrease=config["min_impurity_decrease"],
                                     ccp_alpha=config["ccp_alpha"])
        clf.fit(train_features_for_knn, train_labels)
        prediction = clf.predict_proba(all_features_test)
        all_predictions.append(prediction)
    return all_predictions

"""
config = {"band_1_high": 66.8345680587281, "band_1_low": 28.9397754500819, "band_2_high": 42.4962379926784, "band_2_low": 14.9219384378147, "band_3_high": 21.3291143077662, "band_3_low": 4.6853561606989, "ccp_alpha": 0.0078595375286, "class_weight": "balanced", "criterion": "gini", "max_depth": 23, "max_features": 3, "min_impurity_decrease": 0.0065340769632, "min_samples_leaf": 2, "order_1": 6, "order_2": 5, "order_3": 7}
score = 0.48164535888462995
info = {"data_0_score": 0.6936552197737065, "data_1_score": 0.8718752542667507, "data_2_score": 0.7561185636698465, "data_3_score": 0.7991454937722366, "data_all_score": 0.8520186186186222, "fidelity": 36.0}
#target_function(config, 108)

estimated kappa: 0.7433154657528913 at cost 40.30731821060181 with fitness 0.5299284523646036 for fidelity 36.0
cohens kappas: [0.6219024084628296, 0.8724062310781483, 0.7572046301706477, 0.8021363944542675, 0.8467465888747565]

all_predictions = make_test_prediction(config, 108)


def reorder_array(arr, num_columns):
    # Compute the number of rows needed
    num_rows = len(arr) // num_columns
    # Reshape the array
    reshaped = arr.reshape(num_columns, num_rows)
    # Flatten column-wise
    return reshaped.ravel(order='F')

# 0.5, 0.45, 0.55, 0.7, 0.8 (normal, _v2, _v3, _v4) (fidelity=108)
x = all_predictions[0]
first_prediction = np.where(x[:, 0] > 0.8, 0, 1)
final_prediction_first_set = reorder_array(first_prediction, 5)
x = all_predictions[1]
second_prediction = np.where(x[:, 0] > 0.8, 0, 1)
final_prediction_second_set = reorder_array(second_prediction, 5)
create_submission_from_flattened_preds(final_prediction_first_set, final_prediction_second_set, submission_name="optimized_rf_bandpass_v5")
"""

cs = create_search_space_rf()
dimensions = len(list(cs.values()))
dehb = DEHB(
    f=target_function,
    cs=cs,
    dimensions=dimensions,
    min_fidelity=4,
    max_fidelity=108,
    n_workers=1,
    output_path="./rf_bandpass",
    resume=True
)

analyze_run(dehb, use_top_fidelities_only=True)
analyze_run(dehb, use_top_fidelities_only=False)
"""
trajectory, runtime, history = dehb.run(
    total_cost=8*3600,
    # parameters expected as **kwargs in target_function is passed here
)
"""