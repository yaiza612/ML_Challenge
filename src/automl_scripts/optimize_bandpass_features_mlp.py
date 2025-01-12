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
from sklearn.neural_network import MLPClassifier
import warnings

from src.automl_scripts.run_analyzer import analyze_run

warnings.filterwarnings("ignore")
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
    hidden_layer_2 = Integer("hidden_layer_2", bounds=(16, 64), log=True)
    hidden_layer_3 = Integer("hidden_layer_3", bounds=(8, 32), log=True)
    hidden_layer_4 = Integer("hidden_layer_4", bounds=(4, 16), log=True)
    activation = Categorical("activation", ("relu", "logistic", "tanh"))
    solver = Categorical("solver", ("lbfgs", "sgd", "adam"))
    alpha = Float("alpha", bounds=(0.000001, 0.1), log=True)
    batch_size = Integer("batch_size", bounds=(32, 256), log=True)
    learning_rate = Categorical("learning_rate", ("constant", "invscaling", "adaptive"))
    learning_rate_init = Float("learning_rate_init", bounds=(0.00001, 0.1), log=True)
    momentum = Float("momentum", bounds=(0.5, 0.99), log=True)
    beta_1 = Float("beta_1", bounds=(0.5, 0.99), log=True)
    beta_2 = Float("beta_2", bounds=(0.9, 0.99999), log=True)

    cs = ConfigurationSpace(seed=123)
    cs.add([band_1_low, band_2_low, band_3_low, band_1_high, band_2_high, band_3_high,
            order_1, order_2, order_3, hidden_layer_2, hidden_layer_3, hidden_layer_4, activation,
            solver, alpha, batch_size, learning_rate, learning_rate_init,
            momentum, beta_1, beta_2])
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
        clf = MLPClassifier(hidden_layer_sizes=(config["hidden_layer_2"],
                                                      config["hidden_layer_3"], config["hidden_layer_4"]),
                                  activation=config["activation"], solver=config["solver"],
                                  alpha=config["alpha"], batch_size=config["batch_size"],
                                  learning_rate=config["learning_rate"],
                                  learning_rate_init=config["learning_rate_init"], momentum=config["momentum"],
                                  beta_1=config["beta_1"], beta_2=config["beta_2"],
                                  early_stopping=True, max_iter=int(fidelity), max_fun=int(fidelity) * 75)
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

cs = create_search_space_rf()
dimensions = len(list(cs.values()))
dehb = DEHB(
    f=target_function,
    cs=cs,
    dimensions=dimensions,
    min_fidelity=6,
    max_fidelity=54,
    n_workers=1,
    output_path="./mlp_bandpass",
    resume=True,
    save_freq="step"
)
analyze_run(dehb, use_top_fidelities_only=True)
analyze_run(dehb, use_top_fidelities_only=False)

#trajectory, runtime, history = dehb.run(
#    total_cost=6*3600,
#    # parameters expected as **kwargs in target_function is passed here
#)
