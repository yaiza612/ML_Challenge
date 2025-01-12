# First let's load the training data
from pathlib import Path
import numpy as np
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import LeaveOneGroupOut
import ConfigSpace as CS
from ConfigSpace import Float, Integer, Categorical, ConfigurationSpace
import time
from imp_features_boost import load_features_labels_and_groups
from dehb import DEHB
from sklearn.decomposition import PCA


do_plots = False
ROOT_PATH = Path("../../data/Engineered_features/")
TEST_PATH = Path("../../data/test")

train_data = [(np.load(ROOT_PATH / f"avg_power_multitaper_{i}.npy"),
                  np.load(ROOT_PATH / f"std_power_multitaper_{i}.npy"),
                  np.load(ROOT_PATH / f"amplitude_power_multitaper_{i}.npy"))
                 for i in range(4)]

#num_samples_set_0 = 77125
#num_samples_set_1 = 52320
#num_samples_set_2 = 64215
#num_samples_set_3 = 68095
#num_samples_set_4 = 66020
#num_samples_set_5 = 46595
#total_sample_sizes = [num_samples_set_0, num_samples_set_1, num_samples_set_2, num_samples_set_3]
#train_data = [(np.random.random(size=(total_sample_sizes[i], 5, 10)),
#               np.random.random(size=(total_sample_sizes[i], 5, 10))) for i in range(4)]

merged_data = []
for elem in train_data:
    merged_part = np.concatenate(elem, axis=2)
    print(merged_part.shape)
    merged_part = np.reshape(merged_part,
                             newshape=(merged_part.shape[0] * 5, merged_part.shape[-1]))
    merged_data.append(merged_part)

all_features = np.concatenate(merged_data)
print(all_features.shape)
pca = PCA(n_components=40)
all_features = pca.fit_transform(all_features)
print(all_features.shape)
print(pca.explained_variance_ratio_)

#total_samples = np.sum([num_samples_set_0, num_samples_set_1, num_samples_set_2,
#                        num_samples_set_3])
#all_features = np.random.random(size=(total_samples, 10))


targets, groups = load_features_labels_and_groups(0.01)
targets = np.asarray(targets).ravel()


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


def create_search_space_rf():
    """
    Parameter space to be optimized --- contains the hyperparameters
    """
    max_depth = Integer("max_depth", bounds=(2, 50), log=True)
    min_samples_leaf = Integer("min_samples_leaf", bounds=(1, 1024), log=True)
    max_features = Float("max_features", bounds=(0.1, 1.0), log=True)
    class_weight = Categorical("class_weight", (None, "balanced"))
    learning_rate = Float("learning_rate", bounds=(0.001, 0.5), log=True)
    max_leaf_nodes = Integer("max_leaf_nodes", bounds=(2, 128), log=True)
    l2_regularization = Float("l2_regularization", bounds=(0.000001, 0.1), log=True)
    interaction_cst = Categorical("interaction_cst", (None, "pairwise", "no_interactions"))
    cs = ConfigurationSpace(seed=123)
    cs.add([max_depth, min_samples_leaf, max_features,
            class_weight, learning_rate, max_leaf_nodes, l2_regularization, interaction_cst])
    return cs


def target_function(config, fidelity):
    start_time = time.time()

    example_features = all_features
    all_predictions = []
    logo = LeaveOneGroupOut()
    logo.get_n_splits(X=example_features, y=targets, groups=groups)
    cohens_kappas = []

    for i, (train_index, test_index) in enumerate(logo.split(example_features, targets, groups)):
        train_features_for_knn = all_features[train_index]
        train_labels = targets[train_index]

        val_features_for_knn = all_features[test_index]
        val_labels = targets[test_index]
        clf = HistGradientBoostingClassifier(learning_rate=config["learning_rate"], max_iter=int(fidelity),
                                                   max_leaf_nodes=config["max_leaf_nodes"],
                                                   max_depth=config["max_depth"],
                                                   min_samples_leaf=config["min_samples_leaf"],
                                                   l2_regularization=config["l2_regularization"],
                                                   max_features=config["max_features"],
                                                   interaction_cst=config["interaction_cst"],
                                                   class_weight=config["class_weight"])
        clf.fit(train_features_for_knn, train_labels)
        prediction = clf.predict(val_features_for_knn)
        all_predictions.append(prediction)
        cohens_kappas.append(cohen_kappa_score(prediction, val_labels))
    total_preds = np.concatenate(all_predictions)
    cohens_kappas.append(cohen_kappa_score(total_preds, targets))
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
    min_fidelity=36,
    max_fidelity=972,
    n_workers=1,
    output_path="./histgrad_multitaper",
    save_freq="step"
)

trajectory, runtime, history = dehb.run(
    total_cost=8*3600,
    # parameters expected as **kwargs in target_function is passed here
)
