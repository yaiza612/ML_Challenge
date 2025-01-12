import numpy as np
import pandas as pd
from sklearn.decomposition import FactorAnalysis

from src.data_loading.loader import load_data_as_windows, load_data_single_channel
from sklearn.preprocessing import StandardScaler
from scipy import signal
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.metrics import cohen_kappa_score
import time
from pathlib import Path
from sklearn.model_selection import LeaveOneGroupOut
import ConfigSpace as CS
from ConfigSpace import Integer, Float, Categorical, Normal, ConfigurationSpace
from dehb import DEHB, DE
from run_analyzer import analyze_run


def create_search_space_rf():
    """
    Parameter space to be optimized --- contains the hyperparameters
    """
    feature_fit = Categorical("feature_fit", ("all", "val", "train"))
    rotation = Categorical("rotation", ("varimax", "quartimax", None))
    fa_components = Integer("fa_components", bounds=(3, 100), log=True)
    iterated_power = Integer("iterated_power", bounds=(3, 7))
    tol = Float("tol", bounds=(0.0001, 0.1), log=True)
    criterion = Categorical("criterion", ("gini", "entropy", "log_loss"))
    max_depth = Integer("max_depth", bounds=(2, 50), log=True)
    min_samples_leaf = Integer("min_samples_leaf", bounds=(1, 1024), log=True)
    max_features = Integer("max_features", bounds=(1, 100), log=True)
    class_weight = Categorical("class_weight", (None, "balanced", "balanced_subsample"))
    min_impurity_decrease = Float("min_impurity_decrease", bounds=(0.00001, 0.5), log=True)
    ccp_alpha = Float("ccp_alpha", bounds=(0.000001, 0.5), log=True)
    cs = ConfigurationSpace(seed=123)
    cs.add([feature_fit, rotation, fa_components, iterated_power, tol, criterion, max_depth, min_samples_leaf, max_features,
            class_weight, min_impurity_decrease, ccp_alpha])
    return cs



full_data = [load_data_single_channel(i) for i in range(4)]
groups = np.concatenate([np.ones_like(y) * i for i, (_, y) in enumerate(full_data)])
all_features = np.vstack([x[:, 0, :] for x, _ in full_data])
all_labels = np.concatenate([y for _, y in full_data])


ROOT_PATH = Path("../../data/train")
def target_function(config, fidelity):
    model = RandomForestClassifier(n_estimators=50, criterion=config["criterion"],
                                 max_depth=config["max_depth"], min_samples_leaf=config["min_samples_leaf"],
                                 max_features=config["max_features"], n_jobs=-1,
                                 class_weight=config["class_weight"],
                                 min_impurity_decrease=config["min_impurity_decrease"],
                                 ccp_alpha=config["ccp_alpha"])
    start_time = time.time()

    try:
        assert config["max_features"] <= config["fa_components"]
        logo = LeaveOneGroupOut()
        cohens_kappas = []
        all_predictions = []
        for i, (train_index, test_index) in enumerate(logo.split(all_features, all_labels, groups)):
            train_features, train_labels = all_features[train_index], all_labels[train_index]
            val_features, val_labels = all_features[test_index], all_labels[test_index]
            if config["feature_fit"] != "all" or i == 0:
                ica = FactorAnalysis(n_components=config["fa_components"], rotation=config["rotation"],
                                     iterated_power=config["iterated_power"], tol=config["tol"], max_iter=int(fidelity)//3)
                if config["feature_fit"] == "all":
                    ica.fit(all_features)
                elif config["feature_fit"] == "val":
                    ica.fit(val_features)
                elif config["feature_fit"] == "train":
                    ica.fit(train_features)
            train_features = ica.transform(train_features)
            val_features = ica.transform(val_features)
            model.fit(train_features, train_labels)
            preds = model.predict(val_features)
            cohens_kappas.append(cohen_kappa_score(preds, val_labels))
            all_predictions.append(preds)
        total_preds = np.concatenate(all_predictions)
        cohens_kappas.append(cohen_kappa_score(total_preds, all_labels))
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
    except:
        end_time = time.time()
        cost = end_time - start_time
        print(f"Run failed with cost {cost}")
        result = {
            "fitness": 1,
            "cost": cost
        }
        return result


cs = create_search_space_rf()
dimensions = len(list(cs.values()))
dehb = DEHB(
    f=target_function,
    cs=cs,
    dimensions=dimensions,
    min_fidelity=110,
    max_fidelity=990,
    n_workers=1,
    output_path="./fa",
    resume=True
)


    #for element in hist:
    #    config_vector = np.array(element[1])
    #    config_as_dict = dehb.vector_to_configspace(config_vector)
    #    print(config_as_dict)
analyze_run(dehb, use_top_fidelities_only=True)
analyze_run(dehb, use_top_fidelities_only=False)

#trajectory, runtime, history = dehb.run(
#    total_cost=7 * 3600,
#    # parameters expected as **kwargs in target_function is passed here
#)
"""
a = load_data_as_windows(3)[0]
features, labels = load_data_single_channel(3)
val_features, val_labels = load_data_single_channel(2)
features = features[:, 0, :]
val_features = val_features[:, 0, :]
total_features = np.vstack((features, val_features))
model = RandomForestClassifier(n_estimators=100, max_depth=6)

print("--------------")

ica = FastICA(n_components=50)
ica.fit(val_features)
features = ica.transform(features)
val_features = ica.transform(val_features)
model.fit(features, labels)
preds = model.predict(features)
print(cohen_kappa_score(preds, labels))
preds = model.predict(val_features)
print(cohen_kappa_score(preds, val_labels))
print("-----------------")
print(a.shape)
X = np.reshape(a, newshape=(a.shape[0]*a.shape[1], a.shape[2]))
X = X.T
#scaler = StandardScaler()
#X = scaler.fit_transform(X)
#print(a.shape)
#ica = FastICA(n_components=50)  # Extracting as many components as there are channels, i.e. 88
#components = ica.fit_transform(a)
#print(components.shape)
"""


"""
# Compute ICA
ica = FastICA(n_components=50)
S_ = ica.fit_transform(X)  # Reconstruct signals
A_ = ica.mixing_  # Get estimated mixing matrix

# We can `prove` that the ICA model applies by reverting the unmixing.
#assert np.allclose(X, np.dot(S_, A_.T) + ica.mean_)

# For comparison, compute PCA
pca = PCA(n_components=30)
H = pca.fit_transform(X)  # Reconstruct signals based on orthogonal components

import matplotlib.pyplot as plt

plt.figure()

models = [X, S_, H]
names = [
    "Observations (mixed signal)",
    "ICA recovered signals",
    "PCA recovered signals",
]
colors = ["red", "steelblue", "orange"]

for ii, (model, name) in enumerate(zip(models, names), 1):
    plt.subplot(3, 1, ii)
    plt.title(name)
    for sig, color in zip(model.T, colors):
        plt.plot(sig, color=color)

plt.tight_layout()
plt.show()
"""