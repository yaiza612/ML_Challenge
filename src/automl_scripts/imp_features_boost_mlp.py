from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import numpy as np
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import LeaveOneGroupOut

import pandas as pd

from src.automl_scripts.run_analyzer import analyze_run
from src.constants import feature_importances, feature_names_order_importance
import time
import ConfigSpace as CS
from ConfigSpace import Integer, Float, Categorical, Normal, ConfigurationSpace
from dehb import DEHB, DE
import time


num_samples_set_0 = 77125
num_samples_set_1 = 52320
num_samples_set_2 = 64215
num_samples_set_3 = 68095
num_samples_set_4 = 66020
num_samples_set_5 = 46595


def load_features(indices, cutoff):
    features_to_use = [feature_name for feature_name, feature_importance in
                       zip(feature_names_order_importance, feature_importances) if feature_importance > cutoff]
    print(len(features_to_use))
    filename_extensions = ["frequency_domain", "time_frequency_domain",
                           "nonlinear",
                           "time_domain", "spatial"]
    path_for_features = "../../data/Engineered_features/"
    normalizing_strings = ["", "min_max_scaled_", "standard_scaled_", "robust_scaled_", "detrend_",
                           "global_min_max_scaled_", "global_standard_scaled_", "global_robust_scaled_",
                           "global_detrend_"]
    all_dfs = []
    for normalizing_string in normalizing_strings:
        print(normalizing_string)
        for filename_ext in filename_extensions:
            temp_df = []
            for i in indices:
                filename = f'{path_for_features}eeg_{normalizing_string}{filename_ext}_features_{i}.csv'
                df = pd.read_csv(filename)
                column_names = df.columns
                overlapping_names = list(set(column_names) & set(features_to_use))
                short_df = df[overlapping_names]
                temp_df.append(short_df)
            temp_df = pd.concat(temp_df)
            all_dfs.append(temp_df)
    final_df = pd.concat(all_dfs, axis=1)
    return final_df

def load_features_opt(total_df, cutoff):
    features_to_use = [feature_name for feature_name, feature_importance in
                       zip(feature_names_order_importance, feature_importances) if feature_importance > cutoff]

    column_names = total_df.columns
    overlapping_names = list(set(column_names) & set(features_to_use))
    shortened_df = total_df[overlapping_names]
    return shortened_df



def load_features_labels_and_groups(feat_cutoff):
    path_for_features = "../../data/Engineered_features/"
    train_features = load_features(range(4), feat_cutoff)

    labels = pd.concat([pd.read_csv(f'{path_for_features}eeg_label_features_{i}.csv') for i in range(4)], axis=0)

    train_groups = np.concatenate(
        [[_] * num_samples for _, num_samples in enumerate([num_samples_set_0, num_samples_set_1,
                                                            num_samples_set_2, num_samples_set_3])])
    return train_features, labels, train_groups


def create_search_space_rf():
    """
    Parameter space to be optimized --- contains the hyperparameters
    """
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
    feature_load_cutoff = Float("feature_load_cutoff", bounds=(0.00001, 0.01), log=True)
    cs = ConfigurationSpace(seed=123)
    cs.add([hidden_layer_2, hidden_layer_3, hidden_layer_4,
            activation, solver, alpha, batch_size, learning_rate, learning_rate_init,
            momentum, beta_1, beta_2, feature_load_cutoff])
    return cs


def target_function(config, fidelity):
    print(config)
    features = load_features_opt(df_all_features, config["feature_load_cutoff"])
    logo = LeaveOneGroupOut()
    logo.get_n_splits(X=features, y=labels, groups=groups)
    cohens_kappas = []
    all_predictions = []
    start_time = time.time()
    for i, (train_index, val_index) in enumerate(logo.split(features, labels, groups)):
        scaler = StandardScaler()
        train_features = features.iloc[train_index]
        train_labels = np.asarray(labels.iloc[train_index]).ravel()
        val_features = features.iloc[val_index]
        val_labels = np.asarray(labels.iloc[val_index]).ravel()
        train_features = scaler.fit_transform(train_features)
        val_features = scaler.transform(val_features)
        predictor = MLPClassifier(hidden_layer_sizes=(config["hidden_layer_2"],
                                                      config["hidden_layer_3"], config["hidden_layer_4"]),
                                  activation=config["activation"], solver=config["solver"],
                                  alpha=config["alpha"], batch_size=config["batch_size"],
                                  learning_rate=config["learning_rate"],
                                  learning_rate_init=config["learning_rate_init"], momentum=config["momentum"],
                                  beta_1=config["beta_1"], beta_2=config["beta_2"],
                                  early_stopping=True, max_iter=int(fidelity), max_fun=int(fidelity) * 75)
        predictor.fit(train_features, train_labels)
        preds = predictor.predict(val_features)
        cohens_kappas.append(cohen_kappa_score(preds, val_labels))
        all_predictions.append(preds)
    total_preds = np.concatenate(all_predictions)
    cohens_kappas.append(cohen_kappa_score(total_preds, labels))
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

#df_all_features, labels, groups = load_features_labels_and_groups(0.0001)

cs = create_search_space_rf()
dimensions = len(list(cs.values()))
dehb = DEHB(
    f=target_function,
    cs=cs,
    dimensions=dimensions,
    min_fidelity=4,
    max_fidelity=36,
    n_workers=-1,
    output_path="./imp_features_boost_mlp",
    resume=True
)
analyze_run(dehb, use_top_fidelities_only=True)
analyze_run(dehb, use_top_fidelities_only=False)
#trajectory, runtime, history = dehb.run(
#    total_cost=7*3600,
#    # parameters expected as **kwargs in target_function is passed here
#)




