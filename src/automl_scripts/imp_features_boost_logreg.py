from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import LeaveOneGroupOut

import pandas as pd

from src.automl_scripts.run_analyzer import analyze_run
from src.constants import feature_importances, feature_names_order_importance

from ConfigSpace import Float, Categorical, ConfigurationSpace
from dehb import DEHB
import time
import warnings
warnings.filterwarnings(action="ignore", category=ConvergenceWarning)
warnings.filterwarnings(action="ignore", category=UserWarning)

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
    penalty = Categorical("penalty", ("l2", "l1", "elasticnet", None))
    C = Float("C", bounds=(0.001, 100), log=True)
    class_weight = Categorical("class_weight", ("balanced", None))
    l1_ratio = Float("l1_ratio", (0.001, 1), log=True)
    feature_load_cutoff = Float("feature_load_cutoff", bounds=(0.00001, 0.01), log=True)
    solver = Categorical("solver", ("lbfgs", "saga"))
    cs = ConfigurationSpace(seed=123)
    cs.add([penalty, C, class_weight, l1_ratio, feature_load_cutoff, solver])
    return cs


def target_function(config, fidelity):
    start_time = time.time()
    try:
        features = load_features_opt(df_all_features, config["feature_load_cutoff"])
        logo = LeaveOneGroupOut()
        logo.get_n_splits(X=features, y=labels, groups=groups)
        cohens_kappas = []
        all_predictions = []
        for i, (train_index, val_index) in enumerate(logo.split(features, labels, groups)):
            scaler = StandardScaler()
            train_features = features.iloc[train_index]
            train_labels = np.asarray(labels.iloc[train_index]).ravel()
            val_features = features.iloc[val_index]
            val_labels = np.asarray(labels.iloc[val_index]).ravel()
            train_features = scaler.fit_transform(train_features)
            val_features = scaler.transform(val_features)
            if config["penalty"] == "elasticnet" or config["penalty"] == "l1":
                predictor = LogisticRegression(penalty=config["penalty"], solver="saga", C=config["C"],
                                               class_weight=config["class_weight"],
                                               l1_ratio=config["l1_ratio"], max_iter=int(fidelity),
                                               n_jobs=-1)
            elif config["penalty"] is None:
                predictor = LogisticRegression(penalty=config["penalty"], solver=config["solver"],
                                               class_weight=config["class_weight"],
                                               max_iter=int(fidelity),
                                               n_jobs=-1)
            else:
                predictor = LogisticRegression(penalty=config["penalty"], solver=config["solver"],
                                               C=config["C"], class_weight=config["class_weight"],
                                               max_iter=int(fidelity),
                                               n_jobs=-1)

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
    except:
        end_time = time.time()
        cost = end_time - start_time
        print(f"illegal configuration with cost {cost}")
        return {"fitness": 1, "cost": cost}

#df_all_features, labels, groups = load_features_labels_and_groups(0.0001)

cs = create_search_space_rf()
dimensions = len(list(cs.values()))
dehb = DEHB(
    f=target_function,
    cs=cs,
    dimensions=dimensions,
    min_fidelity=36,
    max_fidelity=108*3,
    n_workers=-1,
    output_path="./imp_features_boost_logreg",
    resume=True
)
analyze_run(dehb, use_top_fidelities_only=True)
analyze_run(dehb, use_top_fidelities_only=False)
#trajectory, runtime, history = dehb.run(
#    total_cost=4*3600,
#    # parameters expected as **kwargs in target_function is passed here
#)




