import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import sys
import os

from Demos.mmapfile_demo import offset
from sklearn.decomposition import FactorAnalysis, FastICA
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.base import BaseEstimator, TransformerMixin

from src.data_loading.create_submission import create_submission_from_flattened_preds
from src.data_loading.loader import reshape_array_into_windows, load_data_single_channel
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from src.automl_scripts.imp_features_boost import load_features_labels_and_groups
from sklearn.neural_network import MLPClassifier
from pystacknet.pystacknet import StackNetClassifier
from sklearn.preprocessing import StandardScaler
from src.constants import *
from src.automl_scripts.optimize_bandpass_features import butter_bandpass_filter
import time

import warnings
warnings.filterwarnings("ignore")
if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore"

custom_knn_parameters = ['num_neighbors_1', 'num_neighbors_2', 'num_neighbors_3',
                         'p_1', 'p_2', 'p_3', 'weights_1', 'weights_2', 'weights_3']
bandpass_parameters = ['band_1_high', 'band_1_low', 'band_2_high', 'band_2_low', 'band_3_high', 'band_3_low',
                       'order_1', 'order_2', 'order_3', 'use_amplitude_1', 'use_amplitude_2',
                       'use_amplitude_3', 'use_mean_1', 'use_mean_2', 'use_mean_3', 'use_variance_1',
                       'use_variance_2', 'use_variance_3']

feature_importance_parameters = ["feature_load_cutoff"]

random_forest_parameters = ['ccp_alpha', 'class_weight', 'criterion', 'max_depth', 'max_features',
                            'min_impurity_decrease', 'min_samples_leaf', "n_estimators", "n_jobs"]

hist_gradient_boosting_parameters = ['class_weight', 'interaction_cst', 'l2_regularization',
                                     'learning_rate', 'max_depth', 'max_features',
                                     'max_leaf_nodes', 'min_samples_leaf', 'max_iter']

logistic_regression_parameters = ['C', 'class_weight', 'l1_ratio', 'penalty', 'solver', 'max_iter',
                                  'n_jobs']

mlp_parameters = ['activation', 'alpha', 'batch_size', 'beta_1', 'beta_2',
                  'hidden_layer_2', 'hidden_layer_3', 'hidden_layer_4',
                  'learning_rate', 'learning_rate_init', 'momentum',
                  'solver', 'max_iter', 'max_fun']


def fa_feature_loading_function(val_indices, train_indices, config):
    fa = FactorAnalysis(n_components=config["fa_components"], rotation=config["rotation"],
                        iterated_power=config["iterated_power"], tol=config["tol"], max_iter=config["max_iter"])
    if config["feature_fit"] == "train":
        fa.fit(features_total_for_fa[train_indices[:, 0]])
    elif config["feature_fit"] == "val":
        fa.fit(features_total_for_fa[val_indices[:, 0]])
    return (fa.transform(features_total_for_fa[train_indices[:, 0]]),
            fa.transform(features_total_for_fa[val_indices[:, 0]]))

def ica_feature_loading_function(val_indices, train_indices, config):
    ica = FastICA(n_components=config["ica_components"], whiten=config["ica_whiten"], fun=config["fun"],
                  max_iter=config["max_iter"])
    if config["feature_fit"] == "train":
        ica.fit(features_total_for_fa[train_indices[:, 0]])
    elif config["feature_fit"] == "val":
        ica.fit(features_total_for_fa[val_indices[:, 0]])
    return (ica.transform(features_total_for_fa[train_indices[:, 0]]),
            ica.transform(features_total_for_fa[val_indices[:, 0]]))


def get_one_set_of_bandpass_features(low, high, order, indices):
    if not os.path.isfile(f"temp_bandpass_features_dataset_4/low_{low}_high_{high}_order_{order}_amplitude.npy"):
        all_data = []
        for data in features_total_for_bandpass:
            filtered_data = butter_bandpass_filter(data, low, high, 250, order)
            reshaped_data = reshape_array_into_windows(filtered_data, 250, 2)
            reshaped_data = np.reshape(reshaped_data, newshape=(reshaped_data.shape[0] * reshaped_data.shape[1], 1,
                                                                reshaped_data.shape[2]), order="F")
            reshaped_data = reshaped_data.reshape((-1, reshaped_data.shape[-1]))
            all_data.append(reshaped_data)
        all_data = np.concatenate(all_data)
        amplitude = (np.max(all_data, -1) - np.min(all_data, -1)).reshape(-1)
        variance = np.std(all_data, -1).reshape(-1)
        mean = np.mean(all_data, -1).reshape(-1)
        np.save(f"temp_bandpass_features_dataset_4/low_{low}_high_{high}_order_{order}_amplitude.npy", amplitude)
        np.save(f"temp_bandpass_features_dataset_4/low_{low}_high_{high}_order_{order}_variance.npy", variance)
        np.save(f"temp_bandpass_features_dataset_4/low_{low}_high_{high}_order_{order}_mean.npy", mean)
    else:
        amplitude = np.load(f"temp_bandpass_features_dataset_4/low_{low}_high_{high}_order_{order}_amplitude.npy")
        variance = np.load(f"temp_bandpass_features_dataset_4/low_{low}_high_{high}_order_{order}_variance.npy")
        mean = np.load(f"temp_bandpass_features_dataset_4/low_{low}_high_{high}_order_{order}_mean.npy")
    return amplitude[indices[:, 0]], mean[indices[:, 0]], variance[indices[:, 0]]


def bandpass_feature_loading_function_knn(indices, config):
    lows = [config["band_1_low"], config["band_2_low"], config["band_3_low"]]
    highs = [config["band_1_high"], config["band_2_high"], config["band_3_high"]]
    orders = [config["order_1"], config["order_2"], config["order_3"]]
    use_amplitudes = [config["use_amplitude_1"], config["use_amplitude_2"], config["use_amplitude_3"]]
    use_means = [config["use_mean_1"], config["use_mean_2"], config["use_mean_3"]]
    use_variances = [config["use_variance_1"], config["use_variance_2"], config["use_variance_3"]]
    feature_list = []
    for low, high, order, use_amplitude, use_mean, use_variance in zip(lows, highs, orders,
                                                                       use_amplitudes, use_means, use_variances):
        amplitude, mean, variance = get_one_set_of_bandpass_features(low, high, order, indices)
        features_in_list = []
        if use_amplitude:
            features_in_list.append(amplitude)
        if use_mean:
            features_in_list.append(mean)
        if use_variance:
            features_in_list.append(variance)
        all_features = np.vstack(features_in_list).T
        feature_list.append(all_features)
    return feature_list


def bandpass_feature_loading_function(indices, config):
    lows = [config["band_1_low"], config["band_2_low"], config["band_3_low"]]
    highs = [config["band_1_high"], config["band_2_high"], config["band_3_high"]]
    orders = [config["order_1"], config["order_2"], config["order_3"]]
    feature_list = []
    for low, high, order in zip(lows, highs, orders):
        amplitude, mean, variance = get_one_set_of_bandpass_features(low, high, order, indices)
        feature_list.extend([amplitude, mean, variance])
    all_features = np.vstack(feature_list).T
    return all_features


def example_feature_loading_function(indices):
    all_dfs = []
    for i in range(6):
        filename = f'../../data/Engineered_features/eeg_time_domain_features_{i}.csv'
        df = pd.read_csv(filename)
        all_dfs.append(df)
    final_df = pd.concat(all_dfs, axis=0)
    return final_df.iloc[indices[:, 0]]


def feature_importance_loading_function(indices, config):
    features_to_use = [feature_name for feature_name, feature_importance in
                       zip(feature_names_order_importance, feature_importances)
                       if feature_importance > config["feature_load_cutoff"]]

    column_names = features_total_for_imp.columns
    overlapping_names = list(set(column_names) & set(features_to_use))
    shortened_df = features_total_for_imp[overlapping_names]
    return shortened_df.iloc[indices[:, 0]]


class CustomKNN(BaseEstimator):
    """
    this class splits the collected means, variances and amplitudes into the bandpass-buckets for classifying them
    separately and merging the predictions
    """
    def __init__(self, config, model_parameters, feature_loading_function_parameters,
                 feature_loading_function, use_scaler, print_name):
        super(CustomKNN, self).__init__()
        self.config = config
        self.model_parameters = model_parameters
        self.feature_loading_function_parameters = feature_loading_function_parameters
        self.filtered_config_for_feature_loading_function = {k: v for k, v in self.config.items()
                                                             if k in self.feature_loading_function_parameters}
        self.feature_loading_function = feature_loading_function
        self.scaler = StandardScaler()
        self.use_scaler = use_scaler
        self.scalers = [StandardScaler(), StandardScaler(), StandardScaler()]
        self.knns = []
        self.time = None
        self.print_name = print_name

    def fit(self, X, y):
        _, ind = np.unique(X, return_index=True)
        X = X[ind]
        y = y[ind]
        self.time = time.time()
        features = self.feature_loading_function(X, self.filtered_config_for_feature_loading_function)
        if self.use_scaler:
            features = [scaler.fit_transform(f) for scaler, f in zip(self.scalers, features)]
        self.knns = [KNeighborsClassifier(n_neighbors=self.config[f"num_neighbors_{_ + 1}"],
                                          weights=self.config[f"weights_{_ + 1}"],
                                          p=self.config[f"p_{_ + 1}"], n_jobs=-1) for _ in range(3)]
        self.knns = [knn.fit(part, y) for part, knn in zip(features, self.knns)]
        print(f"took {time.time() - self.time} for training of model {self.print_name}")

    def predict_proba(self, X):
        self.time = time.time()
        features = self.feature_loading_function(X, self.filtered_config_for_feature_loading_function)
        if self.use_scaler:
            features = [scaler.transform(f) for scaler, f in zip(self.scalers, features)]
        preds = [knn.predict_proba(part) for part, knn in zip(features, self.knns)]
        new_stacked_predictions = np.stack([p for p in preds])
        summed_predictions = np.sum(new_stacked_predictions, axis=0) / 3
        print(f"took {time.time() - self.time} for prediction of model {self.print_name}")
        return summed_predictions

"""
class AutoGluonLoader(BaseEstimator):
    def __init__(self):
        super(AutoGluonLoader, self).__init__()

    def fit(self, X, y):
        pass

    def predict_proba(self, X):

        def detect_fold(indices):
            possible_folds = []
            if 1 not in indices:
                possible_folds.append(0)
            if (1 + num_samples_set_0) not in indices:
                possible_folds.append(1)
            if (1 + num_samples_set_0 + num_samples_set_1) not in indices:
                possible_folds.append(2)
            if (1 + num_samples_set_0 + num_samples_set_1 + num_samples_set_2) not in indices:
                possible_folds.append(3)
            return possible_folds

        folds = detect_fold(X)
        if len(folds) != 1:
            print("uh-oh")

        all_predictions_val = [np.load(f"../automl_scripts/autogluon_fold_{folds[0]}_val_data.npy")
                           for i in range(4)]
        all_predictions_test_0 = [np.load(f"../automl_scripts/autogluon_fold_{folds[0]}_test_data_0.npy")
                                  for i in range(4)]
"""

class CustomModelAnalysis(BaseEstimator):
    def __init__(self, model_type, config, model_parameters, feature_loading_function, print_name):
        super(CustomModelAnalysis, self).__init__()
        self.model_type = model_type
        self.config = config
        self.model_parameters = model_parameters
        self.filtered_config_dict_for_model = {k: v for k, v in self.config.items() if k in self.model_parameters}
        self.model = model_type(**self.filtered_config_dict_for_model)
        self.feature_loading_function = feature_loading_function
        self.train_features = None
        self.train_labels = None
        self.time = None
        self.print_name = print_name
        self.analysis_model = None

    def fit(self, X, y):
        _, ind = np.unique(X, return_index=True)
        X = X[ind]
        y = y[ind]
        self.time = time.time()
        self.train_features = X
        self.train_labels = y
        print(f"took {time.time() - self.time} for training of model {self.print_name}")

    def predict_proba(self, X):
        self.time = time.time()
        train_features, val_features = self.feature_loading_function(X, self.train_features, self.config)

        self.model.fit(train_features, self.train_labels)
        prediction = self.model.predict_proba(val_features)
        print(f"took {time.time() - self.time} for prediction of model {self.print_name}")
        return prediction



class CustomChannelModel(TransformerMixin, BaseEstimator):
    def __init__(self):
        super(CustomChannelModel, self).__init__()

    def fit(self, X, y):
        pass

    def transform(self, X):
        modded = np.mod(X, 5)
        z = np.eye(5)[modded[:, 0]]
        return z


class CustomDataSetModel(TransformerMixin, BaseEstimator):
    def __init__(self):
        super(CustomDataSetModel, self).__init__()

    def fit(self, X, y):
        pass

    def transform(self, X):
        res = np.where(X[:, 0] > np.sum([num_samples_set_0, num_samples_set_1,
                                               num_samples_set_2, num_samples_set_3]), 0, 1)
        z = np.eye(2)[res]
        return z


class CustomMLPModel(BaseEstimator):
    def __init__(self, config, feature_loading_function_parameters,
                 feature_loading_function, use_scaler, print_name):
        super(CustomMLPModel, self).__init__()
        self.config = config
        self.feature_loading_function_parameters = feature_loading_function_parameters
        self.filtered_config_for_feature_loading_function = {k: v for k, v in self.config.items()
                                                             if k in self.feature_loading_function_parameters}
        self.model = MLPClassifier(hidden_layer_sizes=(config["hidden_layer_2"],
                                                      config["hidden_layer_3"], config["hidden_layer_4"]),
                                   activation=config["activation"], solver=config["solver"],
                                   alpha=config["alpha"], batch_size=config["batch_size"],
                                   learning_rate=config["learning_rate"],
                                   learning_rate_init=config["learning_rate_init"], momentum=config["momentum"],
                                   beta_1=config["beta_1"], beta_2=config["beta_2"],
                                   early_stopping=True, max_iter=config["max_iter"],
                                   max_fun=config["max_iter"])
        self.feature_loading_function = feature_loading_function
        self.scaler = StandardScaler()
        self.use_scaler = use_scaler
        self.time = None
        self.print_name = print_name

    def fit(self, X, y):
        _, ind = np.unique(X, return_index=True)
        X = X[ind]
        y = y[ind]
        self.time = time.time()
        features = self.feature_loading_function(X, self.filtered_config_for_feature_loading_function)
        if self.use_scaler:
            features = self.scaler.fit_transform(features)
        self.model.fit(features, y)
        print(f"took {time.time() - self.time} for training of model {self.print_name}")

    def predict_proba(self, X):
        self.time = time.time()
        features = self.feature_loading_function(X, self.filtered_config_for_feature_loading_function)
        if self.use_scaler:
            features = self.scaler.transform(features)
        prediction = self.model.predict_proba(features)
        print(f"took {time.time() - self.time} for prediction of model {self.print_name}")
        return prediction


class CustomModel(BaseEstimator):
    def __init__(self, model_type, config, model_parameters, feature_loading_function_parameters,
                 feature_loading_function, use_scaler, print_name):
        super(CustomModel, self).__init__()
        self.model_type = model_type
        self.config = config
        self.model_parameters = model_parameters
        self.feature_loading_function_parameters = feature_loading_function_parameters
        self.filtered_config_dict_for_model = {k: v for k, v in self.config.items() if k in self.model_parameters}
        self.filtered_config_for_feature_loading_function = {k: v for k, v in self.config.items()
                                                             if k in self.feature_loading_function_parameters}
        self.model = model_type(**self.filtered_config_dict_for_model)
        self.feature_loading_function = feature_loading_function
        self.scaler = StandardScaler()
        self.use_scaler = use_scaler
        self.time = None
        self.print_name = print_name

    def fit(self, X, y):
        _, ind = np.unique(X, return_index=True)
        X = X[ind]
        y = y[ind]
        self.time = time.time()
        features = self.feature_loading_function(X, self.filtered_config_for_feature_loading_function)
        if self.use_scaler:
            features = self.scaler.fit_transform(features)
        self.model.fit(features, y)
        print(f"took {time.time() - self.time} for training of model {self.print_name}")


    def predict_proba(self, X):
        self.time = time.time()
        features = self.feature_loading_function(X, self.filtered_config_for_feature_loading_function)
        if self.use_scaler:
            features = self.scaler.transform(features)
        prediction = self.model.predict_proba(features)
        print(f"took {time.time() - self.time} for prediction of model {self.print_name}")
        return prediction

def custom_cohen_kappa_score(y_true, y_pred, sample_weight=None):
    y_pred = np.where(y_pred > 0.5, 1, 0)
    return cohen_kappa_score(y_true, y_pred, sample_weight=sample_weight)


def save_stacknet(stacknet, iteration, test_index, save_name):
    for layer_idx, layer in enumerate(stacknet.models):
        if layer_idx == 0:
            for model in layer:
                if isinstance(model, CustomKNN):
                    for knn_index in range(3):
                        with open(f"stacknet_models/stack_test_model_{model.print_name}_part_{knn_index}_"
                                  f"{test_index}_{save_name}_iteration_{iteration}.pkl", "wb") as f:
                            pickle.dump(model.knns[knn_index], f)
                        if model.use_scaler:
                            with open(f"stacknet_models/stack_test_model_{model.print_name}_part_{knn_index}_"
                                      f"{test_index}_{save_name}_iteration_{iteration}_scaler.pkl", "wb") as f:
                                pickle.dump(model.scalers[knn_index], f)
                elif isinstance(model, CustomModelAnalysis):
                    with open(f"stacknet_models/stack_test_model_{model.print_name}_{test_index}_"
                              f"{save_name}_iteration_{iteration}.pkl", "wb") as f:
                        pickle.dump(model.model, f)
                    np.save(f"stacknet_models/stack_test_model_{model.print_name}_{test_index}_"
                              f"{save_name}_iteration_{iteration}_features.npy", model.train_features)
                    np.save(f"stacknet_models/stack_test_model_{model.print_name}_{test_index}_"
                            f"{save_name}_iteration_{iteration}_labels.npy", model.train_labels)
                else:
                    with open(f"stacknet_models/stack_test_model_{model.print_name}_{test_index}_"
                              f"{save_name}_iteration_{iteration}.pkl", "wb") as f:
                        pickle.dump(model.model, f)
                    if model.use_scaler:
                        with open(f"stacknet_models/stack_test_model_{model.print_name}_{test_index}_"
                                  f"{save_name}_iteration_{iteration}_scaler.pkl", "wb") as f:
                            pickle.dump(model.scaler, f)
        elif layer_idx > 0:
            for model_idx, model in enumerate(layer):
                with open(f"stacknet_models/stack_test_model_{model_idx}_layer{layer_idx}_{test_index}_"
                     f"{save_name}_iteration_{iteration}.pkl", "wb") as f:
                    pickle.dump(model, f)


def reconstruct_stacknet(stacknet, iteration, test_index, save_name):
    for layer_idx, layer in enumerate(stacknet.models):
        if layer_idx == 0:
            for model in layer:
                if isinstance(model, CustomKNN):
                    for knn_index in range(3):
                        with open(f"stacknet_models/stack_test_model_{model.print_name}_part_{knn_index}_"
                                  f"{test_index}_{save_name}_iteration_{iteration}.pkl", "rb") as f:
                            temp_clf = pickle.load(f)
                            model.knns[knn_index] = temp_clf
                        if model.use_scaler:
                            with open(f"stacknet_models/stack_test_model_{model.print_name}_part_{knn_index}_"
                                      f"{test_index}_{save_name}_iteration_{iteration}_scaler.pkl", "rb") as f:
                                temp_scaler = pickle.load(f)
                                model.scalers[knn_index] = temp_scaler
                elif isinstance(model, CustomModelAnalysis):
                    with open(f"stacknet_models/stack_test_model_{model.print_name}_{test_index}_"
                              f"{save_name}_iteration_{iteration}.pkl", "rb") as f:
                        temp_clf = pickle.load(f)
                        model.model = temp_clf
                    temp_train_features = np.load(f"stacknet_models/stack_test_model_{model.print_name}_{test_index}_"
                                                  f"{save_name}_iteration_{iteration}_features.npy")
                    temp_train_labels = np.load(f"stacknet_models/stack_test_model_{model.print_name}_{test_index}_"
                                                f"{save_name}_iteration_{iteration}_labels.npy")
                    model.train_features = temp_train_features
                    model.train_labels = temp_train_labels
                else:
                    with open(f"stacknet_models/stack_test_model_{model.print_name}_{test_index}_"
                              f"{save_name}_iteration_{iteration}.pkl", "rb") as f:
                        temp_clf = pickle.load(f)
                        model.model = temp_clf
                    if model.use_scaler:
                        with open(f"stacknet_models/stack_test_model_{model.print_name}_{test_index}_"
                                  f"{save_name}_iteration_{iteration}_scaler.pkl", "rb") as f:
                            temp_scaler = pickle.load(f)
                            model.scaler = temp_scaler
        elif layer_idx > 0:
            new_model_list = []
            for model_idx, model in enumerate(layer):
                with open(f"stacknet_models/stack_test_model_{model_idx}_layer{layer_idx}_{test_index}_"
                          f"{save_name}_iteration_{iteration}.pkl", "rb") as f:
                    temp_clf = pickle.load(f)
                    new_model_list.append(temp_clf)
            stacknet.models[layer_idx] = new_model_list


def create_additional_features_labels_groups(prediction, test_features, offset_test_features=None,
                                             offset_predictions=None):
    max_index = len(offset_test_features)
    predictions = prediction[:, 0]
    additional_class_0_indices = (predictions > 0.96766).nonzero()
    additional_class_1_indices = (predictions < 0.03333).nonzero()

    if offset_test_features is not None:
        offset_indices_class_0 = []
        offset_indices_class_1 = []
        for index in additional_class_0_indices[0]:
            if index + 5 in additional_class_0_indices[0] and index < max_index:
                offset_indices_class_0.append(index)
        for index in additional_class_1_indices[0]:
            if index + 5 in additional_class_1_indices[0] and index < max_index:
                offset_indices_class_1.append(index)
    offset_indices_class_0 = np.array(offset_indices_class_0)
    offset_indices_class_1 = np.array(offset_indices_class_1)
    if offset_predictions is not None:
        offset_predictions = offset_predictions[:, 0]
        offset_class_0_indices_from_offset_predictions = (offset_predictions > 0.96666).nonzero()
        offset_class_1_indices_from_offset_predictions = (offset_predictions < 0.03333).nonzero()

        # make sure that the predictions are congruent
        set_offset_indices_class_0 = set(list(offset_indices_class_0))
        set_offset_indices_class_1 = set(list(offset_indices_class_1))
        set_offset_class_0_indices_from_offset_predictions = set(list(offset_class_0_indices_from_offset_predictions[0]))
        set_offset_class_1_indices_from_offset_predictions = set(list(offset_class_1_indices_from_offset_predictions[0]))
        total_set_class_0_offset = ((set_offset_indices_class_0 | set_offset_class_0_indices_from_offset_predictions) -
                                    (set_offset_indices_class_1 & set_offset_class_1_indices_from_offset_predictions))
        total_set_class_1_offset = ((set_offset_indices_class_1 | set_offset_class_1_indices_from_offset_predictions) -
                                    (set_offset_indices_class_0 & set_offset_class_0_indices_from_offset_predictions))
        offset_indices_class_0 = np.array(list(total_set_class_0_offset))
        offset_indices_class_1 = np.array(list(total_set_class_1_offset))

    additional_features_class_0 = np.concatenate([test_features[additional_class_0_indices], offset_test_features[offset_indices_class_0]])
    additional_features_class_1 = np.concatenate([test_features[additional_class_1_indices], offset_test_features[offset_indices_class_1]])
    additional_labels_class_0 = [0] * additional_features_class_0.shape[0]
    additional_labels_class_1 = [1] * additional_features_class_1.shape[0]
    total_added = len(additional_labels_class_1) + len(additional_labels_class_0)
    print(additional_features_class_0.shape)
    print(additional_features_class_1.shape)

    additional_features = np.vstack((additional_features_class_0, additional_features_class_1,
                                     additional_features_class_0, additional_features_class_1,
                                     additional_features_class_0, additional_features_class_1,
                                     additional_features_class_0, additional_features_class_1))
    additional_labels = np.hstack((additional_labels_class_0, additional_labels_class_1,
                                   additional_labels_class_0, additional_labels_class_1,
                                   additional_labels_class_0, additional_labels_class_1,
                                   additional_labels_class_0, additional_labels_class_1))
    additional_groups = np.hstack([np.ones(shape=total_added) * i for i in range(4)])
    return additional_features, additional_labels, additional_groups

def make_test_predictions(save_name="final_predictions", restacking=False, domain_adaptation=False,
                          additional_features=None, additional_labels=None, additional_groups=None,
                          starting_iteration=0):
    for test_index in range(5, 6):
        train_iteration = starting_iteration
        print(f"training for test dataset {test_index}")
        test_features = [test_features_1_indices, test_features_2_indices][test_index - 4]
        flag = True
        total_added = 0
        while flag:
            train_iteration += 1
            flag = False
            if additional_features is not None:
                train_features_internal = np.vstack((train_features_indices, additional_features))
                train_labels_internal = np.hstack((targets, additional_labels))
                train_groups_internal = np.hstack((groups, additional_groups))
            else:
                train_features_internal = train_features_indices
                train_labels_internal = targets
                train_groups_internal = groups
            logo = LeaveOneGroupOut()
            logo.get_n_splits(X=train_features_internal, y=train_labels_internal, groups=train_groups_internal)
            clf = StackNetClassifier(models, metric=custom_cohen_kappa_score,
                                     folds=logo.split(train_features_internal, train_labels_internal,
                                                      train_groups_internal),
                                     restacking=restacking, use_retraining=True,
                                     use_proba=True, random_state=12345, n_jobs=1, verbose=1)
            clf.fit(train_features_internal, train_labels_internal)
            total_predictions = clf.predict_proba(test_features)
            np.save(f"stack_test_preds_{test_index}_{save_name}_iteration_{train_iteration}.npy",
                    total_predictions)
            offset_predictions = clf.predict_proba(offset_test_features_1_indices)
            np.save(f"stack_test_preds_{test_index}_{save_name}_iteration_{train_iteration}_offset.npy",
                    offset_predictions)
            predictions = total_predictions[:, 0]
            additional_class_0_indices = (predictions > 0.96666).nonzero()
            additional_class_1_indices = (predictions < 0.03333).nonzero()
            print(len(additional_class_0_indices[0]) + len(additional_class_1_indices[0]))
            if len(additional_class_0_indices[0]) + len(additional_class_1_indices[0]) > (total_added + 1000) and domain_adaptation:
                total_added = len(additional_class_0_indices[0]) + len(additional_class_1_indices[0])
                additional_features_class_0 = test_features[additional_class_0_indices]
                additional_features_class_1 = test_features[additional_class_1_indices]
                additional_labels_class_0 = [0] * len(additional_class_0_indices[0])
                additional_labels_class_1 = [1] * len(additional_class_1_indices[0])
                print(additional_features_class_0.shape)
                print(additional_features_class_1.shape)

                additional_features = np.vstack((additional_features_class_0, additional_features_class_1,
                                                 additional_features_class_0, additional_features_class_1,
                                                 additional_features_class_0, additional_features_class_1,
                                                 additional_features_class_0, additional_features_class_1))
                additional_labels = np.hstack((additional_labels_class_0, additional_labels_class_1,
                                               additional_labels_class_0, additional_labels_class_1,
                                               additional_labels_class_0, additional_labels_class_1,
                                               additional_labels_class_0, additional_labels_class_1))
                additional_groups = np.hstack([np.ones(shape=total_added) * i for i in range(4)])
                flag = True


def analyze_test_predictions(save_name):
    test_preds = [np.load(f"stack_test_preds_{i}_{save_name}.npy") for i in range(4, 6)]
    final_predictions = []
    cutoff_1, cutoff_2 = 0.5, 0.5
    for test_pred, cutoff in zip(test_preds, [cutoff_1, cutoff_2]):
        prediction = test_pred[:, 0]
        final_predictions.append(np.where(prediction > cutoff, 0, 1))
    final_prediction = np.concatenate(final_predictions)
    final_prediction_first_set = final_prediction[:13204 * 5]
    final_prediction_second_set = final_prediction[13204 * 5:]

    def reorder_array(arr, num_columns):
        # Compute the number of rows needed
        num_rows = len(arr) // num_columns
        # Reshape the array
        reshaped = arr.reshape(num_columns, num_rows)
        # Flatten column-wise
        return reshaped.ravel(order='F')

    final_prediction_first_set = reorder_array(final_prediction_first_set, 5)
    final_prediction_second_set = reorder_array(final_prediction_second_set, 5)
    create_submission_from_flattened_preds(final_prediction_first_set, final_prediction_second_set)


if __name__ == "__main__":
    ROOT_PATH = Path("../../data/train")
    TEST_PATH = Path("../../data/test")

    timeseries_training_for_bandpass = [np.array(np.load(ROOT_PATH / f"data_{i}.npy") / 2 ** 8, dtype=np.float16)
                                        for i in range(4)]
    timeseries_testing_for_bandpass = [np.array(np.load(TEST_PATH / f"data_{i}.npy") / 2 ** 8, dtype=np.float16)
                                       for i in range(4, 6)]
    offset_timeseries_for_bandpass = [np.array(np.load(TEST_PATH / "data_4.npy") / 2 ** 8, dtype=np.float16)[:, 250:]]
    features_total_for_bandpass = timeseries_training_for_bandpass + timeseries_testing_for_bandpass + offset_timeseries_for_bandpass

    targets, groups = load_features_labels_and_groups(0.0001)
    targets = np.asarray(targets).ravel()
    features_total_for_imp_no_offset = pd.read_csv("important_features.csv")
    features_total_for_imp_offset = pd.read_csv("important_features_offset_4.csv")
    features_total_for_imp = pd.concat([features_total_for_imp_no_offset, features_total_for_imp_offset], axis=0)
    del features_total_for_imp_offset
    del features_total_for_imp_no_offset


    full_data = [load_data_single_channel(i) for i in range(6)]
    features_total_for_fa = np.vstack([x[:, 0, :] for x, _ in full_data])
    del full_data
    del timeseries_testing_for_bandpass
    del timeseries_training_for_bandpass
    del features_total_for_fa

    #models_1 = [CustomModelAnalysis(RandomForestClassifier, config=fa_configs[i],
    #                               model_parameters=random_forest_parameters,
    #                               feature_loading_function=fa_feature_loading_function,
    #                                print_name=f"fa_rf_{i}") for i in range(2)]
    #models_2 = [CustomModelAnalysis(RandomForestClassifier, config=ica_configs[i],
    #                               model_parameters=random_forest_parameters,
    #                               feature_loading_function=ica_feature_loading_function,
    #                                print_name=f"ica_rf_{i}") for i in range(6)]
    models_3 = [CustomModel(model_type=HistGradientBoostingClassifier, config=imp_features_boost_configs[i],
                           model_parameters=hist_gradient_boosting_parameters,
                           feature_loading_function_parameters=feature_importance_parameters,
                           feature_loading_function=feature_importance_loading_function, use_scaler=False,
                           print_name=f"imp_grad_boost_{i}") for i in range(8)]
    models_4 = [CustomModel(model_type=LogisticRegression, config=imp_features_boost_logreg_configs[i],
                           model_parameters=logistic_regression_parameters,
                           feature_loading_function_parameters=feature_importance_parameters,
                           feature_loading_function=feature_importance_loading_function, use_scaler=True,
                           print_name=f"imp_log_reg_{i}") for i in range(5)]
    models_5 = [CustomMLPModel(config=imp_features_boost_mlp_configs[i],
                              feature_loading_function_parameters=feature_importance_parameters,
                              feature_loading_function=feature_importance_loading_function,
                              use_scaler=True, print_name=f"imp_mlp_{i}") for i in range(8)]
    models_6 = [CustomKNN(config=bandpass_features_knn_configs[i], model_parameters=custom_knn_parameters,
                         feature_loading_function_parameters=bandpass_parameters,
                         feature_loading_function=bandpass_feature_loading_function_knn, use_scaler=True,
                         print_name=f"band_knn_{i}") for i in range(10)]
    models_7 = [CustomMLPModel(config=bandpass_features_mlp_configs[i],
                              feature_loading_function_parameters=bandpass_parameters,
                              feature_loading_function=bandpass_feature_loading_function,
                              use_scaler=True, print_name=f"band_mlp_{i}") for i in range(3)]
    models_8 = [CustomModel(model_type=RandomForestClassifier, config=bandpass_features_rf_configs[i],
                           model_parameters=random_forest_parameters,
                           feature_loading_function_parameters=bandpass_parameters,
                           feature_loading_function=bandpass_feature_loading_function, use_scaler=True,
                           print_name=f"band_rf_{i}") for i in range(9)]
    models_9 = [CustomChannelModel()]
    models_10 = [CustomDataSetModel()]
    first_layer_models = models_3 + models_4 + models_5 + models_6 + models_7 + models_8 + models_9 + models_10

    models = [first_layer_models,
          [RandomForestClassifier(n_estimators=1000, criterion="gini", max_depth=10, max_features=0.3,
                                  n_jobs=-1, min_impurity_decrease=0.1, class_weight="balanced"),
            RandomForestClassifier(n_estimators=1000, criterion="entropy", max_depth=3, max_features=0.7,
                                  n_jobs=-1, min_impurity_decrease=0.001, class_weight="balanced"),
            MLPClassifier(hidden_layer_sizes=(24, 16, 8), activation="relu", solver="adam", alpha=0.01,
                         batch_size=64, learning_rate="constant", learning_rate_init=0.002,
                         max_iter=3, early_stopping=True),
           MLPClassifier(hidden_layer_sizes=(24), activation="logistic", solver="adam", alpha=0.01,
                         batch_size=64, learning_rate="invscaling", learning_rate_init=0.02,
                         max_iter=3, early_stopping=True),
           HistGradientBoostingClassifier(max_iter=1000, learning_rate=0.001, max_depth=10,
                                      max_features=0.2, l2_regularization=0.001, min_samples_leaf=24),
           HistGradientBoostingClassifier(max_iter=1000, learning_rate=0.001, max_depth=3,
                                          max_features=0.2, l2_regularization=0.0001, min_samples_leaf=24),
           LogisticRegression(class_weight=None, penalty=None, solver="lbfgs", max_iter=1000, n_jobs=-1)
           ],
          [RandomForestClassifier(n_estimators=1000, criterion="gini", max_depth=10, max_features=3,
                                  n_jobs=-1, min_impurity_decrease=0.01, class_weight="balanced")]]

# RandomForestClassifier(n_estimators=1000, criterion="gini", max_depth=4, max_features=2,
    #                                   min_samples_leaf=20, ccp_alpha=0.001, n_jobs=-1)
    ROOT_PATH = Path("../../data/train")

    num_samples_set_0 = 77125
    num_samples_set_1 = 52320
    num_samples_set_2 = 64215
    num_samples_set_3 = 68095
    num_samples_set_4 = 66020
    num_samples_set_5 = 46595
    num_samples_set_4_offset = 66015
    num_samples_set_5_offset = 46595


    train_features_indices = np.arange(np.sum([num_samples_set_0, num_samples_set_1,
                                               num_samples_set_2, num_samples_set_3])).reshape(-1, 1)
    test_features_1_indices = (np.arange(num_samples_set_4).reshape(-1, 1) +
                               np.sum([num_samples_set_0, num_samples_set_1, num_samples_set_2,
                                       num_samples_set_3]))
    test_features_2_indices = (np.arange(num_samples_set_5).reshape(-1, 1) +
                               np.sum([num_samples_set_0, num_samples_set_1, num_samples_set_2,
                                       num_samples_set_3, num_samples_set_4]))
    #offset_test_features_2_indices = (np.arange(num_samples_set_5).reshape(-1, 1) +
    #                          np.sum([num_samples_set_0, num_samples_set_1, num_samples_set_2,
    #                                   num_samples_set_3, num_samples_set_4, num_samples_set_5]))
    offset_test_features_1_indices = (np.arange(num_samples_set_4 - 5).reshape(-1, 1) +
                               np.sum([num_samples_set_0, num_samples_set_1, num_samples_set_2,
                                       num_samples_set_3, num_samples_set_4, num_samples_set_5]))

    previous_predictions = np.load("stack_test_preds_4_final_version_definite_fixed_channels_iteration_4.npy")
    previous_offset_predictions = np.load("stack_test_preds_4_final_version_definite_fixed_channels_iteration_4_offset.npy")
    add_features, add_labels, add_groups = create_additional_features_labels_groups(prediction=previous_predictions,
                                                                                    test_features=test_features_1_indices,
                                                                                    offset_test_features=offset_test_features_1_indices,
                                                                                    offset_predictions=previous_offset_predictions)

    # 24643 + 8445 = 33088 test set 1; abridged_iteration_1
    # 27334 + 13583 = 40917 test set 1; definite_iteration_2
    # 27768 + 12658 = 40426 test set 1; definite_iteration_3

    # 10083 + 4175 = 14258 test set 2; abridged_iteration_1
    # 11589 + 5057 = 16646 test set 2; definite_iteration_2
    # changed from RandomForest to LogisticRegression as final layer classifier
    # 199 + 2830 = 3029 test set 2; definite_iteration_3
    make_test_predictions(save_name="final_version_definite_fixed_channels", restacking=False, domain_adaptation=False,
                          additional_features=add_features, additional_labels=add_labels,
                          additional_groups=add_groups, starting_iteration=4)