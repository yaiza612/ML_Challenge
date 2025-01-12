from pathlib import Path
import numpy as np
from scipy.signal import butter, sosfilt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import cohen_kappa_score, accuracy_score
from sklearn.model_selection import LeaveOneGroupOut

from src.data_loading.create_submission import create_submission_from_flattened_preds
from src.data_loading.loader import reshape_array_into_windows
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from pystacknet.pystacknet import StackNetClassifier
from sklearn.impute import SimpleImputer

import pandas as pd


num_samples_set_0 = 77125
num_samples_set_1 = 52320
num_samples_set_2 = 64215
num_samples_set_3 = 68095
num_samples_set_4 = 66020
num_samples_set_5 = 46595


def load_features_labels_and_groups(feature_type, normalizing_strings, end_index=6):
    ROOT_PATH = "../../data/Engineered_features/"
    indices = range(end_index)
    train_features = \
        pd.concat(
            [pd.concat(
                [pd.read_csv(f"{ROOT_PATH}eeg_{normalizing_string}{feature_type}_features_{train_index}.csv")
                 for normalizing_string in normalizing_strings], axis=1)
                for train_index in indices], axis=0)
    return train_features


def missing_values_table(df):
    mis_val = df.isnull().sum()
    mis_val_percent = 100 * df.isnull().sum() / len(df)
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
    mis_val_table_ren_columns = mis_val_table.rename(columns={0: 'Missing Values',
                                                              1: '% of Total Values'})
    mis_val_table_ren_columns = mis_val_table_ren_columns[
        mis_val_table_ren_columns.iloc[:, 1] != 0].sort_values('% of Total Values', ascending=False).round(1)
    print("Your selected dataframe has "
          + str(df.shape[1]) + " columns.\n" 
                               "There are " +
          str(mis_val_table_ren_columns.shape[0]) +
          " columns that have missing values.")
    return mis_val_table_ren_columns


def print_missing_values(feature_types):
    for feature_type in feature_types:
        train_features = load_features_labels_and_groups(feature_type=feature_type,
                                                         normalizing_strings=normalizing_types)
        df = missing_values_table(train_features)
        print(df)  # the missing features are very rare, so it is worthy to impute them


def get_feature_names_of_feature_type(feature_type):
    ROOT_PATH = "../../data/Engineered_features/"
    df = pd.read_csv(f"{ROOT_PATH}eeg_{feature_type}_features_0.csv")
    return df.columns


def select_uncorrelated_features(correlation_matrix, threshold):
    """
    Given a correlation matrix and a threshold, this function returns a list of feature indices
    such that no pairwise correlation exceeds the threshold.

    Parameters:
    correlation_matrix (numpy.ndarray): Correlation matrix of features.
    threshold (float): The maximum allowed correlation between any two features.

    Returns:
    numpy.ndarray: Indices of the selected features.
    """
    # Number of features
    n_features = correlation_matrix.shape[0]

    # Create a boolean array to mark which features to keep (True means keep)
    keep = np.ones(n_features, dtype=bool)

    # Iterate over each pair of features
    for i in range(n_features):
        for j in range(i + 1, n_features):
            if abs(correlation_matrix[i, j]) > threshold:
                # If both features are still marked for keeping, remove the second feature (j)
                if keep[j]:
                    keep[j] = False

    # Return the indices of the features that are kept
    return np.where(keep)[0]

if __name__ == "__main__":
    normalizing_types = ["", "min_max_scaled_", "standard_scaled_", "robust_scaled_", "detrend_",
                           "global_min_max_scaled_", "global_standard_scaled_", "global_robust_scaled_",
                           "global_detrend_"]
    feature_types = ["frequency_domain", "time_frequency_domain", "nonlinear", "time_domain", "spatial"]
    counter = 0
    feature_names_to_use = []
    for feature_type in feature_types:
        train_features = load_features_labels_and_groups(feature_type=feature_type,
                                                         normalizing_strings=normalizing_types, end_index=2)
        feature_names = get_feature_names_of_feature_type(feature_type)
        #train_features.replace([np.inf, -np.inf], np.nan, inplace=True)
        #imp = SimpleImputer()
        #train_array = imp.fit_transform(train_features)
        #train_features = pd.DataFrame(data=train_array, columns=train_features.columns)

        for feature_name in feature_names:
            feature_names_with_normalization = [normalizing_string + feature_name for
                                                normalizing_string in normalizing_types]
            X = train_features[feature_names_with_normalization]
            spearman_corr = X.corr(method='spearman')
            z = select_uncorrelated_features(np.array(spearman_corr), threshold=0.5)
            for element in z:
                feature_names_to_use.append(feature_names_with_normalization[element])
    #make_test_prediction_and_create_submission(features=train_features, labels=train_labels,
    #                                           test_features_as_list=test_features, model=model)

    print(len(feature_names_to_use))
    print(feature_names_to_use)