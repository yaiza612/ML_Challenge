from pathlib import Path
from autogluon.tabular import TabularPredictor
import numpy as np
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import LeaveOneGroupOut
from src.automl_scripts.custom_autogluon_kappa import custom_kappa_metric
from src.data_loading.create_submission import create_submission_from_flattened_preds

import pandas as pd
from src.constants import feature_importances, feature_names_order_importance
import time


num_samples_set_0 = 77125
num_samples_set_1 = 52320
num_samples_set_2 = 64215
num_samples_set_3 = 68095
num_samples_set_4 = 66020
num_samples_set_5 = 46595

def load_features(indices):
    features_to_use = [feature_name for feature_name, feature_importance in
                       zip(feature_names_order_importance, feature_importances) if feature_importance > 0.001]
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


def load_features_labels_and_groups():
    path_for_features = "../../data/Engineered_features/"
    train_features = load_features(range(4))
    test_features_1 = load_features(range(4, 5))
    test_features_2 = load_features(range(5, 6))

    labels = pd.concat([pd.read_csv(f'{path_for_features}eeg_label_features_{i}.csv') for i in range(4)], axis=0)

    train_groups = np.concatenate(
        [[_] * num_samples for _, num_samples in enumerate([num_samples_set_0, num_samples_set_1,
                                                            num_samples_set_2, num_samples_set_3])])
    return train_features, labels, train_groups, [test_features_1, test_features_2]



def make_val_predictions(features, labels, groups, features_and_labels, test_features):
    logo = LeaveOneGroupOut()
    logo.get_n_splits(X=features, y=labels, groups=groups)
    cohens_kappas = []
    all_predictions = []
    for i, (train_index, val_index) in enumerate(logo.split(features, labels, groups)):
        train_features_and_labels = features_and_labels.iloc[train_index]
        val_features_and_labels = features_and_labels.iloc[val_index]
        print(i)
        predictor = TabularPredictor(label="label", eval_metric=custom_kappa_metric).fit(train_data=train_features_and_labels,
                                                                                         tuning_data=val_features_and_labels,
                                                                                         time_limit=3600*8, presets="best_quality",
                                                                                         use_bag_holdout=True)
        preds = predictor.predict(val_features_and_labels)
        cohens_kappas.append(cohen_kappa_score(preds, val_features_and_labels["label"]))
        all_predictions.append(preds)
        for idx, element in enumerate(test_features):
            test_preds = predictor.predict_proba(element)
            with open(f"autogluon_fold_{i}_test_data_{idx}.npy", "wb") as f:
                np.save(f, test_preds)
        val_preds = predictor.predict_proba(val_features_and_labels)
        with open(f"autogluon_fold_{i}_val_data.npy", "wb") as f:
            np.save(f, val_preds)
        train_preds = predictor.predict_proba(train_features_and_labels)
        with open(f"autogluon_fold_{i}_train_data.npy", "wb") as f:
            np.save(f, val_preds)

    total_preds = np.concatenate(all_predictions)
    cohens_kappas.append(cohen_kappa_score(total_preds, labels))

    print(np.array(cohens_kappas))


def make_test_prediction_and_create_submission(features, labels, test_features_as_list, model):
    train_features, train_labels = features, labels
    model.fit(train_features, train_labels)
    preds_as_list = [model.predict(test_features) for test_features in test_features_as_list]
    create_submission_from_flattened_preds(preds_as_list[0], preds_as_list[1])


def analyze_autogluon_preds():
    test_preds_total = [[np.load(f"autogluon_fold_{i}_test_data_{j}.npy")[:, 1] for i in range(4)] for j in range(2)]
    test_preds_stacked = [np.stack(test_preds_part) for test_preds_part in test_preds_total]
    #test_preds_positive = [np.where(test_pred_stacked > 0.95, 1, 0) for test_pred_stacked in test_preds_stacked]
    #test_preds_negative = [np.where(test_pred_stacked < 0.05, 1, 0) for test_pred_stacked in test_preds_stacked]
    #summed_positive = [np.sum(test_pred_positive, axis=0) for test_pred_positive in test_preds_positive]
    #summed_negative = [np.sum(test_pred_negative, axis=0) for test_pred_negative in test_preds_negative]
    #certain_positive = [np.where(summed_positive > 2, 1, 0) for test_pred_summed in test_preds_summed]
    test_preds_summed = [np.sum(test_pred_stacked, axis=0) for test_pred_stacked in test_preds_stacked]
    final_predictions = [np.where(test_pred_summed > 1.8, 1, 0) for test_pred_summed in test_preds_summed]
    create_submission_from_flattened_preds(final_predictions[0], final_predictions[1], "autogluon")


if __name__ == "__main__":
    #train_features, train_labels, train_groups, test_features = load_features_labels_and_groups()
    #features_and_labels = pd.concat((train_features, train_labels), axis=1)
    #make_val_predictions(features=train_features, labels=train_labels, groups=train_groups,
    #                     features_and_labels=features_and_labels, test_features=test_features)
    analyze_autogluon_preds()
    # [0.71857053 0.77780801 0.72208875 0.78377242 0.82134881]
    #make_test_prediction_and_create_submission(features=train_features, labels=train_labels,
    #                                           test_features_as_list=test_features, model=model)
    # analyze_predictions()
    # make_test_predictions()

