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
import pandas as pd


num_samples_set_0 = 77125
num_samples_set_1 = 52320
num_samples_set_2 = 64215
num_samples_set_3 = 68095
num_samples_set_4 = 66020
num_samples_set_5 = 46595


def load_features_labels_and_groups(feature_type: str):
    ROOT_PATH = "../../data/Engineered_features/"
    train_indices = range(4)
    test_indices = range(4, 6)
    train_features = pd.concat([pd.read_csv(f"{ROOT_PATH}eeg_{feature_type}_features_{train_index}.csv")
                                  for train_index in train_indices], axis=0)
    train_labels = pd.concat([pd.read_csv(f"{ROOT_PATH}eeg_label_features_{train_index}.csv")
                                  for train_index in train_indices], axis=0)
    train_groups = np.concatenate([[_] * num_samples for _, num_samples in enumerate([num_samples_set_0, num_samples_set_1,
                                                                       num_samples_set_2, num_samples_set_3])])
    test_features = [np.asarray(pd.read_csv(f"{ROOT_PATH}eeg_{feature_type}_features_{test_index}.csv"))
                                for test_index in test_indices]
    return np.asarray(train_features), np.asarray(train_labels).ravel(), np.asarray(train_groups), test_features



def make_val_predictions(features, labels, groups, model):
    logo = LeaveOneGroupOut()
    logo.get_n_splits(X=features, y=labels, groups=groups)
    cohens_kappas = []
    all_predictions = []
    for i, (train_index, val_index) in enumerate(logo.split(features, labels, groups)):
        train_features, train_labels = features[train_index], labels[train_index]
        val_features, val_labels = features[val_index], labels[val_index]
        print(i)

        model.fit(train_features, train_labels)
        preds = model.predict(val_features)
        cohens_kappas.append(cohen_kappa_score(preds, val_labels))
        all_predictions.append(preds)
    total_preds = np.concatenate(all_predictions)
    cohens_kappas.append(cohen_kappa_score(total_preds, labels))

    print(np.array(cohens_kappas))


def make_test_prediction_and_create_submission(features, labels, test_features_as_list, model):
    train_features, train_labels = features, labels
    model.fit(train_features, train_labels)
    preds_as_list = [model.predict(test_features) for test_features in test_features_as_list]
    create_submission_from_flattened_preds(preds_as_list[0], preds_as_list[1])



if __name__ == "__main__":
    train_features, train_labels, train_groups, test_features = \
        load_features_labels_and_groups(feature_type="time_frequency_domain")
    model = HistGradientBoostingClassifier(max_depth=5, max_features=0.5)
    make_val_predictions(features=train_features, labels=train_labels, groups=train_groups, model=model)
    make_test_prediction_and_create_submission(features=train_features, labels=train_labels,
                                               test_features_as_list=test_features, model=model)
    # analyze_predictions()
    # make_test_predictions()

