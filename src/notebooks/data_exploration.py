from pathlib import Path
import numpy as np
from scipy.signal import butter, sosfilt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import cohen_kappa_score, accuracy_score
from sklearn.model_selection import LeaveOneGroupOut

from src.data_loading.create_submission import create_submission_from_flattened_preds
from src.data_loading.loader import reshape_array_into_windows
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from pystacknet.pystacknet import StackNetClassifier
from sklearn.preprocessing import StandardScaler


class CustomKNN(KNeighborsClassifier):
    """
    this class splits the collected means, variances and amplitudes into the bandpass-buckets for classifying them
    separately and merging the predictions
    """
    def __init__(self, n_neighbors):
        super(CustomKNN, self).__init__()
        self.n_neighbors = n_neighbors
        self.knns = []

    def fit(self, X, y):
        first_part = X[:, :3]
        second_part = X[:, 3:6]
        third_part = X[:, 6:]
        parts = [first_part, second_part, third_part]
        self.knns = [KNeighborsClassifier(n_neighbors=self.n_neighbors).fit(part, y) for part in parts]

    def predict_proba(self, X):
        first_part = X[:, :3]
        second_part = X[:, 3:6]
        third_part = X[:, 6:]
        parts = [first_part, second_part, third_part]
        preds = [knn.predict_proba(part) for part, knn in zip(parts, self.knns)]

        stacked_predictions = np.vstack([p[:, 1] for p in preds])
        summed_predictions = np.sum(stacked_predictions, axis=0)
        return summed_predictions / 3


n_estimators = 30
models = [[RandomForestClassifier(n_estimators=n_estimators, criterion="entropy", max_depth=5, max_features=0.5,
                                  random_state=1),
           KNeighborsClassifier(n_neighbors=50),
           GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=0.1, max_depth=5, max_features=0.5,
                                      random_state=1),
           LogisticRegression(random_state=1),
           MLPClassifier(hidden_layer_sizes=(20, 10, 5), max_iter=300),
           ],
          [RandomForestClassifier(n_estimators=n_estimators, criterion="entropy", max_depth=5, max_features=0.5,
                                  random_state=1),
           MLPClassifier(hidden_layer_sizes=(20, 10, 5), max_iter=300),
           GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=0.1, max_depth=5, max_features=0.5,
                                      random_state=1)],
          [RandomForestClassifier(n_estimators=n_estimators, criterion="entropy", max_depth=5, max_features=0.5,
                                  random_state=1)]]

# TODO figure out why SVC does not work
ROOT_PATH = Path("../../data/train")
TEST_PATH = Path("../../data/test")

training_data = [(np.array(np.load(ROOT_PATH / f"data_{i}.npy") / 2 ** 8, dtype=np.float16),
                  np.array(np.load(ROOT_PATH / f"target_{i}.npy"), dtype=np.float16)) for i in
                 range(4)]

test_data = [(np.array(np.load(TEST_PATH / f"data_{i}.npy") / 2 ** 8, dtype=np.float16), None)
             for i in range(4, 6)]


def custom_cohen_kappa_score(y_true, y_pred, sample_weight=None):
    y_pred = np.where(y_pred > 0.5, 1, 0)
    return cohen_kappa_score(y_true, y_pred, sample_weight=sample_weight)


def butter_bandpass(lowcut, highcut, fs, order=5):
    return butter(order, [lowcut, highcut], fs=fs, btype='band')


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    sos = butter(order, [lowcut, highcut], fs=fs, btype='band', output="sos")
    filtered_signal = sosfilt(sos, data)
    return filtered_signal


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
    #scaler = StandardScaler()
    #amplitude = scaler.fit_transform(amplitude.reshape(-1, 1))[:, 0]
    #mean = scaler.fit_transform(mean.reshape(-1, 1))[:, 0]
    #variance = scaler.fit_transform(variance.reshape(-1, 1))[:, 0]
    return amplitude, mean, variance


def get_labels_and_groups_from_training_data_for_channel_prediction(train_data):
    # for predicting the channel, we want the groups to be according to the dataset,
    # same as if we would want to predict the actual labels
    all_targets = []
    groups = []
    for group_idx, (data, target) in enumerate(train_data):
        filtered_data = butter_bandpass_filter(data, 0.1, 18, 250, 4)
        reshaped_data = reshape_array_into_windows(filtered_data, 250, 2)  # 5, 200, 500
        targets = np.array([[_] * reshaped_data.shape[1] for _ in range(5)]).reshape(-1)
        all_targets.append(targets)
        groups.append(np.ones_like(targets) * group_idx)
    all_targets = np.concatenate(all_targets)
    all_groups = np.concatenate(groups)
    return all_targets, all_groups


def get_labels_and_groups_from_training_data_for_dataset_prediction(train_data):
    # for predicting the channel, we want the groups to be according to the dataset,
    # same as if we would want to predict the actual labels
    all_targets = []
    groups = []
    for group_idx, (data, target) in enumerate(train_data):
        filtered_data = butter_bandpass_filter(data, 0.1, 18, 250, 4)
        reshaped_data = reshape_array_into_windows(filtered_data, 250, 2)  # 5, 200, 500
        targets = np.array([group_idx] * reshaped_data.shape[1] * 5)
        all_targets.append(targets)
        groups.append(np.random.randint(low=0, high=5, size=targets.shape))
    all_targets = np.concatenate(all_targets)
    all_groups = np.concatenate(groups)
    return all_targets, all_groups


def get_labels_and_groups_from_training_data_for_class_prediction(train_data):
    all_targets = []
    groups = []
    for group_idx, (data, target) in enumerate(train_data):
        filtered_data = butter_bandpass_filter(data, 0.1, 18, 250, 4)
        reshaped_data = reshape_array_into_windows(filtered_data, 250, 2)
        targets_flatten = target[..., :len(reshaped_data[0])].reshape(-1)
        all_targets.append(targets_flatten)
        groups.append(np.ones_like(targets_flatten) * group_idx)
    all_targets = np.concatenate(all_targets)
    all_groups = np.concatenate(groups)
    return all_targets, all_groups


def construct_features(low_list, high_list, dataset, concatenate_all=False):
    feature_list = []
    for low, high in zip(low_list, high_list):
        amplitude, mean, variance = get_bandpass_features_from_training_data(dataset, low, high, 5)
        all_features = concatenate_features([amplitude, mean, variance])
        feature_list.append(all_features)
    if concatenate_all:
        feature_list = [np.hstack(feature_list)]
    return feature_list


def make_val_predictions():
    lows = [1, 3, 20]
    highs = [20, 30, 35]
    feature_list = construct_features(lows, highs, training_data)
    test_feature_list = construct_features(lows, highs, test_data)
    all_targets, all_groups = get_labels_and_groups_from_training_data_for_class_prediction(training_data)

    example_features = feature_list[0]

    logo = LeaveOneGroupOut()
    logo.get_n_splits(X=example_features, y=all_targets, groups=all_groups)
    cohens_kappas = []
    for i, (train_index, test_index) in enumerate(logo.split(example_features, all_targets, all_groups)):
        print(i)
        flag = True
        additional_features = None
        additional_labels = None
        total_added = 0
        while flag:
            flag = False
            train_features_for_knn = [f[train_index] for f in feature_list]
            train_labels = all_targets[train_index]
            if additional_features is not None:
                train_features_for_knn = [np.vstack((train_f, added_f)) for train_f, added_f in
                                          zip(train_features_for_knn, additional_features)]
                train_labels = np.hstack((train_labels, additional_labels))
            val_features_for_knn = [f[test_index] for f in feature_list]
            val_labels = all_targets[test_index]
            neighbor_algs = [StackNetClassifier(models, metric=custom_cohen_kappa_score, folds=4, restacking=False,
                                                use_retraining=True,
                                                use_proba=True, random_state=12345, n_jobs=1, verbose=1)
                             for _ in range(len(val_features_for_knn))]
            for neighbor_alg, train_features in zip(neighbor_algs, train_features_for_knn):
                neighbor_alg.fit(train_features, train_labels)
            predictions = [neighbor_alg.predict_proba(val_features) for neighbor_alg, val_features in
                           zip(neighbor_algs, val_features_for_knn)]

            stacked_predictions = np.vstack([p[:, 0] for p in predictions])
            summed_predictions = np.sum(stacked_predictions, axis=0)  # 3 means class 0
            additional_class_0_indices = (summed_predictions > 2.9).nonzero()
            additional_class_1_indices = (summed_predictions < 0.1).nonzero()
            print(len(additional_class_0_indices[0]) + len(additional_class_1_indices[0]))
            if len(additional_class_0_indices[0]) + len(additional_class_1_indices[0]) > (total_added + 1000):
                total_added = len(additional_class_0_indices[0]) + len(additional_class_1_indices[0])
                additional_features_class_0 = [val_features[additional_class_0_indices] for val_features in
                                               val_features_for_knn]
                additional_features_class_1 = [val_features[additional_class_1_indices] for val_features in
                                               val_features_for_knn]
                additional_labels_class_0 = [0] * len(additional_class_0_indices[0])
                additional_labels_class_1 = [1] * len(additional_class_1_indices[0])
                additional_features = [np.vstack((class_0, class_1)) for class_0, class_1 in
                                       zip(additional_features_class_0, additional_features_class_1)]
                additional_labels = np.hstack((additional_labels_class_0, additional_labels_class_1))
                flag = True
            else:
                cohens_kappas.append(
                    [cohen_kappa_score(np.where(summed_predictions > cutoff, 0, 1), val_labels) for cutoff in
                     np.linspace(0.2, 2.8, 30)])
                # cutoff = np.linspace(0.2, 2.8, 30)[27]
                # prediction = np.where(summed_predictions > cutoff, 0, 1)
                # cohens_kappas.append(cohen_kappa_score(prediction, val_labels))
                test_predictions = np.array(
                    [neighbor_alg.predict_proba(test_features) for neighbor_alg, test_features in
                     zip(neighbor_algs, test_feature_list)])
                np.save(f"stack_test_preds_{i}.npy", test_predictions)
                np.save(f"stack_val_preds_{i}.npy", np.array(predictions))

    print(np.mean(np.array(cohens_kappas), axis=0))
    print(np.mean(np.array(cohens_kappas), axis=1))
    print(np.array(cohens_kappas))


def make_test_predictions(save_name="for_test_predictions_correct", concatenate_all=False, restacking=False,
                          domain_adaptation=False):
    lows = [1, 3, 20]
    highs = [20, 30, 35]

    feature_list = construct_features(lows, highs, training_data, concatenate_all=concatenate_all)
    all_targets, all_groups = get_labels_and_groups_from_training_data_for_class_prediction(training_data)

    example_features = feature_list[0]


    for test_index in range(4, 6):
        print(f"training for test dataset {test_index}")
        test_feature_list = construct_features(lows, highs, [test_data[test_index - 4]], concatenate_all=concatenate_all)
        flag = True
        additional_features = None
        additional_labels = None
        total_added = 0
        while flag:
            flag = False
            train_features_for_knn = [f for f in feature_list]
            train_labels = all_targets
            train_groups = all_groups
            num_feature_sets = len(train_features_for_knn)
            if additional_features is not None:
                train_features_for_knn = [np.vstack((train_f, added_f)) for train_f, added_f in
                                          zip(train_features_for_knn, additional_features)]
                train_labels = np.hstack((train_labels, additional_labels))
                train_groups = np.hstack((train_groups, additional_groups))
            logo = LeaveOneGroupOut()
            logo.get_n_splits(X=train_features_for_knn[0], y=train_labels, groups=train_groups)
            neighbor_algs = [StackNetClassifier(models, metric=custom_cohen_kappa_score,
                                                folds=logo.split(train_features_for_knn[0], train_labels, train_groups),
                                                restacking=restacking, use_retraining=True,
                                                use_proba=True, random_state=12345, n_jobs=1, verbose=1)
                             for _ in range(num_feature_sets)]
            for neighbor_alg, train_features in zip(neighbor_algs, train_features_for_knn):
                neighbor_alg.fit(train_features, train_labels)
            predictions = [neighbor_alg.predict_proba(val_features) for neighbor_alg, val_features in
                           zip(neighbor_algs, test_feature_list)]

            stacked_predictions = np.vstack([p[:, 0] for p in predictions])
            summed_predictions = np.sum(stacked_predictions, axis=0)  # 3 means class 0
            additional_class_0_indices = (summed_predictions > (2.9 / 3) * num_feature_sets).nonzero()
            additional_class_1_indices = (summed_predictions < (0.1 / 3) * num_feature_sets).nonzero()
            print(len(additional_class_0_indices[0]) + len(additional_class_1_indices[0]))
            if len(additional_class_0_indices[0]) + len(additional_class_1_indices[0]) > (total_added + 1000) and domain_adaptation:
                total_added = len(additional_class_0_indices[0]) + len(additional_class_1_indices[0])
                additional_features_class_0 = [val_features[additional_class_0_indices] for val_features in
                                               test_feature_list]
                additional_features_class_1 = [val_features[additional_class_1_indices] for val_features in
                                               test_feature_list]
                additional_labels_class_0 = [0] * len(additional_class_0_indices[0])
                additional_labels_class_1 = [1] * len(additional_class_1_indices[0])
                additional_features = [np.vstack((class_0, class_1, class_0, class_1,
                                                  class_0, class_1, class_0, class_1)) for class_0, class_1 in
                                       zip(additional_features_class_0, additional_features_class_1)]
                additional_labels = np.hstack((additional_labels_class_0, additional_labels_class_1,
                                               additional_labels_class_0, additional_labels_class_1,
                                               additional_labels_class_0, additional_labels_class_1,
                                               additional_labels_class_0, additional_labels_class_1))
                additional_groups = np.hstack([np.ones(shape=(total_added)) * i for i in range(4)])
                flag = True
            else:
                test_prediction = np.array([neighbor_alg.predict_proba(test_features) for neighbor_alg, test_features in
                                            zip(neighbor_algs, test_feature_list)])
                val_prediction = np.array([neighbor_alg.predict_proba(train_features) for neighbor_alg, train_features in
                                            zip(neighbor_algs, feature_list)])
                np.save(f"stack_test_preds_{test_index}_{save_name}.npy",
                        test_prediction)
                np.save(f"stack_val_preds_{test_index}_{save_name}.npy",
                        val_prediction)


def analyze_predictions():
    val_preds = [np.load(f"stack_val_preds_{i}.npy") for i in range(4)]
    test_preds = [np.load(f"stack_test_preds_{i}.npy") for i in range(4)]
    val_targets, groups = get_labels_and_groups_from_training_data_for_class_prediction(training_data)
    val_targets_as_list = [val_targets[groups == i] for i in range(4)]
    cohens_kappas = []
    for val_pred, val_target in zip(val_preds, val_targets_as_list):
        stacked_predictions = np.vstack([p[:, 0] for p in val_pred])
        summed_predictions = np.sum(stacked_predictions, axis=0)
        cohens_kappas.append([cohen_kappa_score(np.where(summed_predictions > cutoff, 0, 1), val_target) for cutoff in
                              np.linspace(0.2, 2.8, 300)])
    print(np.mean(np.array(cohens_kappas), axis=0))
    print(np.mean(np.array(cohens_kappas), axis=1))
    print(np.argmax(np.mean(np.array(cohens_kappas), axis=0)))
    final_predictions = []
    for test_pred in test_preds:
        stacked_predictions = np.vstack([p[:, 0] for p in test_pred])
        summed_predictions = np.sum(stacked_predictions, axis=0)
        final_predictions.append(np.where(summed_predictions > np.linspace(0.2, 2.8, 300)[133], 0, 1))
    # first test set: 13204 * 5 predictions
    # second test set: 9319 * 5 predictions
    stacked_final_predictions = np.vstack([p[:, 0] for test_pred in test_preds for p in test_pred])
    summed_final_predictions = np.sum(stacked_final_predictions, axis=0)
    final_prediction = np.where(summed_final_predictions > np.linspace(0.2, 2.8, 300)[133] * 4, 0, 1)
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


def analyze_test_predictions(save_name):
    val_preds = [np.load(f"stack_val_preds_{i}_{save_name}.npy") for i in range(4, 6)]  #1, samples, 2
    test_preds = [np.load(f"stack_test_preds_{i}_{save_name}.npy") for i in range(4, 6)]
    val_targets, groups = get_labels_and_groups_from_training_data_for_class_prediction(training_data)
    val_targets_as_list = [val_targets for i in range(4)]
    cohens_kappas = []
    for val_pred, val_target in zip(val_preds, val_targets_as_list):
        stacked_predictions = np.vstack([p[:, 0] for p in val_pred])
        summed_predictions = np.sum(stacked_predictions, axis=0)
        cohens_kappas.append([cohen_kappa_score(np.where(summed_predictions > cutoff, 0, 1), val_target) for cutoff in
                              np.linspace((0.2/3) * len(val_preds[0]), (2.8/3) * len(val_preds[0]), 300)])
    print(np.mean(np.array(cohens_kappas), axis=0))
    print(np.mean(np.array(cohens_kappas), axis=1))
    print(np.argmax(np.mean(np.array(cohens_kappas), axis=0)))
    cutoff_index = np.argmax(np.mean(np.array(cohens_kappas), axis=0))
    final_predictions = []
    for test_pred in test_preds:
        stacked_predictions = np.vstack([p[:, 0] for p in test_pred])
        summed_predictions = np.sum(stacked_predictions, axis=0)
        final_predictions.append(np.where(summed_predictions > np.linspace((0.2/3) * len(val_preds[0]), (2.8/3) * len(val_preds[0]), 300)[165], 0, 1))
    final_prediction = np.concatenate(final_predictions)
    # first test set: 13204 * 5 predictions
    # second test set: 9319 * 5 predictions
    #z = np.vstack([p[:, 0] for p in test_pred for test_pred in test_preds])
    #stacked_final_predictions = np.vstack([p[:, 0] for test_pred in test_preds for p in test_pred])
    #summed_final_predictions = np.sum(stacked_final_predictions, axis=0)
    #final_prediction = np.where(summed_final_predictions > np.linspace(0.2, 2.8, 300)[133] * 4, 0, 1)
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
    # analyze_test_predictions(save_name="for_test_predictions_w_restacking_concatenated_features_solved")
    #analyze_predictions()
    make_test_predictions(save_name="for_test_predictions_solved_non_scaled", concatenate_all=False, restacking=False)
    #make_test_predictions(save_name="for_test_predictions_w_restacking_solved_scaled_v2", concatenate_all=False, restacking=True)
    #make_test_predictions(save_name="for_test_predictions_w_restacking_concatenated_features_solved", concatenate_all=True, restacking=True)
    # TODO / IDEA: increase the weight of the test dataset samples in training