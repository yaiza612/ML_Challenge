from src.data_loading.loader import load_data_single_channel
import numpy as np
from sktime.dists_kernels import FlatDist, ScipyDist
from sktime.classification.distance_based import KNeighborsTimeSeriesClassifier
from sktime.classification.dictionary_based import BOSSEnsemble, WEASEL, IndividualBOSS, ContractableBOSS
from sktime.classification.kernel_based import RocketClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import cohen_kappa_score, accuracy_score
from tqdm import tqdm
from scipy.signal import detrend



for first_index in range(1, 4):
    for second_index in range(2, 4):
        if first_index == second_index:
            continue
        print(f"training on index {first_index} and validating on index {second_index}")
        train_features, train_labels = load_data_single_channel(first_index)
        val_features, val_labels = load_data_single_channel(second_index)
        num_samples = 3000
        in_group_features = train_features[num_samples:]
        in_group_labels = train_labels[num_samples:]
        train_features = train_features[:num_samples]
        train_labels = train_labels[:num_samples]
        scaler = StandardScaler()
        for i in range(len(train_features)):
            train_features[i] = scaler.fit_transform(detrend(train_features[i])[0].reshape(-1, 1)).T
        for j in range(len(val_features)):
            val_features[j] = scaler.fit_transform(detrend(val_features[j])[0].reshape(-1, 1)).T
        for k in range(len(in_group_features)):
            in_group_features[k] = scaler.fit_transform(detrend(in_group_features[k])[0].reshape(-1, 1)).T
        # example 1 - 3-NN with simple dynamic time warping distance (requires numba)
        #clf = KNeighborsTimeSeriesClassifier(n_neighbors=3)

        # example 2 - custom distance:
        # 3-nearest neighbour classifier with Euclidean distance (on flattened time series)
        # (requires scipy)
        clf = RocketClassifier(num_kernels=100)
        #clf = WEASEL(bigrams=False)
        #clf = ContractableBOSS(feature_selection="chi2", max_ensemble_size=5, n_parameter_samples=10,
        #                       time_limit_in_minutes=3)
        #clf = IndividualBOSS(feature_selection="chi2")
        clf.fit(train_features, train_labels)
        y_pred = clf.predict(train_features)
        print(cohen_kappa_score(y_pred, train_labels))
        print(accuracy_score(y_pred, train_labels))

        batch_size = 500
        in_group_features_batches = [in_group_features[i * batch_size:(i + 1) * batch_size] for i in
                                range(len(in_group_features) // batch_size)]
        in_group_labels_batches = [in_group_labels[i * batch_size:(i + 1) * batch_size] for i in
                              range(len(in_group_features) // batch_size)]
        in_group_labels_batches.append(in_group_labels[(len(in_group_features) // batch_size) * batch_size:])
        y_pred = [clf.predict(elem) for elem in tqdm(in_group_features_batches)]
        y_pred.append(clf.predict(in_group_features[(len(in_group_features) // batch_size) * batch_size:]))
        y_pred = np.hstack(y_pred)
        in_group_labels = np.hstack(in_group_labels_batches)
        print(cohen_kappa_score(y_pred, in_group_labels))
        print(accuracy_score(y_pred, in_group_labels))

        val_features_batches = [val_features[i * batch_size:(i + 1) * batch_size] for i in
                                range(len(val_features) // batch_size)]
        val_labels_batches = [val_labels[i * batch_size:(i + 1) * batch_size] for i in
                              range(len(val_features) // batch_size)]
        val_labels_batches.append(val_labels[(len(val_features)//batch_size) * batch_size:])
        y_pred = [clf.predict(elem) for elem in tqdm(val_features_batches)]
        y_pred.append(clf.predict(val_features[(len(val_features)//batch_size) * batch_size:]))
        y_pred = np.hstack(y_pred)
        val_labels = np.hstack(val_labels_batches)
        print(cohen_kappa_score(y_pred, val_labels))
        print(accuracy_score(y_pred, val_labels))
        """  0%|          | 0/104 [00:00<?, ?it/s]0.9591733973857366
        0.988
        100%|██████████| 104/104 [02:41<00:00,  1.55s/it]
        -0.3180562572104242
        0.3353019877675841
        """