import numpy as np
from src.data_loading.loader import load_data_single_channel
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from tslearn.metrics import dtw, lb_keogh, lb_envelope

ranges = [[2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]]
warping_window_size = 50
for first_index in range(1, 6):
    train_features, _ = load_data_single_channel(first_index)
    train_features = train_features
    scaler = StandardScaler()
    for i in range(len(train_features)):
        train_features[i] = scaler.fit_transform(train_features[i][0].reshape(-1, 1)).T
    for second_index in ranges[first_index]:
        if first_index == second_index:
            continue
        previously_calculated_dist_features = np.load(f"../../data/Engineered_features/dtw_features_{first_index}.npy")
        val_features, val_labels = load_data_single_channel(second_index)
        channels = np.array([(_ % 5) + 1 for _ in range(len(val_features))])
        # downsample the validation features
        np.random.seed(42)
        indices_for_downsampling = np.random.choice(np.arange(len(val_features)), size=len(val_features),
                                                    replace=False)[:1000]
        val_features = val_features[indices_for_downsampling]
        val_labels = val_labels[indices_for_downsampling]
        channels = channels[indices_for_downsampling]
        for j in range(len(val_features)):
            val_features[j] = scaler.fit_transform(val_features[j][0].reshape(-1, 1)).T
        dtw_features = np.zeros(shape=(train_features.shape[0], 80))
        for channel_idx, train_feat in tqdm(enumerate(train_features)):
            closest_distances_overall = list(previously_calculated_dist_features[channel_idx, :20])
            labels_closest_distances_overall = list(previously_calculated_dist_features[channel_idx, 20:40])
            closest_distances_same_channel = list(previously_calculated_dist_features[channel_idx, 40:60])
            labels_closest_distances_same_channel = list(previously_calculated_dist_features[channel_idx, 60:])
            train_channel = (channel_idx % 5) + 1
            # order the train_features according to lb_keogh:
            envelope = lb_envelope(train_feat[0], radius=50)
            lower_bounds = np.array([lb_keogh(ts_query=val_feat[0], ts_candidate=None, radius=warping_window_size,
                                              envelope_candidate=envelope) for val_feat in val_features])
            # only take indices into account where the lower bound is lower than the maximum feature
            max_closest_dist_overall = np.max(closest_distances_overall)
            max_closest_dist_same_channel = np.max(closest_distances_same_channel)
            for val_channel_idx, (val_feat, lower_bound) in enumerate(zip(val_features, lower_bounds)):
                val_channel = (val_channel_idx % 5) + 1
                if lower_bound < max_closest_dist_overall or \
                        lower_bound < max_closest_dist_same_channel and val_channel == train_channel:
                    distance = dtw(train_feat[0], val_feat[0], global_constraint="sakoe_chiba",
                                   sakoe_chiba_radius=warping_window_size)

                    closest_distances_overall.append(distance)
                    labels_closest_distances_overall.append(val_labels[val_channel_idx])

                    if val_channel == train_channel:
                        closest_distances_same_channel.append(distance)
                        labels_closest_distances_same_channel.append(val_labels[val_channel_idx])

            # sort the distances and get the indices and the distances of the closest 20 samples
            indices_of_closest_distances = np.argpartition(closest_distances_overall, kth=range(20))
            closest_distances_overall = list(np.array(closest_distances_overall)[indices_of_closest_distances[:20]])
            labels_closest_distances_overall = list(np.array(labels_closest_distances_overall)[indices_of_closest_distances[:20]])
            # now only consider samples of the same channel
            indices_of_closest_distances_same_channel = np.argpartition(closest_distances_same_channel, kth=range(20))
            closest_distances_same_channel = list(np.array(closest_distances_same_channel)[indices_of_closest_distances_same_channel[:20]])
            labels_closest_distances_same_channel = list(np.array(labels_closest_distances_same_channel)[indices_of_closest_distances_same_channel[:20]])

            dtw_features[channel_idx, :20] = np.array(closest_distances_overall)
            dtw_features[channel_idx, 20:40] = np.array(labels_closest_distances_overall)
            dtw_features[channel_idx, 40:60] = np.array(closest_distances_same_channel)
            dtw_features[channel_idx, 60:] = np.array(labels_closest_distances_same_channel)
        np.save(f"../../data/Engineered_features/dtw_features_{first_index}.npy", dtw_features)

