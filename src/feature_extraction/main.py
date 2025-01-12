import os
import pandas as pd
from src.data_loading.loader import load_data_as_windows, load_data, reshape_array_into_windows
from src.feature_extraction.frequency_domain_features import compute_frequency_domain_features, bandpass_filter
from src.feature_extraction.frequency_time_domain_features import compute_time_frequency_domain_features
from src.feature_extraction.non_linear_features import compute_nonlinear_features
from src.feature_extraction.time_domain_features import compute_time_domain_features
from src.feature_extraction.label_features import compute_label_features
from src.feature_extraction.channel_features import compute_channel_features
from src.feature_extraction.sp import compute_spatial_features
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from scipy.signal import detrend
import numpy as np


if __name__ == "__main__":
    freq_bands = {'delta': (0.5, 4), 'theta': (4, 8), 'alpha': (8, 13), 'beta': (13, 30), 'gamma': (30, 45)}
    key_list = ['delta', 'theta', 'alpha', 'beta', 'gamma']
    feature_calculation_functions = [compute_frequency_domain_features, compute_time_frequency_domain_features,
                                     compute_nonlinear_features, compute_time_domain_features, compute_spatial_features]#,
                                     #compute_label_features, compute_channel_features]
    filename_extensions = ["frequency_domain", "time_frequency_domain",
                           "nonlinear",
                           "time_domain", "spatial"]#, "label", "channel"]
    path_for_features = "../../data/Engineered_features/"
    normalizing_types = [None, MinMaxScaler, StandardScaler, RobustScaler, None]
    normalizing_strings = ["", "min_max_scaled_", "standard_scaled_", "robust_scaled_", "detrend_"]
    for global_normalization in [True, False]:
        global_string = ["", "global_"][global_normalization]
        offsets = [250]
        for offset in offsets:
            for normalizing_type, normalizing_string in zip(normalizing_types, normalizing_strings):
                for i in [5, 4]:
                    for feat_calc_func, filename_ext in zip(feature_calculation_functions, filename_extensions):
                        filename = f'{path_for_features}eeg_{global_string}{normalizing_string}{filename_ext}_features_{i}_offset{offset}.csv'
                        if not os.path.isfile(filename):
                            data, labels = load_data(i)
                            print(data.shape)
                            data = data[:, offset:]
                            print(data.shape)
                            filtered_data = dict(zip(key_list, [reshape_array_into_windows(
                                bandpass_filter(data, fs=250, low=freq_bands[key][0], high=freq_bands[key][1]), 250, 2)
                                                      for key in key_list]))
                            all_segment_features = []
                            if global_normalization and normalizing_type is not None:
                                scaler = normalizing_type()
                                for _ in range(5):
                                    data[_] = scaler.fit_transform(data[_].reshape(-1, 1))[:, 0]
                            elif global_normalization and normalizing_string == "detrend_":
                                for _ in range(5):
                                    data[_] = detrend(data[_])
                            data = reshape_array_into_windows(data, sample_rate=250, window_duration_in_seconds=2)

                            for idx in tqdm(range(data.shape[1]), desc=f"Processing data of index {i} for {filename_ext} features with {normalizing_string} Normalization"):
                                segment_data = data[:, idx, :]  # Shape: (5, 500)
                                filtered_segment_data = dict(zip(key_list, [filtered_data[key][:, idx, :] for key in key_list]))
                                if labels is not None:
                                    segment_labels = labels[:, idx]  # Shape: (5,)
                                else:
                                    segment_labels = None
                                if normalizing_type is not None and not global_normalization:
                                    scaler = normalizing_type()
                                    for _ in range(5):
                                        segment_data[_] = scaler.fit_transform(segment_data[_].reshape(-1, 1))[:, 0]
                                elif normalizing_string == "detrend_" and not global_normalization:
                                    for _ in range(5):
                                        segment_data[_] = detrend(segment_data[_])
                                time_freq_features_df = feat_calc_func(segment_data, labels=segment_labels,
                                                                       filtered_segment_data=filtered_segment_data)#,
                                                                       #prepend_string=normalizing_string)

                                all_segment_features.append(time_freq_features_df)
                                #print(time_freq_features_df)

                            final_features_df = pd.concat(all_segment_features, ignore_index=True)
                            final_features_df.to_csv(filename, index=False)
                            print(f"{filename_ext} features and labels saved to {filename}")
