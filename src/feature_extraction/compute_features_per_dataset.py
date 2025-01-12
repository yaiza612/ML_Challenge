from frequency_domain_features import compute_frequency_domain_features
from frequency_time_domain_features import compute_time_frequency_domain_features
from non_linear_features import compute_nonlinear_features
from src.data_loading.loader import load_train_data_as_windows, load_test_data_as_windows
import pandas as pd
from tqdm import tqdm

if __name__ == "__main__":

    path_for_features = ("../../data/Engineered_features/")
    all_segment_features = []
    for feature_compute_function, file_name in zip([compute_frequency_domain_features,
                                                compute_time_frequency_domain_features, compute_nonlinear_features],
        ["frequency_domain", "time_frequency_domain", "nonlinear"]):
        for index, function in zip([0, 1, 2, 3, 4, 5], [load_train_data_as_windows, load_train_data_as_windows,
                                                        load_train_data_as_windows, load_train_data_as_windows,
                                                        load_test_data_as_windows, load_test_data_as_windows]):

            features = []
            if index < 4:
                data, labels = function(index)
            else:
                data = function(index)
            #print(f"Labels shape: {labels.shape}")
            print(f"Data shape: {data.shape}")

            for idx in tqdm(range(data.shape[1]), desc=f"Processing train data of index {index} for {file_name} features"):
                segment_data = data[:, idx, :]  # Shape: (5, 500)
                if index < 4:
                    segment_labels = labels[:, idx]  # Shape: (5,)
                    time_freq_features_df = feature_compute_function(segment_data, labels=segment_labels)
                else:
                    time_freq_features_df = feature_compute_function(segment_data, labels=None)

                features.append(time_freq_features_df)

            final_features_df = pd.concat(features, ignore_index=True)
            filename = f'{path_for_features}eeg_{file_name}_features_data_{index}.csv'
            final_features_df.to_csv(filename, index=False)
            print(f"Time-frequency features and labels saved to {filename}")