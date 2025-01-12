import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer


def rename_columns():
    filename_extensions = ["frequency_domain", "time_frequency_domain",
                           "nonlinear",
                           "time_domain", "spatial"]
    path_for_features = "../../data/Engineered_features/"

    normalizing_strings = ["min_max_scaled_", "standard_scaled_", "robust_scaled_", "detrend_",
                           "global_min_max_scaled_", "global_standard_scaled_", "global_robust_scaled_",
                           "global_detrend_"]
    for normalizing_string in normalizing_strings:
        for i in [4, 5]:
            for filename_ext in filename_extensions:
                filename = f'{path_for_features}eeg_{normalizing_string}{filename_ext}_features_{i}_offset250.csv'
                df = pd.read_csv(filename)
                df.rename(columns=lambda x: f"{normalizing_string}{x}", inplace=True)
                df.to_csv(filename, index=False)


def impute_values():
    filename_extensions = ["frequency_domain", "time_frequency_domain",
                           "nonlinear",
                           "time_domain", "spatial"]
    path_for_features = "../../data/Engineered_features/"

    normalizing_strings = ["", "min_max_scaled_", "standard_scaled_", "robust_scaled_", "detrend_",
                           "global_min_max_scaled_", "global_standard_scaled_", "global_robust_scaled_",
                           "global_detrend_"]
    for normalizing_string in normalizing_strings:
        for i in [4]:
            for filename_ext in filename_extensions:
                filename = f'{path_for_features}eeg_{normalizing_string}{filename_ext}_features_{i}_offset250.csv'
                df = pd.read_csv(filename)
                df.replace([np.inf, -np.inf], np.nan, inplace=True)
                imp = SimpleImputer()
                X = imp.fit_transform(df)
                Y = pd.DataFrame(data=X, columns=df.columns)
                Y.to_csv(filename, index=False)


if __name__ == "__main__":
    impute_values()
    #rename_columns()



