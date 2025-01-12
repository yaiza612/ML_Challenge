import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from dask.datasets import timeseries
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from src.constants import feature_names_to_use
import time


def load_train_data():
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
            for i in range(4):
                filename = f'{path_for_features}eeg_{normalizing_string}{filename_ext}_features_{i}.csv'
                df = pd.read_csv(filename)
                column_names = df.columns
                overlapping_names = list(set(column_names) & set(feature_names_to_use))
                short_df = df[overlapping_names]
                temp_df.append(short_df)
            temp_df = pd.concat(temp_df)
            all_dfs.append(temp_df)

    labels = pd.concat([pd.read_csv(f'{path_for_features}eeg_label_features_{i}.csv') for i in range(4)], axis=0)
    final_df = pd.concat(all_dfs, axis=1)
    return final_df, labels




# 1. Calculate Pearson and Spearman Correlation
def calculate_correlations(X):
    # Pearson correlation
    pearson_corr = X.corr(method='pearson')
    # Spearman correlation
    spearman_corr = X.corr(method='spearman')

    # Plot heatmaps for Pearson and Spearman correlations
    #plt.figure(figsize=(12, 8))
    #sns.heatmap(pearson_corr, annot=False, cmap='coolwarm', center=0)
    #plt.title('Pearson Correlation Heatmap')
    #plt.show()

    #plt.figure(figsize=(12, 8))
    #sns.heatmap(spearman_corr, annot=False, cmap='coolwarm', center=0)
    #plt.title('Spearman Correlation Heatmap')
    #plt.show()

    return pearson_corr, spearman_corr


# 2. Variance and Standard Deviation
def calculate_variance_std(X):
    variance = X.var()
    std_deviation = X.std()
    low_variance_features = variance[variance < 0.01]  # Threshold for low variance

    print("Features with low variance:")
    print(low_variance_features)

    return variance, std_deviation


# 3. Mutual Information between each feature and the target
def calculate_mutual_information(X, y):
    mutual_info = mutual_info_classif(X, y, random_state=42)
    mutual_info_series = pd.Series(mutual_info, index=X.columns).sort_values(ascending=False)

    print("Mutual Information scores:")
    print(mutual_info_series)

    return mutual_info_series


# 4. Feature Importance from a Random Forest model
def calculate_feature_importance(X, y):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    feature_importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)

    return feature_importance


# Run the analyses
# For the correlations, we are interested whether features with different standardization ways are correlated
# Assume df is your pandas dataframe and 'target' is the column with the labels
# Split the data into features (X) and target (y)


#variance, std_deviation = calculate_variance_std(X)
#mutual_info = calculate_mutual_information(X, y)
X, y = load_train_data()
start = time.time()
feature_importance = calculate_feature_importance(X, y)
end = time.time()
print(end - start)
print(feature_importance.index.tolist())
print(feature_importance.values.tolist())


