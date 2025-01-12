import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


def plot_correlated_features(df, correlation_threshold=0.3):
    """
    Analyzes features in a DataFrame to identify those with high correlation to the label.
    For strongly correlated features, plots the relationship with the label and differentiates by channel.

    Parameters:
    - df: pd.DataFrame - DataFrame containing features and labels.
    - correlation_threshold: float - Threshold for the absolute correlation to consider a feature significant.
    """
    # Drop any rows with NaN values, if present
    df = df.dropna()

    # Calculate correlation matrix
    correlation_matrix = df.corr()

    # Iterate over each channel label column (label_ch1, label_ch2, ..., label_ch5)
    for channel_idx in range(1, 6):  # Assuming you have label_ch1 to label_ch5
        label_col = f'label_ch{channel_idx}'

        # Extract correlations of features with the label
        label_correlation = correlation_matrix[label_col].drop(label_col)  # Exclude the label's self-correlation

        # Filter features that meet the correlation threshold
        significant_features = label_correlation[label_correlation.abs() >= correlation_threshold]

        print(f"Significantly correlated features with {label_col}:")
        print(significant_features)

        # Plot each feature with high correlation
        for feature, corr_value in significant_features.items():
            plt.figure(figsize=(8, 4))

            # Determine if feature is channel-specific by checking if "channel" is in the name
            is_channel_specific = f"channel_{channel_idx}" in feature

            # Plot title and label with correlation direction
            correlation_type = "Positive" if corr_value > 0 else "Negative"
            plt.title(f"{correlation_type} Correlation: {feature} vs. {label_col} (corr = {corr_value:.2f})")

            if is_channel_specific:
                # Plot feature values by channel, colored by label
                sns.violinplot(x=label_col, y=feature, data=df, inner="quart")
            else:
                # Plot feature values (non-channel-specific) with respect to the label
                sns.scatterplot(x=df[feature], y=df[label_col])

            plt.show()


def plot_facetgrid(df, correlation_threshold=0.3):
    """
    Creates a FacetGrid of features against the labels, based on strong correlation with the labels.

    Parameters:
    - df: pd.DataFrame - DataFrame containing features and labels.
    - correlation_threshold: float - Threshold for the absolute correlation to consider a feature significant.
    """
    # Drop any rows with NaN values, if present
    df = df.dropna()

    # Calculate correlation matrix
    correlation_matrix = df.corr()

    # Iterate over each channel label column (label_ch1, label_ch2, ..., label_ch5)
    for channel_idx in range(1, 6):  # Assuming you have label_ch1 to label_ch5
        label_col = f'label_ch{channel_idx}'

        # Extract correlations of features with the label
        label_correlation = correlation_matrix[label_col].drop(label_col)  # Exclude the label's self-correlation

        # Filter features that meet the correlation threshold
        significant_features = label_correlation[label_correlation.abs() >= correlation_threshold]

        print(f"Significantly correlated features with {label_col}:")
        print(significant_features)

        # Create FacetGrid for visualizing features against the label
        g = sns.FacetGrid(df, col=label_col, col_wrap=3, height=4, sharex=False, sharey=False)

        # Map the significant features to the FacetGrid
        for feature in significant_features.index:
            g.map(sns.boxplot, feature)

        plt.subplots_adjust(top=0.9)
        g.fig.suptitle(f"FacetGrid of Features vs {label_col}", fontsize=16)
        plt.show()

def plot_global_spatial_features(df, label_prefix='label_ch', correlation_threshold=0.3):
    """
    Analyzes global spatial features to identify those with high correlation to channel-specific labels.
    For each strongly correlated spatial feature, plots the relationship with each channel's label.

    Parameters:
    - df: pd.DataFrame - DataFrame containing global spatial features and channel-specific labels.
    - label_prefix: str - Prefix for channel-specific label columns (e.g., 'label_ch').
    - correlation_threshold: float - Threshold for absolute correlation to consider a feature important.
    """
    # Identify all channel-specific label columns
    label_columns = [col for col in df.columns if col.startswith(label_prefix)]

    # Drop any rows with NaN values, if present
    df = df.dropna()

    # Calculate correlation matrix for features and labels
    correlation_matrix = df.corr()

    # Loop through each channel-specific label
    for label_col in label_columns:
        # Extract correlations of spatial features with the current channel's label
        label_correlation = correlation_matrix[label_col].drop(label_columns)  # Exclude label columns

        # Filter features that meet the correlation threshold
        significant_features = label_correlation[label_correlation.abs() >= correlation_threshold]

        print(f"\nSignificantly correlated spatial features with {label_col}:")
        print(significant_features)

        # Plot each feature with high correlation for the current channel's label
        for feature, corr_value in significant_features.items():
            plt.figure(figsize=(8, 4))

            # Determine the correlation direction
            correlation_type = "Positive" if corr_value > 0 else "Negative"
            plt.title(f"{correlation_type} Correlation: {feature} vs. {label_col} (corr = {corr_value:.2f})")

            # Plotting: Boxplot for discrete labels (e.g., quality 0/1) or scatterplot if label is continuous
            if df[label_col].nunique() <= 2:  # Assumes binary labels (good/bad)
                sns.boxplot(x=label_col, y=feature, data=df)
                plt.ylabel("Spatial Feature Value")
            else:  # Continuous labels
                sns.scatterplot(x=df[label_col], y=df[feature])
                plt.xlabel(f"{label_col}")
                plt.ylabel("Spatial Feature Value")

            plt.show()

if __name__ == "__main__":
    path_for_features = ("../../data/Engineered_features/")
    #df_spatial_features = pd.read_csv(f"{path_for_features}eeg_spatial_features.csv",)
    #print(df_spatial_features.head)
    #plot_global_spatial_features(df_spatial_features)
    df_nl_domain_features = pd.read_csv(f"{path_for_features}eeg_nonlinear_features.csv")
    print(df_nl_domain_features.head)
    plot_correlated_features(df_nl_domain_features)
