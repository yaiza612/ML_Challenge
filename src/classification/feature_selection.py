import json

from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, AdaBoostClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.metrics import cohen_kappa_score
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
scaling = True
from tqdm import tqdm
import warnings
warnings.simplefilter("ignore", category=ConvergenceWarning)
"""Current Selected Features: ['band_power_alpha', 'peak_to_peak', 'diffuse_slowing']
Progress of Cohens Kappa: [np.float64(0.6833924167142755), np.float64(0.7075338175848329), np.float64(0.7403098369436502)]

Second Iteration:
Progress of Cohens Kappa: [np.float64(0.6166070542938406), np.float64(0.7014806348903257)]
Final selected features: ['band_power_beta', 'band_power_theta']"""


all_feature_names = ['psd_total', 'spectral_entropy', 'sharp_spike', 'power_std',
                     'low_signal_amplitude', 'band_ratio_delta',
                     'band_ratio_theta', 'band_ratio_alpha', 'band_ratio_beta',
                     'band_power_gamma', 'band_ratio_gamma',
                     'band_power_alpha', 'peak_to_peak', 'diffuse_slowing',
                     'band_power_theta', 'band_power_beta',
                     'band_power_delta',
                     'hjorth_mobility', 'hjorth_complexity', 'false_nearest_neighbors',
                     'shannon_entropy', 'arma_coef_1', 'arma_coef_2', 'mean_frequency',
                     'entropy', 'avg_power_delta', 'band_ratio_delta', 'avg_power_theta',
                     'band_ratio_theta', 'avg_power_alpha', 'band_ratio_alpha',
                     'avg_power_beta', 'band_ratio_beta', 'avg_power_gamma',
                     'band_ratio_gamma', 'mean', 'std_dv', 'rms',
                     'skewness', 'kurt', 'mean_amplitude', 'std_amplitude', 'mean_correlation',
                     'mean_plv', 'mean_theta_coh', 'mean_alpha_coh',
                     'mean_beta_coh', 'mean_gamma_coh']



def format_array_to_target_format(array, record_number):
    assert isinstance(record_number, int)
    assert isinstance(array, np.ndarray)
    assert len(array.shape) == 2
    assert array.shape[0] == 5
    assert set(np.unique(array)) == {0, 1}
    formatted_target = []
    for i in range(array.shape[0]):
        channel_encoding = (i + 1) * 100000
        record_number_encoding = record_number * 1000000
        for j in range(array.shape[1]):
            formatted_target.append(
                {
                    "identifier": record_number_encoding + channel_encoding + j,
                    "target": array[i, j],
                }
            )
    return formatted_target


def get_overall_results(preds_1, preds_2):
    results = []
    for record_number, pred in enumerate([preds_1, preds_2]):
        pred = np.array(pred)
        pred = np.reshape(pred, (5, len(pred) // 5), order="F")
        record_number += 4
        formatted_preds = format_array_to_target_format(pred,record_number)
        results.extend(formatted_preds)
    df = pd.DataFrame(results)
    df.to_csv("submission.csv",index = False)


def merge_dataframes(dfs_by_feature, train_data):
    if train_data:
        common_column_names = ["channel", "label_ch1", "label_ch2", "label_ch3", "label_ch4", "label_ch5"]
    else:
        common_column_names = ["channel"]
    # ["frequency_domain", "nonlinear", "time_frequency_domain", "time_domain", "spatial"]
    features_0 = dfs_by_feature[0]
    features_1 = dfs_by_feature[1]
    features_2 = dfs_by_feature[2]
    features_3 = dfs_by_feature[3]
    features_4 = dfs_by_feature[4]
    features_1_unique = features_1.drop(columns=common_column_names)
    features_2_unique = features_2.drop(columns=common_column_names)
    features_2_unique = features_2_unique.drop(columns=['segment'])
    features_3_unique = features_3.drop(columns=common_column_names)
    features_3_unique = features_3_unique.drop(columns=['segment'])
    if train_data:
        column_names_features_4 = ['channel', 'label']
    else:
        column_names_features_4 = ['channel']
    features_4_unique = features_4.drop(columns=column_names_features_4)
    merge_1 = pd.concat([features_0, features_1_unique], axis=1)
    merge_2 = pd.concat([merge_1, features_2_unique], axis=1)
    merge_3 = pd.concat([merge_2, features_3_unique], axis=1)
    df = pd.concat([merge_3, features_4_unique], axis=1)
    # one-hot encode the channel feature
    one_hot = pd.get_dummies(df['channel'], dtype=int)
    one_hot.columns = [f'channel_{i}' for i in range(1, 6)]  # sklearn does not like that some column names
    # are strings, while the others are integers
    # Drop column channel as it is now one-hot encoded
    #df = df.drop('channel', axis=1)
    # Join the encoded df
    df = df.join(one_hot)
    return df


def split_by_channels_and_prepare_data(merged_df):
    split_by_channel = [merged_df[merged_df["channel"] == idx] for idx in range(1, 6)]
    labels_by_channel = [split_by_channel[idx - 1][f"label_ch{idx}"] for idx in range(1, 6)]
    split_by_channel_labels_removed = [
        df.drop(columns=["label_ch1", "label_ch2", "label_ch3", "label_ch4", "label_ch5"])
        for df in split_by_channel]

    features = pd.concat(split_by_channel_labels_removed, axis=0)
    features = features.drop('channel', axis=1)
    labels = pd.concat(labels_by_channel, axis=0)
    return features, labels

def initialize_dfs():
    ROOT_PATH = "../../data/Engineered_features/"
    feature_types = ["frequency_domain", "nonlinear", "time_frequency_domain", "time_domain", "spatial"]
    train_feature_sets = []
    train_label_sets = []
    val_feature_sets = []
    val_label_sets = []
    for val_index in range(4):
        dfs_per_feature_train = []
        dfs_per_feature_val = []
        train_indices = [_ for _ in range(4) if _ != val_index]
        for feature_type in feature_types:
            dfs_by_train_index = [pd.read_csv(f"{ROOT_PATH}eeg_{feature_type}_features_data_{train_index}.csv")
                                  for train_index in train_indices]
            dfs_by_val_index = [pd.read_csv(f"{ROOT_PATH}eeg_{feature_type}_features_data_{train_index}.csv")
                                for train_index in [val_index]]
            df_train = pd.concat(dfs_by_train_index, axis=0)
            dfs_per_feature_train.append(df_train)
            df_val = pd.concat(dfs_by_val_index, axis=0)
            dfs_per_feature_val.append(df_val)

        train_df = merge_dataframes(dfs_per_feature_train, train_data=True)
        X_train, y_train = split_by_channels_and_prepare_data(train_df)
        val_df = merge_dataframes(dfs_per_feature_val, train_data=True)
        X_val, y_val = split_by_channels_and_prepare_data(val_df)
        if scaling:
            imp = SimpleImputer(missing_values=np.nan)
            imputed_values = imp.fit_transform(X_train)
            X_train = pd.DataFrame(imputed_values, columns=X_train.columns)
            imputed_values = imp.transform(X_val)
            X_val = pd.DataFrame(imputed_values, columns=X_val.columns)
            # optionally scale the data
            scaler = MinMaxScaler()
            scaled_values = scaler.fit_transform(X_train)
            X_train = pd.DataFrame(scaled_values, columns=X_train.columns)
            scaled_values = scaler.transform(X_val)
            X_val = pd.DataFrame(scaled_values, columns=X_val.columns)
        val_feature_sets.append(X_val)
        val_label_sets.append(y_val)
        train_feature_sets.append(X_train)
        train_label_sets.append(y_train)

    return train_feature_sets, train_label_sets, val_feature_sets, val_label_sets

def find_best_features(partial_feature_names, train_feature_sets, train_label_sets, val_feature_sets, val_label_sets):
    perf_measures = []
    for val_index in range(4):
        X_train = train_feature_sets[val_index]
        X_val = val_feature_sets[val_index]
        y_train = train_label_sets[val_index]
        y_val = val_label_sets[val_index]

        X_train_partial_features = X_train[partial_feature_names]
        X_val_partial_features = X_val[partial_feature_names]
        clf = KNeighborsClassifier(n_neighbors=5)
        clf.fit(X=X_train_partial_features, y=y_train)
        preds_val = clf.predict(X=X_val_partial_features)
        perf_measure = cohen_kappa_score(preds_val, y_val)
        perf_measures.append(perf_measure)

    return np.mean(perf_measures)


if __name__ == "__main__":

    train_feature_sets, train_label_sets, val_feature_sets, val_label_sets = initialize_dfs()

    selected_features = []
    best_cohens_kappa = -2
    cohens_kappa_list = []
    while all_feature_names:  # continues loop until list is empty
        best_feature = None
        for index in tqdm(range(len(all_feature_names)), desc="Trying features"):
            feature = all_feature_names[index]
            current_features = selected_features + [feature]
            score = find_best_features(partial_feature_names=current_features,
                                       train_feature_sets=train_feature_sets,
                                       train_label_sets=train_label_sets,
                                       val_feature_sets=val_feature_sets,
                                       val_label_sets=val_label_sets)
            if score > best_cohens_kappa:
                best_cohens_kappa = score
                best_feature = feature
        if best_feature is not None:
            selected_features.append(best_feature)
            all_feature_names.remove(best_feature)
            cohens_kappa_list.append(best_cohens_kappa)
            print(f"Selected Feature {best_feature}, Cohens Kappa: {best_cohens_kappa:.4f}")
            print(f"Current Selected Features: {selected_features}")
            print(f"Progress of Cohens Kappa: {cohens_kappa_list}")
        else:
            break
    print(f"Final selected features: {selected_features}")

"""
k = 8 for KNN:
First Iteration:
Progress of Cohens Kappa: [np.float64(0.6237643976522289), np.float64(0.6511029009354065), np.float64(0.6712029069399523), np.float64(0.6767384432813931)]
Final selected features: ['band_power_alpha', 'band_power_theta', 'band_power_beta', 'band_power_delta']

Second Iteration:
Progress of Cohens Kappa: [np.float64(0.4791341698828815), np.float64(0.5052470284054397), np.float64(0.5242162072177063)]
Final selected features: ['band_power_gamma', 'psd_total', 'power_std']
"""



"""
clf = DecisionTreeClassifier(max_depth=5)
First Iteration: (stopped because convergence rate was slow)
Current Selected Features: ['band_power_alpha', 'peak_to_peak', 'diffuse_slowing']
Progress of Cohens Kappa: [np.float64(0.6833924167142755), np.float64(0.7075338175848329), np.float64(0.7403098369436502)]

Second Iteration:
Progress of Cohens Kappa: [np.float64(0.6166070542938406), np.float64(0.7014806348903257)]
Final selected features: ['band_power_beta', 'band_power_theta']

Third Iteration:
Progress of Cohens Kappa: [np.float64(0.4173161257587067), np.float64(0.6109679413258498), np.float64(0.6256037409973461), np.float64(0.631475086858585), np.float64(0.6333568302691754)]
Final selected features: ['std_dv', 'band_ratio_alpha', 'mean_beta_coh', 'hjorth_complexity', 'kurt']
"""


"""
clf = MLPClassifier(hidden_layer_sizes=(10, 10), batch_size=128, max_iter=10)
First Iteration:
Progress of Cohens Kappa: [np.float64(0.5129916211255613)]
Final selected features: ['peak_to_peak']
"""


"""
clf = LinearSVC()
First Iteration:
Progress of Cohens Kappa: [np.float64(0.2583344244032778), np.float64(0.29463855779367476), np.float64(0.3747339056518566), np.float64(0.4056614771306062), np.float64(0.42111641027146407), np.float64(0.43016170006267074), np.float64(0.4373369971057893), np.float64(0.4382468868409125), np.float64(0.43861080111594375), np.float64(0.4386421830279973)]
Final selected features: ['rms', 'band_ratio_gamma', 'false_nearest_neighbors', 'band_ratio_beta', 'mean_correlation', 'mean_gamma_coh', 'kurt', 'mean_frequency', 'band_ratio_gamma', 'band_power_beta']

"""

"""
clf = GaussianNB()
First Iteration:
Progress of Cohens Kappa: [np.float64(0.33947777111567734), np.float64(0.3815119534910015), np.float64(0.4188931019015663), np.float64(0.4239191308367094), np.float64(0.42902510950269523), np.float64(0.43166966010095187), np.float64(0.43336004800263117), np.float64(0.4368388862480871), np.float64(0.4389552927046873), np.float64(0.43981720482411657), np.float64(0.4414584669269387), np.float64(0.44260482773412513)]
Final selected features: ['peak_to_peak', 'arma_coef_1', 'shannon_entropy', 'band_ratio_gamma', 'skewness', 'band_ratio_gamma', 'mean_beta_coh', 'false_nearest_neighbors', 'band_ratio_beta', 'band_ratio_beta', 'kurt', 'mean_theta_coh']
"""


"""
clf = AdaBoostClassifier(n_estimators=10, algorithm="SAMME")
First Iteration: (stopped because convergence rate was low
Current Selected Features: ['peak_to_peak', 'avg_power_gamma', 'band_ratio_beta', 'band_ratio_gamma', 'spectral_entropy']
Progress of Cohens Kappa: [np.float64(0.4917334129569309), np.float64(0.4923435313158762), np.float64(0.5009630163264025), np.float64(0.5125280567249944), np.float64(0.5131409732197914)]
"""

"""
clf = QuadraticDiscriminantAnalysis()
First Iteration:
Final selected features: ['peak_to_peak', 'std_dv', 'arma_coef_1', 'mean_frequency', 'low_signal_amplitude', 'avg_power_delta', 'avg_power_theta', 'mean_beta_coh', 'shannon_entropy', 'spectral_entropy', 'skewness', 'mean_gamma_coh', 'false_nearest_neighbors', 'mean_alpha_coh', 'mean_theta_coh']
Progress of Cohens Kappa: [np.float64(0.33947777111567734), np.float64(0.4228253723888965), np.float64(0.4699119353325), np.float64(0.48561888200511494), np.float64(0.5040077535908463), np.float64(0.5215545351923778), np.float64(0.5330370671485719), np.float64(0.5375172457946031), np.float64(0.5406163866247518), np.float64(0.5414767724908292), np.float64(0.5425557094609965), np.float64(0.5440269508666887), np.float64(0.5453500723220774), np.float64(0.546158768440472), np.float64(0.5462758187424487)]
"""


"""
clf = RandomForestClassifier(max_depth=5, n_estimators=10)
First Iteration:
Progress of Cohens Kappa: [np.float64(0.5036639151931797), np.float64(0.5737180825955361), np.float64(0.6078214911538767), np.float64(0.6363827522970962)]
Final selected features: ['peak_to_peak', 'band_ratio_alpha', 'hjorth_complexity', 'band_power_gamma']

"""



"""
clf = LogisticRegression()
First Iteration:
Progress of Cohens Kappa: [np.float64(0.31384212680540274), np.float64(0.43097398833221506), np.float64(0.4469673460007629), np.float64(0.4488149378211994), np.float64(0.45199380607102607), np.float64(0.454254194885819), np.float64(0.4556204031265726)]
Final selected features: ['band_ratio_beta', 'false_nearest_neighbors', 'peak_to_peak', 'std_dv', 'kurt', 'band_ratio_beta', 'std_amplitude']
"""


# TODO do reverse feature selection by discarding the feature that is least informative by comparing
# all features via feature importance to a random feature (np.random.random(0, 1, size=len(data))

# TODO maybe minmax scaler is not optimal, but other scaler should be used, not sure if it is impactful
# for finding the best features, though

# TODO make a function that takes a classifier and a
# pipeline as input and does the feature importance calculation

# TODO check the cohens kappa score between the classifiers. Ideally, the cohens kappa score
# is small, as that means the classifiers don't agree with each other all the time
# if the classifiers would agree everytime, the additional information on how the
# other classifiers predict samples is meaningless