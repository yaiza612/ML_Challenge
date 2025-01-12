from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import pandas as pd
from sklearn.metrics import cohen_kappa_score, f1_score
from sklearn.model_selection import train_test_split
import numpy as np
from functools import reduce
from sklearn.preprocessing import MinMaxScaler
# from sklearn.experimental import enable_iterative_imputer  <- takes too long
# from sklearn.impute import IterativeImputer
from sklearn.impute import SimpleImputer
scaling = True


spatial_feature_names = [f"autoenc_{i+1}" for i in range(125)]
spatial_feature_names.extend([f'channel_{i}' for i in range(1, 6)])

good_features = [['peak_to_peak', 'band_ratio_alpha', 'hjorth_complexity', 'band_power_gamma'],
['band_power_alpha', 'band_power_theta', 'band_power_beta', 'band_power_delta'],
['band_power_gamma', 'psd_total', 'power_std'],['band_power_alpha', 'peak_to_peak', 'diffuse_slowing'],
['band_power_beta', 'band_power_theta'],['std_dv', 'band_ratio_alpha', 'mean_beta_coh', 'hjorth_complexity', 'kurt'],
['rms', 'band_ratio_gamma', 'false_nearest_neighbors', 'band_ratio_beta', 'mean_correlation', 'mean_gamma_coh', 'kurt', 'mean_frequency', 'band_ratio_gamma', 'band_power_beta'],
['peak_to_peak', 'arma_coef_1', 'shannon_entropy', 'band_ratio_gamma', 'skewness', 'band_ratio_gamma', 'mean_beta_coh', 'false_nearest_neighbors', 'band_ratio_beta', 'band_ratio_beta', 'kurt', 'mean_theta_coh'],
['peak_to_peak', 'avg_power_gamma', 'band_ratio_beta', 'band_ratio_gamma', 'spectral_entropy'],
['peak_to_peak', 'std_dv', 'arma_coef_1', 'mean_frequency', 'low_signal_amplitude', 'avg_power_delta', 'avg_power_theta', 'mean_beta_coh', 'shannon_entropy', 'spectral_entropy', 'skewness', 'mean_gamma_coh', 'false_nearest_neighbors', 'mean_alpha_coh', 'mean_theta_coh'],
['band_ratio_beta', 'false_nearest_neighbors', 'peak_to_peak', 'std_dv', 'kurt', 'band_ratio_beta', 'std_amplitude']]
set_of_good_features = set()
for g_f in good_features:
    for f in g_f:
        set_of_good_features.add(f)

list_of_good_features = list(set_of_good_features)
list_of_good_features.extend([f'channel_{i}' for i in range(1, 6)])


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
    features_5 = dfs_by_feature[5]
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
    merge_4 = pd.concat([merge_3, features_4_unique], axis=1)
    df = pd.concat([merge_4, features_5], axis=1)
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
    feature_types = ["frequency_domain", "nonlinear", "time_frequency_domain", "time_domain", "spatial", "autoencoder"]
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

train_feature_sets, train_label_sets, val_feature_sets, val_label_sets = initialize_dfs()


def run_random_forest_classifier(n_estimators, max_depth, get_test_perf=False):
    ROOT_PATH = "../../data/Engineered_features/"
    feature_types = ["frequency_domain", "nonlinear", "time_frequency_domain", "time_domain", "spatial", "autoencoder"]
    perf_measures = []
    merged_val_labels = []
    merged_val_preds = []
    if not get_test_perf:
        for val_index in range(4):
            X_train = train_feature_sets[val_index]
            X_val = val_feature_sets[val_index]
            y_train = train_label_sets[val_index]
            y_val = val_label_sets[val_index]
            X_train = X_train[list_of_good_features]
            X_val = X_val[list_of_good_features]
            clf = MLPClassifier(hidden_layer_sizes=(30, 10, 10), max_iter=30, alpha=0.01)
            clf.fit(X=X_train, y=y_train)
            preds_train = clf.predict(X=X_train)
            print(f"Train performance for CV split {val_index}")
            print(clf.score(X=X_train, y=y_train))
            print(cohen_kappa_score(preds_train, y_train))
            print(f1_score(y_train, preds_train))

            print(f"Test performance for CV split {val_index}:")
            preds_val = clf.predict(X=X_val)
            merged_val_preds.extend(preds_val)
            merged_val_labels.extend(y_val)
            print(clf.score(X=X_val, y=y_val))
            perf_measure = cohen_kappa_score(preds_val, y_val)
            print(perf_measure)
            print(f1_score(y_val, preds_val))

            #print("-------------")
            perf_measures.append(perf_measure)

    if get_test_perf:
        print("Training on full dataset")
        dfs_per_feature_train = []
        train_indices = range(4)
        for feature_type in feature_types:
            dfs_by_train_index = [pd.read_csv(f"{ROOT_PATH}eeg_{feature_type}_features_data_{train_index}.csv")
                                  for train_index in train_indices]
            df_train = pd.concat(dfs_by_train_index, axis=0)
            dfs_per_feature_train.append(df_train)
        train_df = merge_dataframes(dfs_per_feature_train, train_data=True)
        X_train, y_train = split_by_channels_and_prepare_data(train_df)
        X_train = X_train[list_of_good_features]
        if scaling:
            imp = SimpleImputer(missing_values=np.nan)
            imputed_values = imp.fit_transform(X_train)
            X_train = pd.DataFrame(imputed_values, columns=X_train.columns)
            # optionally scale the data
            scaler = MinMaxScaler()
            scaled_values = scaler.fit_transform(X_train)
            X_train = pd.DataFrame(scaled_values, columns=X_train.columns)
        clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
        clf.fit(X=X_train, y=y_train)
        preds_total = []
        for test_index in range(4, 6):
            dfs_per_feature = []
            for feature_type in feature_types:
                df_by_feature = pd.read_csv(f"{ROOT_PATH}eeg_{feature_type}_features_data_{test_index}.csv")
                dfs_per_feature.append(df_by_feature)



            df = merge_dataframes(dfs_per_feature, train_data=False)
            df = df[list_of_good_features]
            if scaling:
                imputed_values = imp.transform(df)
                df = pd.DataFrame(imputed_values, columns=df.columns)
                scaled_values = scaler.transform(df)
                df = pd.DataFrame(scaled_values, columns=df.columns)
            preds_total.append(clf.predict(df))

        get_overall_results(preds_total[0], preds_total[1])
    return np.mean(perf_measures), np.min(perf_measures), cohen_kappa_score(merged_val_preds, merged_val_labels)

"""
#learning_rate, max_depth, l2_regularization
performance_dict_mean = {}
performance_dict_min = {}
performance_dict_merged = {}
#print(np.logspace(0, -2, num=5, base=10))
#print(np.logspace(-1, -5, num=5, base=10))
for n_estimators in range(10, 51, 10):
    for max_depth in range(5, 10):
        performance_mean, performance_min, performance_merged = run_random_forest_classifier(n_estimators, max_depth)
        print(f"average performance for {n_estimators} estimators and max depth {max_depth}: "
              f"mean: {performance_mean} min: {performance_min} merged: {performance_merged}")
        performance_dict_mean[f"n_est:{n_estimators} depth:{max_depth}"] = performance_mean
        performance_dict_min[f"n_est:{n_estimators} depth:{max_depth}"] = performance_min
        performance_dict_merged[f"n_est:{n_estimators} depth:{max_depth}"] = performance_merged


sorted_by_values = dict(sorted(performance_dict_mean.items(), key=lambda item:item[1]))
print(sorted_by_values)
sorted_by_values = dict(sorted(performance_dict_min.items(), key=lambda item:item[1]))
print(sorted_by_values)
sorted_by_values = dict(sorted(performance_dict_merged.items(), key=lambda item:item[1]))
print(sorted_by_values)
"""
"""
good features:
mean, min and merged:
{'n_est:10 depth:9': np.float64(0.6586149185281348), 'n_est:10 depth:8': np.float64(0.678420036485637), 'n_est:10 depth:7': np.float64(0.6859076967541365), 'n_est:50 depth:9': np.float64(0.6863005563359097), 'n_est:30 depth:8': np.float64(0.6914031361755668), 'n_est:50 depth:8': np.float64(0.6916538997760873), 'n_est:10 depth:5': np.float64(0.6948215489561771), 'n_est:20 depth:9': np.float64(0.6955424748290499), 'n_est:40 depth:9': np.float64(0.6987937169491768), 'n_est:20 depth:7': np.float64(0.698849373005705), 'n_est:30 depth:9': np.float64(0.6990429175811725), 'n_est:20 depth:8': np.float64(0.7016999456812028), 'n_est:30 depth:7': np.float64(0.7023000407472348), 'n_est:40 depth:7': np.float64(0.7040107528185203), 'n_est:40 depth:8': np.float64(0.7058250207626078), 'n_est:10 depth:6': np.float64(0.7064516404813992), 'n_est:20 depth:6': np.float64(0.7073064073599388), 'n_est:40 depth:5': np.float64(0.7082581996253421), 'n_est:50 depth:6': np.float64(0.7090991363782737), 'n_est:50 depth:7': np.float64(0.7109265045745462), 'n_est:50 depth:5': np.float64(0.7124461199544223), 'n_est:40 depth:6': np.float64(0.7125597076334913), 'n_est:30 depth:5': np.float64(0.7125754693282963), 'n_est:30 depth:6': np.float64(0.7128508636107536), 
'n_est:20 depth:5': np.float64(0.7165150936488704)}
{'n_est:10 depth:9': np.float64(0.5303609962009987), 'n_est:10 depth:7': np.float64(0.6082178613853113), 'n_est:10 depth:8': np.float64(0.6094990592374498), 'n_est:30 depth:8': np.float64(0.6253118774839035), 'n_est:10 depth:5': np.float64(0.6257405918772188), 'n_est:20 depth:6': np.float64(0.6267032864176725), 'n_est:10 depth:6': np.float64(0.6301170440978242), 'n_est:20 depth:7': np.float64(0.6302413838581313), 'n_est:20 depth:9': np.float64(0.631448787443975), 'n_est:20 depth:8': np.float64(0.6325833414394317), 'n_est:30 depth:6': np.float64(0.6357566741928525), 'n_est:50 depth:9': np.float64(0.6358254999364288), 'n_est:30 depth:7': np.float64(0.6361495836088225), 'n_est:50 depth:8': np.float64(0.6371289001429126), 'n_est:40 depth:6': np.float64(0.6377050863529061), 'n_est:40 depth:8': np.float64(0.6380365315946326), 'n_est:50 depth:5': np.float64(0.6387365354783003), 'n_est:30 depth:9': np.float64(0.6406856983715703), 'n_est:20 depth:5': np.float64(0.6407044002366145), 'n_est:50 depth:6': np.float64(0.6418361963066911), 'n_est:40 depth:9': np.float64(0.6452535139063338), 'n_est:40 depth:7': np.float64(0.6454320980264524), 'n_est:40 depth:5': np.float64(0.6455068783630982), 'n_est:50 depth:7': np.float64(0.6489449772759611), 
'n_est:30 depth:5': np.float64(0.6494505274660358)}
{'n_est:10 depth:9': np.float64(0.7667955699892672), 'n_est:10 depth:8': np.float64(0.7719614870304389), 'n_est:50 depth:9': np.float64(0.7766222551936083), 'n_est:10 depth:7': np.float64(0.780304919815955), 'n_est:50 depth:8': np.float64(0.7821456630122903), 'n_est:30 depth:8': np.float64(0.7833974420902141), 'n_est:10 depth:5': np.float64(0.7850730112050969), 'n_est:40 depth:9': np.float64(0.7858880399494709), 'n_est:20 depth:9': np.float64(0.7861135862429904), 'n_est:30 depth:9': np.float64(0.7874425145919735), 'n_est:20 depth:7': np.float64(0.7893119618330504), 'n_est:20 depth:8': np.float64(0.790968287775476), 'n_est:40 depth:7': np.float64(0.7920099220909498), 'n_est:30 depth:7': np.float64(0.7923964119043697), 'n_est:10 depth:6': np.float64(0.7945712242542458), 'n_est:40 depth:5': np.float64(0.7946404002952232), 'n_est:40 depth:8': np.float64(0.7948456248897906), 'n_est:50 depth:6': np.float64(0.7978398727392751), 'n_est:50 depth:7': np.float64(0.7979678618629669), 'n_est:30 depth:5': np.float64(0.7983603445218679), 'n_est:20 depth:6': np.float64(0.799721364891748), 'n_est:40 depth:6': np.float64(0.8011890083217028), 'n_est:30 depth:6': np.float64(0.8011971489478307), 'n_est:50 depth:5': np.float64(0.8019821005686791), 
'n_est:20 depth:5': np.float64(0.8042453791282005)}


"""


# 'n_est:10 depth:6 max_f:0.18371681153444983': np.float64(0.7361341373640897)
#print(performance_dict)
#'lr:0.10000 depth:8 l2:0.1': np.float64(0.7303016342678117)
# 'n_est:10 depth:7 max_f:0.0729080649735073': np.float64(0.6681468051139976)
performance = run_random_forest_classifier(n_estimators=20, max_depth=5, get_test_perf=False)
#print(performance)
#performance = run_random_forest_classifier(n_estimators=30, max_depth=6, get_test_perf=False)
#print(performance)

# TODO use automl to get good parameters for random forest, but unlikely the performance can improve much with
# the same featureset
# TODO maybe cohens kappa should be weighted according to how many data points are in a dataset
# or the predictions should be merged for calculating cohens kappa
# TODO add random seed to function call as the results over multiple runs vary

"""
Sorted performance dictionary for HistGradientBoostingClassifier:
{'lr:1.00000 depth:8 l2:0.0001': np.float64(0.5320796383709807), 'lr:1.00000 depth:7 l2:0.01': np.float64(0.5600248695456531), 
'lr:1.00000 depth:5 l2:0.001': np.float64(0.5816416464599616), 'lr:1.00000 depth:6 l2:0.01': np.float64(0.5841587223864608), 
'lr:1.00000 depth:9 l2:0.01': np.float64(0.5885826694265424), 'lr:1.00000 depth:6 l2:0.001': np.float64(0.6128991840233128), 
'lr:1.00000 depth:6 l2:0.0001': np.float64(0.6240325128297892), 'lr:1.00000 depth:5 l2:0.01': np.float64(0.635770220222575), 
'lr:1.00000 depth:7 l2:0.001': np.float64(0.6387092339299205), 'lr:1.00000 depth:8 l2:0.001': np.float64(0.6404987750415593), 
'lr:0.31623 depth:8 l2:0.0001': np.float64(0.6451020753588109), 'lr:1.00000 depth:5 l2:0.1': np.float64(0.6456299019443755), 
'lr:1.00000 depth:7 l2:0.1': np.float64(0.6494790824629562), 'lr:1.00000 depth:5 l2:1e-05': np.float64(0.6549124849501341), 
'lr:1.00000 depth:6 l2:0.1': np.float64(0.6572832861510993), 'lr:1.00000 depth:8 l2:0.01': np.float64(0.6631890926916008), 
'lr:1.00000 depth:8 l2:0.1': np.float64(0.6650640199337903), 'lr:1.00000 depth:7 l2:0.0001': np.float64(0.665591239010558), 
'lr:1.00000 depth:9 l2:0.1': np.float64(0.6666253420081818), 'lr:1.00000 depth:7 l2:1e-05': np.float64(0.6677883580597287), 
'lr:1.00000 depth:9 l2:1e-05': np.float64(0.6698455834800384), 'lr:1.00000 depth:5 l2:0.0001': np.float64(0.6709616265445855), 
'lr:1.00000 depth:9 l2:0.0001': np.float64(0.67249293978447), 'lr:0.01000 depth:8 l2:0.1': np.float64(0.6773700827769561), 
'lr:0.01000 depth:7 l2:1e-05': np.float64(0.6781279961826787), 'lr:0.01000 depth:9 l2:0.1': np.float64(0.678702507441399), 
'lr:0.01000 depth:9 l2:0.01': np.float64(0.6792640840011286), 'lr:0.01000 depth:9 l2:1e-05': np.float64(0.6810633996268838), 
'lr:0.01000 depth:8 l2:0.0001': np.float64(0.6812798709098069), 'lr:0.01000 depth:7 l2:0.001': np.float64(0.6813542143650424), 
'lr:0.31623 depth:7 l2:1e-05': np.float64(0.6814885396964019), 'lr:0.01000 depth:7 l2:0.01': np.float64(0.6815283267471057), 
'lr:0.01000 depth:6 l2:0.1': np.float64(0.6825381309369083), 'lr:0.01000 depth:8 l2:0.001': np.float64(0.682711375351417), 
'lr:0.01000 depth:7 l2:0.1': np.float64(0.6828927380191839), 'lr:0.31623 depth:6 l2:0.1': np.float64(0.6830002844005256), 
'lr:0.01000 depth:9 l2:0.0001': np.float64(0.6838028713898847), 'lr:0.01000 depth:8 l2:0.01': np.float64(0.6838269788110113), 
'lr:0.01000 depth:7 l2:0.0001': np.float64(0.6839739056595632), 'lr:0.01000 depth:9 l2:0.001': np.float64(0.6845488130679875), 
'lr:0.01000 depth:6 l2:0.0001': np.float64(0.6858015129971682), 'lr:1.00000 depth:9 l2:0.001': np.float64(0.6868546227131207), 
'lr:0.01000 depth:8 l2:1e-05': np.float64(0.687091581264685), 'lr:0.31623 depth:5 l2:0.01': np.float64(0.6877661152432901), 
'lr:0.01000 depth:6 l2:0.001': np.float64(0.6882307254389616), 'lr:0.31623 depth:8 l2:0.01': np.float64(0.6883651841897467), 
'lr:0.01000 depth:6 l2:0.01': np.float64(0.6893526781592831), 'lr:0.31623 depth:9 l2:0.1': np.float64(0.6895909202525656), 
'lr:1.00000 depth:8 l2:1e-05': np.float64(0.6911663497224216), 'lr:0.01000 depth:6 l2:1e-05': np.float64(0.6912423416764062), 
'lr:0.31623 depth:6 l2:0.0001': np.float64(0.6930811354245603), 'lr:0.31623 depth:7 l2:0.0001': np.float64(0.6940324943375882), 
'lr:0.31623 depth:7 l2:0.1': np.float64(0.694738795148628), 'lr:1.00000 depth:6 l2:1e-05': np.float64(0.695111888495908), 
'lr:0.31623 depth:6 l2:0.001': np.float64(0.6958008544653461), 'lr:0.31623 depth:5 l2:1e-05': np.float64(0.6972518933601608), 
'lr:0.31623 depth:5 l2:0.0001': np.float64(0.698544480675036), 'lr:0.31623 depth:8 l2:1e-05': np.float64(0.6988954487901734), 
'lr:0.31623 depth:8 l2:0.1': np.float64(0.6993835311272315), 'lr:0.31623 depth:5 l2:0.001': np.float64(0.7002328369135036), 
'lr:0.01000 depth:5 l2:0.001': np.float64(0.7002665232581702), 'lr:0.01000 depth:5 l2:0.0001': np.float64(0.7004838347048259),
'lr:0.01000 depth:5 l2:0.1': np.float64(0.7017410899739293), 'lr:0.31623 depth:8 l2:0.001': np.float64(0.7031550155300135), 
'lr:0.31623 depth:5 l2:0.1': np.float64(0.7033135753728795), 'lr:0.01000 depth:5 l2:0.01': np.float64(0.7033313665122966), 
'lr:0.31623 depth:9 l2:0.0001': np.float64(0.7036374532764427), 'lr:0.01000 depth:5 l2:1e-05': np.float64(0.7037597924431537), 
'lr:0.31623 depth:9 l2:0.01': np.float64(0.7043658613236142), 'lr:0.31623 depth:6 l2:0.01': np.float64(0.7048040920131851), 
'lr:0.31623 depth:9 l2:1e-05': np.float64(0.7048744741999222), 'lr:0.31623 depth:7 l2:0.01': np.float64(0.7053012129943945), 
'lr:0.10000 depth:7 l2:0.0001': np.float64(0.7063709070689705), 'lr:0.10000 depth:5 l2:1e-05': np.float64(0.7089104494568308), 
'lr:0.31623 depth:7 l2:0.001': np.float64(0.7128564969298398), 'lr:0.03162 depth:7 l2:0.01': np.float64(0.7131427919306119), 
'lr:0.10000 depth:7 l2:0.001': np.float64(0.7132137887953053), 'lr:0.10000 depth:7 l2:0.01': np.float64(0.7133501819049038), 
'lr:0.10000 depth:6 l2:0.01': np.float64(0.7139867384604679), 'lr:0.03162 depth:5 l2:0.01': np.float64(0.7142510688668884), 
'lr:0.10000 depth:7 l2:0.1': np.float64(0.7145102861581529), 'lr:0.03162 depth:5 l2:0.001': np.float64(0.7147435765079087), 
'lr:0.10000 depth:6 l2:1e-05': np.float64(0.7147973932940768), 'lr:0.03162 depth:5 l2:0.1': np.float64(0.7147997103985895), 
'lr:0.10000 depth:9 l2:0.01': np.float64(0.7152694739909912), 'lr:0.10000 depth:8 l2:0.01': np.float64(0.7152722631481426), 
'lr:0.10000 depth:6 l2:0.0001': np.float64(0.715348529913146), 'lr:0.03162 depth:8 l2:0.0001': np.float64(0.7155079298732283), 
'lr:0.03162 depth:8 l2:0.1': np.float64(0.7155675324484065), 'lr:0.10000 depth:5 l2:0.1': np.float64(0.715948302904378), 
'lr:0.03162 depth:6 l2:0.0001': np.float64(0.7162022759300055), 'lr:0.03162 depth:5 l2:0.0001': np.float64(0.7162798897373025), 
'lr:0.03162 depth:9 l2:1e-05': np.float64(0.716550403564566), 'lr:0.10000 depth:8 l2:0.0001': np.float64(0.7165647956029926), 
'lr:0.03162 depth:9 l2:0.0001': np.float64(0.7166768036830506), 'lr:0.03162 depth:8 l2:0.01': np.float64(0.7168190891450172), 
'lr:0.03162 depth:7 l2:0.0001': np.float64(0.7169019341845166), 'lr:0.10000 depth:6 l2:0.1': np.float64(0.716914807446132), 
'lr:0.10000 depth:5 l2:0.001': np.float64(0.716961024206067), 'lr:0.03162 depth:6 l2:0.1': np.float64(0.7171342615717063), 
'lr:0.03162 depth:7 l2:0.1': np.float64(0.7173217440457562), 'lr:0.03162 depth:8 l2:0.001': np.float64(0.7173912077448111), 
'lr:0.03162 depth:6 l2:0.01': np.float64(0.71748846317061), 'lr:0.03162 depth:5 l2:1e-05': np.float64(0.7175034983935623), 
'lr:0.03162 depth:9 l2:0.1': np.float64(0.7175471695443594), 'lr:0.31623 depth:9 l2:0.001': np.float64(0.717646695325595), 
'lr:0.03162 depth:7 l2:1e-05': np.float64(0.7178759446575852), 'lr:0.10000 depth:9 l2:1e-05': np.float64(0.7181597471380132), 
'lr:0.03162 depth:7 l2:0.001': np.float64(0.7182708048082481), 'lr:0.03162 depth:8 l2:1e-05': np.float64(0.7186651067481449), 
'lr:0.10000 depth:5 l2:0.0001': np.float64(0.7186659300803927), 'lr:0.10000 depth:6 l2:0.001': np.float64(0.718700576306251), 
'lr:0.03162 depth:6 l2:0.001': np.float64(0.7190021364235561), 'lr:0.03162 depth:6 l2:1e-05': np.float64(0.7195261703914619), 
'lr:0.03162 depth:9 l2:0.001': np.float64(0.7195769625571422), 'lr:0.10000 depth:9 l2:0.001': np.float64(0.7202793896689081), 
'lr:0.03162 depth:9 l2:0.01': np.float64(0.7208478383274719), 'lr:0.10000 depth:5 l2:0.01': np.float64(0.7211958890575956), 
'lr:0.10000 depth:9 l2:0.0001': np.float64(0.7216067775745777), 'lr:0.31623 depth:6 l2:1e-05': np.float64(0.7254188917188769), 
'lr:0.10000 depth:8 l2:1e-05': np.float64(0.7255776445422871), 'lr:0.10000 depth:7 l2:1e-05': np.float64(0.7263984581550158), 
'lr:0.10000 depth:9 l2:0.1': np.float64(0.7285018568146837), 'lr:0.10000 depth:8 l2:0.001': np.float64(0.7285812954795851), 
'lr:0.10000 depth:8 l2:0.1': np.float64(0.7303016342678117)}

"""

"""
Sorted performance dictionaries for RandomForestClassifier:
{'n_est:10 depth:5 max_f:0.0625': np.float64(0.6062908722536955), 'n_est:10 depth:9 max_f:0.2143109957132682': np.float64(0.6103590445503748), 'n_est:10 depth:9 max_f:0.0729080649735073': np.float64(0.633308450862368), 'n_est:40 depth:5 max_f:0.0729080649735073': np.float64(0.6334889232420393), 'n_est:10 depth:6 max_f:0.0625': np.float64(0.6369084415588524), 'n_est:10 depth:8 max_f:0.0625': np.float64(0.6388829868530728), 'n_est:10 depth:8 max_f:0.0729080649735073': np.float64(0.6406267182504828), 'n_est:10 depth:5 max_f:0.0729080649735073': np.float64(0.6475731311823645), 'n_est:20 depth:5 max_f:0.0625': np.float64(0.6476770853485998), 'n_est:10 depth:9 max_f:0.08504937501089857': np.float64(0.6503422038685496), 'n_est:40 depth:6 max_f:0.0625': np.float64(0.6506287181559897), 'n_est:50 depth:5 max_f:0.0729080649735073': np.float64(0.6506617723122672), 'n_est:10 depth:6 max_f:0.0729080649735073': np.float64(0.6513676555948953), 'n_est:10 depth:7 max_f:0.08504937501089857': np.float64(0.6518021328156115), 'n_est:20 depth:7 max_f:0.0729080649735073': np.float64(0.6548052016597381), 'n_est:30 depth:5 max_f:0.0625': np.float64(0.6557353762844742), 'n_est:30 depth:7 max_f:0.0729080649735073': np.float64(0.655878968627013), 'n_est:10 depth:7 max_f:0.18371681153444983': np.float64(0.6565409604014885), 'n_est:30 depth:8 max_f:0.0729080649735073': np.float64(0.6573881947740152), 'n_est:10 depth:5 max_f:0.08504937501089857': np.float64(0.6579790656113398), 'n_est:40 depth:5 max_f:0.0625': np.float64(0.6585073648800792), 'n_est:10 depth:9 max_f:0.0625': np.float64(0.6589631525498785), 'n_est:10 depth:7 max_f:0.09921256574801249': np.float64(0.6592133868268977), 'n_est:20 depth:9 max_f:0.0729080649735073': np.float64(0.6605629574665683), 'n_est:30 depth:9 max_f:0.0625': np.float64(0.660881233668233), 'n_est:50 depth:8 max_f:0.0625': np.float64(0.6615794410652117), 'n_est:50 depth:7 max_f:0.0625': np.float64(0.6619523289923324), 'n_est:30 depth:9 max_f:0.0729080649735073': np.float64(0.6639442284981436), 'n_est:20 depth:7 max_f:0.0625': np.float64(0.6656313685126718), 'n_est:40 depth:7 max_f:0.0625': np.float64(0.6663946965283739), 'n_est:40 depth:6 max_f:0.0729080649735073': np.float64(0.6665272123536159), 'n_est:30 depth:8 max_f:0.0625': np.float64(0.6668918762651354), 'n_est:30 depth:5 max_f:0.0729080649735073': np.float64(0.6670493673494221), 'n_est:10 depth:8 max_f:0.08504937501089857': np.float64(0.6673882570153385), 'n_est:50 depth:8 max_f:0.0729080649735073': np.float64(0.6684027804056722), 'n_est:50 depth:9 max_f:0.0625': np.float64(0.6687209746691573), 'n_est:10 depth:8 max_f:0.1157343390359113': np.float64(0.668958783161547), 'n_est:30 depth:6 max_f:0.0625': np.float64(0.6696421912346422), 'n_est:50 depth:9 max_f:0.0729080649735073': np.float64(0.6704420338253331), 'n_est:40 depth:9 max_f:0.0729080649735073': np.float64(0.6713217510867545), 'n_est:50 depth:7 max_f:0.0729080649735073': np.float64(0.6714344615107566), 'n_est:20 depth:5 max_f:0.0729080649735073': np.float64(0.6718531211150574), 'n_est:20 depth:9 max_f:0.0625': np.float64(0.6731412761955322), 'n_est:20 depth:8 max_f:0.09921256574801249': np.float64(0.6731709050068814), 'n_est:20 depth:6 max_f:0.0729080649735073': np.float64(0.6737519074202435), 'n_est:20 depth:8 max_f:0.0625': np.float64(0.673867118864269), 'n_est:20 depth:6 max_f:0.0625': np.float64(0.6764280829664964), 'n_est:10 depth:9 max_f:0.15749013123685915': np.float64(0.6788144110007341), 'n_est:50 depth:5 max_f:0.0625': np.float64(0.6788312537564862), 'n_est:10 depth:8 max_f:0.2143109957132682': np.float64(0.6794563046226656), 'n_est:30 depth:7 max_f:0.0625': np.float64(0.6796886566013688), 'n_est:50 depth:8 max_f:0.08504937501089857': np.float64(0.6797844157150699), 'n_est:50 depth:9 max_f:0.08504937501089857': np.float64(0.6804057638201498), 'n_est:50 depth:6 max_f:0.0625': np.float64(0.6804451853792064), 'n_est:40 depth:9 max_f:0.0625': np.float64(0.6813452757713406), 'n_est:50 depth:5 max_f:0.08504937501089857': np.float64(0.681663133538782), 'n_est:30 depth:6 max_f:0.08504937501089857': np.float64(0.6817924576891832), 'n_est:30 depth:7 max_f:0.08504937501089857': np.float64(0.681873636078923), 'n_est:50 depth:6 max_f:0.0729080649735073': np.float64(0.6825862238795931), 'n_est:40 depth:8 max_f:0.0625': np.float64(0.6829259408590299), 'n_est:10 depth:7 max_f:0.0729080649735073': np.float64(0.6841192955560318), 'n_est:30 depth:8 max_f:0.1157343390359113': np.float64(0.684218955915946), 'n_est:40 depth:5 max_f:0.08504937501089857': np.float64(0.684399409411852), 'n_est:30 depth:9 max_f:0.1157343390359113': np.float64(0.6849523565782173), 'n_est:40 depth:8 max_f:0.1157343390359113': np.float64(0.6860674376910408), 'n_est:20 depth:5 max_f:0.08504937501089857': np.float64(0.6861918447763289), 'n_est:10 depth:7 max_f:0.1157343390359113': np.float64(0.6862645585670688), 'n_est:10 depth:5 max_f:0.2143109957132682': np.float64(0.6864617475872565), 'n_est:20 depth:8 max_f:0.08504937501089857': np.float64(0.6874362533478438), 'n_est:40 depth:8 max_f:0.0729080649735073': np.float64(0.6878271064433384), 'n_est:30 depth:7 max_f:0.1157343390359113': np.float64(0.6878391335161722), 'n_est:30 depth:6 max_f:0.0729080649735073': np.float64(0.6884152660530105), 'n_est:40 depth:7 max_f:0.08504937501089857': np.float64(0.6884800606626287), 'n_est:10 depth:6 max_f:0.08504937501089857': np.float64(0.6892065946551538), 'n_est:30 depth:7 max_f:0.09921256574801249': np.float64(0.6896215244998393), 'n_est:30 depth:5 max_f:0.1157343390359113': np.float64(0.689674962642107), 'n_est:30 depth:9 max_f:0.09921256574801249': np.float64(0.6903289998558668), 'n_est:10 depth:9 max_f:0.13500746736153826': np.float64(0.6905899297551226), 'n_est:50 depth:6 max_f:0.08504937501089857': np.float64(0.6906827045541769), 'n_est:20 depth:6 max_f:0.1157343390359113': np.float64(0.690752457006006), 'n_est:40 depth:9 max_f:0.1157343390359113': np.float64(0.6912361124669637), 'n_est:20 depth:9 max_f:0.08504937501089857': np.float64(0.6914062796845366), 'n_est:40 depth:9 max_f:0.08504937501089857': np.float64(0.6915690513208037), 'n_est:40 depth:6 max_f:0.08504937501089857': np.float64(0.6916225591251275), 'n_est:50 depth:8 max_f:0.1157343390359113': np.float64(0.6916641992874525), 'n_est:10 depth:5 max_f:0.18371681153444983': np.float64(0.6919422326211493), 'n_est:20 depth:5 max_f:0.1157343390359113': np.float64(0.6919837482786282), 'n_est:20 depth:6 max_f:0.08504937501089857': np.float64(0.6922081642058754), 'n_est:50 depth:7 max_f:0.08504937501089857': np.float64(0.6924041690173669), 'n_est:20 depth:9 max_f:0.25': np.float64(0.6926497044131409), 'n_est:40 depth:5 max_f:0.1157343390359113': np.float64(0.6926537207776903), 'n_est:40 depth:7 max_f:0.0729080649735073': np.float64(0.6931352372491951), 'n_est:20 depth:6 max_f:0.09921256574801249': np.float64(0.6931599967641906), 'n_est:30 depth:9 max_f:0.08504937501089857': np.float64(0.6933353870430001), 'n_est:20 depth:5 max_f:0.09921256574801249': np.float64(0.6935193662220585), 'n_est:40 depth:8 max_f:0.08504937501089857': np.float64(0.6935965025808408), 'n_est:10 depth:7 max_f:0.0625': np.float64(0.6937463442133959), 'n_est:30 depth:5 max_f:0.09921256574801249': np.float64(0.694182264174402), 'n_est:10 depth:8 max_f:0.13500746736153826': np.float64(0.6941954342366885), 'n_est:20 depth:7 max_f:0.09921256574801249': np.float64(0.6944413317812868), 'n_est:30 depth:5 max_f:0.08504937501089857': np.float64(0.6946904008183508), 'n_est:10 depth:5 max_f:0.09921256574801249': np.float64(0.6947191105415444), 'n_est:30 depth:8 max_f:0.13500746736153826': np.float64(0.6954722365250241), 'n_est:20 depth:7 max_f:0.13500746736153826': np.float64(0.6957400509506952), 'n_est:30 depth:8 max_f:0.08504937501089857': np.float64(0.6962256733095518), 'n_est:10 depth:5 max_f:0.1157343390359113': np.float64(0.6970539048047799), 'n_est:50 depth:6 max_f:0.1157343390359113': np.float64(0.6975239172914585), 'n_est:50 depth:5 max_f:0.09921256574801249': np.float64(0.6976638305595331), 'n_est:50 depth:6 max_f:0.09921256574801249': np.float64(0.6977049465281715), 'n_est:20 depth:8 max_f:0.0729080649735073': np.float64(0.6979325818009545), 'n_est:20 depth:9 max_f:0.18371681153444983': np.float64(0.6979544634284656), 'n_est:40 depth:6 max_f:0.09921256574801249': np.float64(0.6981802592780738), 'n_est:50 depth:9 max_f:0.1157343390359113': np.float64(0.6983087610939374), 'n_est:10 depth:6 max_f:0.15749013123685915': np.float64(0.6985878472205826), 'n_est:30 depth:7 max_f:0.13500746736153826': np.float64(0.6992095208033984), 'n_est:50 depth:8 max_f:0.18371681153444983': np.float64(0.6992724966776096), 'n_est:30 depth:9 max_f:0.18371681153444983': np.float64(0.6994147096357888), 'n_est:20 depth:5 max_f:0.13500746736153826': np.float64(0.6994177288169396), 'n_est:30 depth:9 max_f:0.15749013123685915': np.float64(0.6997422680292837), 'n_est:10 depth:8 max_f:0.18371681153444983': np.float64(0.6998317551705003), 'n_est:40 depth:6 max_f:0.13500746736153826': np.float64(0.6999805526976026), 'n_est:20 depth:8 max_f:0.18371681153444983': np.float64(0.7000523841919764), 'n_est:40 depth:7 max_f:0.09921256574801249': np.float64(0.7001591430318386), 'n_est:40 depth:6 max_f:0.1157343390359113': np.float64(0.7001620661034739), 'n_est:50 depth:7 max_f:0.09921256574801249': np.float64(0.7004274502899683), 'n_est:30 depth:8 max_f:0.09921256574801249': np.float64(0.7006316076496772), 'n_est:30 depth:9 max_f:0.13500746736153826': np.float64(0.7007234028604531), 'n_est:50 depth:5 max_f:0.13500746736153826': np.float64(0.7007408008139597), 'n_est:10 depth:6 max_f:0.2143109957132682': np.float64(0.7009678874287456), 'n_est:50 depth:9 max_f:0.09921256574801249': np.float64(0.7009737289148945), 'n_est:50 depth:7 max_f:0.1157343390359113': np.float64(0.7011434460605397), 'n_est:50 depth:5 max_f:0.1157343390359113': np.float64(0.701414745032571), 'n_est:20 depth:9 max_f:0.2143109957132682': np.float64(0.7015694805570991), 'n_est:20 depth:6 max_f:0.15749013123685915': np.float64(0.7016205964098602), 'n_est:10 depth:6 max_f:0.1157343390359113': np.float64(0.7016924335588299), 'n_est:10 depth:8 max_f:0.09921256574801249': np.float64(0.7017723763632722), 'n_est:20 depth:6 max_f:0.13500746736153826': np.float64(0.7019960249530012), 'n_est:10 depth:5 max_f:0.13500746736153826': np.float64(0.7019966849520334), 'n_est:50 depth:8 max_f:0.13500746736153826': np.float64(0.7020917498956952), 'n_est:20 depth:9 max_f:0.13500746736153826': np.float64(0.7024399442156041), 'n_est:10 depth:8 max_f:0.15749013123685915': np.float64(0.7028199614998485), 'n_est:30 depth:6 max_f:0.09921256574801249': np.float64(0.7031286839323384), 'n_est:20 depth:8 max_f:0.1157343390359113': np.float64(0.7031806023115412), 'n_est:40 depth:8 max_f:0.09921256574801249': np.float64(0.703187553216968), 'n_est:40 depth:7 max_f:0.13500746736153826': np.float64(0.7035122265073497), 'n_est:50 depth:9 max_f:0.15749013123685915': np.float64(0.703833558207835), 'n_est:40 depth:8 max_f:0.18371681153444983': np.float64(0.7038796322061092), 'n_est:20 depth:8 max_f:0.15749013123685915': np.float64(0.7039902389089353), 'n_est:50 depth:8 max_f:0.09921256574801249': np.float64(0.7041041795230352), 'n_est:10 depth:9 max_f:0.18371681153444983': np.float64(0.7043857251396728), 'n_est:30 depth:5 max_f:0.15749013123685915': np.float64(0.7043952717924132), 'n_est:20 depth:9 max_f:0.15749013123685915': np.float64(0.7044949757770582), 'n_est:10 depth:5 max_f:0.25': np.float64(0.7045184569257239), 'n_est:30 depth:8 max_f:0.18371681153444983': np.float64(0.7046753414925468), 'n_est:50 depth:9 max_f:0.13500746736153826': np.float64(0.7047057290437369), 'n_est:10 depth:9 max_f:0.1157343390359113': np.float64(0.7047700115025218), 'n_est:40 depth:9 max_f:0.18371681153444983': np.float64(0.7048355672787397), 'n_est:10 depth:6 max_f:0.09921256574801249': np.float64(0.7053211972313801), 'n_est:50 depth:8 max_f:0.15749013123685915': np.float64(0.7054248928299454), 'n_est:40 depth:5 max_f:0.13500746736153826': np.float64(0.7055683776675775), 'n_est:10 depth:9 max_f:0.09921256574801249': np.float64(0.7055856789912598), 'n_est:40 depth:5 max_f:0.09921256574801249': np.float64(0.7056386788660898), 'n_est:40 depth:8 max_f:0.2143109957132682': np.float64(0.7056869329283566), 'n_est:40 depth:9 max_f:0.09921256574801249': np.float64(0.7057568223325547), 'n_est:20 depth:8 max_f:0.13500746736153826': np.float64(0.705898380006919), 'n_est:20 depth:7 max_f:0.2143109957132682': np.float64(0.706116131536811), 'n_est:50 depth:7 max_f:0.15749013123685915': np.float64(0.7062420142426082), 'n_est:20 depth:7 max_f:0.08504937501089857': np.float64(0.7064856890516376), 'n_est:40 depth:7 max_f:0.1157343390359113': np.float64(0.7064918502158652), 'n_est:30 depth:6 max_f:0.15749013123685915': np.float64(0.7065534355619596), 'n_est:40 depth:5 max_f:0.2143109957132682': np.float64(0.7066784280147023), 'n_est:40 depth:6 max_f:0.18371681153444983': np.float64(0.706705331838772), 'n_est:50 depth:7 max_f:0.2143109957132682': np.float64(0.7067336541620197), 'n_est:30 depth:7 max_f:0.15749013123685915': np.float64(0.7069153317278385), 'n_est:30 depth:8 max_f:0.25': np.float64(0.7069993810527735), 'n_est:50 depth:5 max_f:0.18371681153444983': np.float64(0.7072073699003588), 'n_est:10 depth:5 max_f:0.15749013123685915': np.float64(0.7072186834164906), 'n_est:40 depth:9 max_f:0.13500746736153826': np.float64(0.7076947558795533), 'n_est:20 depth:5 max_f:0.2143109957132682': np.float64(0.7077025751733068), 'n_est:40 depth:9 max_f:0.25': np.float64(0.7077880287858445), 'n_est:30 depth:8 max_f:0.15749013123685915': np.float64(0.708290602411408), 'n_est:40 depth:5 max_f:0.25': np.float64(0.7085881127323611), 'n_est:20 depth:7 max_f:0.1157343390359113': np.float64(0.7086828726180523), 'n_est:40 depth:9 max_f:0.2143109957132682': np.float64(0.7087198077430363), 'n_est:30 depth:9 max_f:0.2143109957132682': np.float64(0.7087857987630045), 'n_est:30 depth:6 max_f:0.1157343390359113': np.float64(0.7088132687819964), 'n_est:10 depth:7 max_f:0.2143109957132682': np.float64(0.7088756620708674), 'n_est:20 depth:6 max_f:0.2143109957132682': np.float64(0.7090191622151084), 'n_est:50 depth:9 max_f:0.25': np.float64(0.7093911724647166), 'n_est:20 depth:9 max_f:0.1157343390359113': np.float64(0.709410792013542), 'n_est:40 depth:7 max_f:0.15749013123685915': np.float64(0.7094631556103611), 'n_est:30 depth:7 max_f:0.2143109957132682': np.float64(0.7095768923319623), 'n_est:10 depth:7 max_f:0.13500746736153826': np.float64(0.7096296427461548), 'n_est:20 depth:7 max_f:0.18371681153444983': np.float64(0.7097238416124653), 'n_est:20 depth:8 max_f:0.2143109957132682': np.float64(0.7097289606030384), 'n_est:50 depth:7 max_f:0.13500746736153826': np.float64(0.7097649486182156), 'n_est:30 depth:6 max_f:0.2143109957132682': np.float64(0.7099164911429444), 'n_est:30 depth:5 max_f:0.13500746736153826': np.float64(0.7099299849252765), 'n_est:40 depth:6 max_f:0.25': np.float64(0.7100887499731778), 'n_est:30 depth:9 max_f:0.25': np.float64(0.7101052529321366), 'n_est:40 depth:8 max_f:0.15749013123685915': np.float64(0.7101366390875565), 'n_est:50 depth:5 max_f:0.2143109957132682': np.float64(0.7104011445687577), 'n_est:40 depth:8 max_f:0.13500746736153826': np.float64(0.7104152361016605), 'n_est:50 depth:6 max_f:0.18371681153444983': np.float64(0.7104247963679251), 'n_est:20 depth:8 max_f:0.25': np.float64(0.7104345499451923), 'n_est:20 depth:9 max_f:0.09921256574801249': np.float64(0.7104582780496937), 'n_est:10 depth:6 max_f:0.13500746736153826': np.float64(0.7104725071848096), 'n_est:30 depth:5 max_f:0.2143109957132682': np.float64(0.7107231955560807), 'n_est:50 depth:9 max_f:0.18371681153444983': np.float64(0.7107750549778374), 'n_est:50 depth:7 max_f:0.25': np.float64(0.7112899639344646), 'n_est:20 depth:5 max_f:0.15749013123685915': np.float64(0.7115390259915848), 'n_est:20 depth:7 max_f:0.25': np.float64(0.7116332285053206), 'n_est:50 depth:6 max_f:0.15749013123685915': np.float64(0.7116648767387493), 'n_est:40 depth:5 max_f:0.15749013123685915': np.float64(0.7118303202099141), 'n_est:30 depth:6 max_f:0.18371681153444983': np.float64(0.7120900751672902), 'n_est:20 depth:5 max_f:0.18371681153444983': np.float64(0.7123710002835386), 'n_est:10 depth:9 max_f:0.25': np.float64(0.7124390699464713), 'n_est:50 depth:9 max_f:0.2143109957132682': np.float64(0.7129110269244812), 'n_est:20 depth:6 max_f:0.18371681153444983': np.float64(0.7131045980100275), 'n_est:10 depth:7 max_f:0.15749013123685915': np.float64(0.713168493032188), 'n_est:50 depth:5 max_f:0.25': np.float64(0.7135290077016794), 'n_est:30 depth:7 max_f:0.18371681153444983': np.float64(0.7135494714345394), 'n_est:20 depth:6 max_f:0.25': np.float64(0.7139123062868236), 'n_est:40 depth:7 max_f:0.18371681153444983': np.float64(0.7139773305978415), 'n_est:20 depth:5 max_f:0.25': np.float64(0.714059703417377), 'n_est:20 depth:7 max_f:0.15749013123685915': np.float64(0.7141866009891131), 'n_est:30 depth:5 max_f:0.25': np.float64(0.7142862190939183), 'n_est:50 depth:8 max_f:0.25': np.float64(0.714659540513134), 'n_est:40 depth:8 max_f:0.25': np.float64(0.7150324082737077), 'n_est:40 depth:9 max_f:0.15749013123685915': np.float64(0.7151684314459017), 'n_est:40 depth:6 max_f:0.2143109957132682': np.float64(0.7155190240363534), 'n_est:10 depth:8 max_f:0.25': np.float64(0.7155249152888096), 'n_est:40 depth:5 max_f:0.18371681153444983': np.float64(0.7155263564952559), 'n_est:50 depth:6 max_f:0.25': np.float64(0.7155549020698915), 'n_est:30 depth:6 max_f:0.25': np.float64(0.7157785519741264), 'n_est:50 depth:8 max_f:0.2143109957132682': np.float64(0.7162870733265474), 'n_est:50 depth:6 max_f:0.2143109957132682': np.float64(0.7170463831208782), 'n_est:30 depth:8 max_f:0.2143109957132682': np.float64(0.7172226796119254), 'n_est:40 depth:7 max_f:0.25': np.float64(0.7173978393265135), 'n_est:50 depth:7 max_f:0.18371681153444983': np.float64(0.7176523277530751), 'n_est:40 depth:7 max_f:0.2143109957132682': np.float64(0.7192012348535229), 'n_est:10 depth:6 max_f:0.25': np.float64(0.7217477790556482), 'n_est:10 depth:7 max_f:0.25': np.float64(0.7219692550106867), 'n_est:30 depth:7 max_f:0.25': np.float64(0.7236320635800605), 'n_est:30 depth:5 max_f:0.18371681153444983': np.float64(0.7240398305661998), 'n_est:30 depth:6 max_f:0.13500746736153826': np.float64(0.724238465480465), 'n_est:50 depth:6 max_f:0.13500746736153826': np.float64(0.7259068760281358), 'n_est:50 depth:5 max_f:0.15749013123685915': np.float64(0.7265850524630884), 'n_est:40 depth:6 max_f:0.15749013123685915': np.float64(0.7266115385210753), 
'n_est:10 depth:6 max_f:0.18371681153444983': np.float64(0.7361341373640897)}
{'n_est:10 depth:9 max_f:0.2143109957132682': np.float64(0.2415443907702426), 'n_est:10 depth:7 max_f:0.18371681153444983': np.float64(0.42067977116563016), 'n_est:10 depth:5 max_f:0.0625': np.float64(0.5084421859905752), 'n_est:10 depth:7 max_f:0.09921256574801249': np.float64(0.5158162472120857), 'n_est:10 depth:8 max_f:0.0625': np.float64(0.5420011465529042), 'n_est:10 depth:6 max_f:0.0625': np.float64(0.5441667820338862), 'n_est:10 depth:9 max_f:0.0729080649735073': np.float64(0.5593029382137849), 'n_est:10 depth:8 max_f:0.2143109957132682': np.float64(0.5661169128369525), 'n_est:10 depth:8 max_f:0.08504937501089857': np.float64(0.5727882010276046), 'n_est:20 depth:7 max_f:0.0729080649735073': np.float64(0.5794588207531473), 'n_est:10 depth:9 max_f:0.15749013123685915': np.float64(0.5845331209932736), 'n_est:10 depth:7 max_f:0.08504937501089857': np.float64(0.585690213886251), 'n_est:40 depth:6 max_f:0.0625': np.float64(0.5873384955762787), 'n_est:10 depth:5 max_f:0.0729080649735073': np.float64(0.5886767772606181), 'n_est:10 depth:6 max_f:0.0729080649735073': np.float64(0.5961950675734462), 'n_est:40 depth:5 max_f:0.0729080649735073': np.float64(0.5965520437854703), 'n_est:30 depth:8 max_f:0.0625': np.float64(0.5990889300443272), 'n_est:10 depth:6 max_f:0.2143109957132682': np.float64(0.6001621180590535), 'n_est:10 depth:5 max_f:0.08504937501089857': np.float64(0.6039948849069869), 'n_est:10 depth:9 max_f:0.25': np.float64(0.6082753248877267), 'n_est:20 depth:6 max_f:0.0729080649735073': np.float64(0.6086050079127481), 'n_est:10 depth:8 max_f:0.13500746736153826': np.float64(0.6099623029036545), 'n_est:30 depth:8 max_f:0.0729080649735073': np.float64(0.6107003995848042), 'n_est:10 depth:9 max_f:0.08504937501089857': np.float64(0.6114105785921295), 'n_est:20 depth:5 max_f:0.1157343390359113': np.float64(0.612071765669933), 'n_est:30 depth:7 max_f:0.0729080649735073': np.float64(0.6131282990228193), 'n_est:20 depth:6 max_f:0.0625': np.float64(0.6158723051635164), 'n_est:10 depth:7 max_f:0.25': np.float64(0.6165358892505957), 'n_est:10 depth:5 max_f:0.2143109957132682': np.float64(0.6169889387528013), 'n_est:20 depth:5 max_f:0.0625': np.float64(0.6172800936985178), 'n_est:10 depth:7 max_f:0.2143109957132682': np.float64(0.6178345665555287), 'n_est:30 depth:9 max_f:0.0625': np.float64(0.6180761833362319), 'n_est:50 depth:9 max_f:0.0729080649735073': np.float64(0.6182380706053139), 'n_est:20 depth:8 max_f:0.18371681153444983': np.float64(0.6183767456884834), 'n_est:10 depth:8 max_f:0.25': np.float64(0.6190456955897033), 'n_est:10 depth:8 max_f:0.0729080649735073': np.float64(0.6197669812772122), 'n_est:30 depth:5 max_f:0.25': np.float64(0.6197805114948074), 'n_est:10 depth:9 max_f:0.13500746736153826': np.float64(0.6202907094861309), 'n_est:20 depth:9 max_f:0.25': np.float64(0.6203627967507833), 'n_est:40 depth:9 max_f:0.0729080649735073': np.float64(0.6204661553492954), 'n_est:20 depth:7 max_f:0.0625': np.float64(0.6208356047569923), 'n_est:10 depth:9 max_f:0.0625': np.float64(0.6210309639415521), 'n_est:10 depth:6 max_f:0.25': np.float64(0.6217271814385538), 'n_est:20 depth:9 max_f:0.09921256574801249': np.float64(0.6231959546825545), 'n_est:30 depth:6 max_f:0.0625': np.float64(0.623202247172143), 'n_est:50 depth:5 max_f:0.0625': np.float64(0.6236024466996549), 'n_est:20 depth:7 max_f:0.18371681153444983': np.float64(0.6236959981033845), 'n_est:40 depth:5 max_f:0.25': np.float64(0.6238645223144794), 'n_est:20 depth:5 max_f:0.25': np.float64(0.6239584078565711), 'n_est:30 depth:9 max_f:0.18371681153444983': np.float64(0.6242416694877129), 'n_est:20 depth:8 max_f:0.15749013123685915': np.float64(0.6247041671510717), 'n_est:20 depth:6 max_f:0.1157343390359113': np.float64(0.6248765817924196), 'n_est:10 depth:6 max_f:0.13500746736153826': np.float64(0.6249053361464227), 'n_est:20 depth:6 max_f:0.18371681153444983': np.float64(0.6249412066600515), 'n_est:20 depth:5 max_f:0.0729080649735073': np.float64(0.6252524084862877), 'n_est:30 depth:7 max_f:0.2143109957132682': np.float64(0.6256302107606386), 'n_est:10 depth:5 max_f:0.25': np.float64(0.6260876539405744), 'n_est:30 depth:5 max_f:0.08504937501089857': np.float64(0.6262144802725895), 'n_est:50 depth:9 max_f:0.0625': np.float64(0.6263675925319685), 'n_est:40 depth:6 max_f:0.25': np.float64(0.626558807512841), 'n_est:20 depth:7 max_f:0.25': np.float64(0.6266795147576414), 'n_est:50 depth:6 max_f:0.25': np.float64(0.6267843137276836), 'n_est:10 depth:8 max_f:0.1157343390359113': np.float64(0.6270297777957543), 'n_est:50 depth:7 max_f:0.2143109957132682': np.float64(0.6275200289354341), 'n_est:10 depth:8 max_f:0.18371681153444983': np.float64(0.62767750613026), 'n_est:10 depth:8 max_f:0.15749013123685915': np.float64(0.6279360387601297), 'n_est:10 depth:9 max_f:0.1157343390359113': np.float64(0.6283016844318937), 'n_est:20 depth:6 max_f:0.25': np.float64(0.6286933638435135), 'n_est:20 depth:9 max_f:0.0625': np.float64(0.628764023118164), 'n_est:20 depth:7 max_f:0.15749013123685915': np.float64(0.6291181563791135), 'n_est:20 depth:5 max_f:0.15749013123685915': np.float64(0.6293120111366735), 'n_est:50 depth:8 max_f:0.18371681153444983': np.float64(0.6293937502091824), 'n_est:50 depth:9 max_f:0.13500746736153826': np.float64(0.6294211929399688), 'n_est:30 depth:7 max_f:0.15749013123685915': np.float64(0.6294380242591537), 'n_est:20 depth:5 max_f:0.18371681153444983': np.float64(0.6294988447253851), 'n_est:40 depth:5 max_f:0.18371681153444983': np.float64(0.6298112695622671), 'n_est:40 depth:9 max_f:0.2143109957132682': np.float64(0.6298458696604772), 'n_est:30 depth:9 max_f:0.25': np.float64(0.6298644831131899), 'n_est:50 depth:6 max_f:0.2143109957132682': np.float64(0.6301645030836771), 'n_est:30 depth:8 max_f:0.09921256574801249': np.float64(0.6302961459179144), 'n_est:50 depth:5 max_f:0.25': np.float64(0.6304526280462999), 'n_est:30 depth:6 max_f:0.18371681153444983': np.float64(0.6305276685003331), 'n_est:20 depth:6 max_f:0.13500746736153826': np.float64(0.630529297410781), 'n_est:40 depth:8 max_f:0.15749013123685915': np.float64(0.6305651813397712), 'n_est:40 depth:7 max_f:0.15749013123685915': np.float64(0.6305680826351281), 'n_est:40 depth:8 max_f:0.2143109957132682': np.float64(0.6306168601036384), 'n_est:30 depth:6 max_f:0.08504937501089857': np.float64(0.6307322680027858), 'n_est:30 depth:8 max_f:0.13500746736153826': np.float64(0.6308363086006281), 'n_est:20 depth:5 max_f:0.2143109957132682': np.float64(0.6308829239436763), 'n_est:40 depth:7 max_f:0.18371681153444983': np.float64(0.630883424261896), 'n_est:20 depth:8 max_f:0.2143109957132682': np.float64(0.6309604757220475), 'n_est:30 depth:7 max_f:0.25': np.float64(0.631008032741307), 'n_est:40 depth:6 max_f:0.18371681153444983': np.float64(0.6312749768216201), 'n_est:10 depth:5 max_f:0.18371681153444983': np.float64(0.6313024760581865), 'n_est:50 depth:5 max_f:0.18371681153444983': np.float64(0.6314610951060636), 'n_est:10 depth:9 max_f:0.18371681153444983': np.float64(0.6314692610003396), 'n_est:10 depth:7 max_f:0.0625': np.float64(0.6314790533737604), 'n_est:30 depth:5 max_f:0.1157343390359113': np.float64(0.6315137288247994), 'n_est:30 depth:6 max_f:0.13500746736153826': np.float64(0.6317008573309013), 'n_est:20 depth:9 max_f:0.15749013123685915': np.float64(0.631984020444065), 'n_est:50 depth:5 max_f:0.09921256574801249': np.float64(0.6319862421432644), 'n_est:30 depth:8 max_f:0.25': np.float64(0.6321162938256439), 'n_est:20 depth:8 max_f:0.09921256574801249': np.float64(0.6321830878675405), 'n_est:40 depth:6 max_f:0.0729080649735073': np.float64(0.632197736791272), 'n_est:30 depth:9 max_f:0.15749013123685915': np.float64(0.6322017268181072), 'n_est:20 depth:7 max_f:0.13500746736153826': np.float64(0.6322404537673516), 'n_est:50 depth:5 max_f:0.2143109957132682': np.float64(0.632383780866205), 'n_est:20 depth:9 max_f:0.0729080649735073': np.float64(0.6325110809419645), 'n_est:50 depth:7 max_f:0.09921256574801249': np.float64(0.632720238579393), 'n_est:20 depth:7 max_f:0.08504937501089857': np.float64(0.6328273466783211), 'n_est:30 depth:8 max_f:0.1157343390359113': np.float64(0.632829489752573), 'n_est:40 depth:7 max_f:0.25': np.float64(0.6328506283788273), 'n_est:20 depth:7 max_f:0.1157343390359113': np.float64(0.6329223869236329), 'n_est:40 depth:7 max_f:0.13500746736153826': np.float64(0.6331546700352595), 'n_est:20 depth:5 max_f:0.08504937501089857': np.float64(0.6331918367890599), 'n_est:40 depth:6 max_f:0.2143109957132682': np.float64(0.6333205287902043), 'n_est:30 depth:5 max_f:0.0729080649735073': np.float64(0.6333279852563583), 'n_est:30 depth:7 max_f:0.18371681153444983': np.float64(0.6335553638123661), 'n_est:40 depth:9 max_f:0.25': np.float64(0.6335649850425731), 'n_est:10 depth:6 max_f:0.15749013123685915': np.float64(0.6335883906396726), 'n_est:30 depth:6 max_f:0.09921256574801249': np.float64(0.6336085735412426), 'n_est:30 depth:9 max_f:0.2143109957132682': np.float64(0.63369233684203), 'n_est:20 depth:5 max_f:0.09921256574801249': np.float64(0.6337750483721674), 'n_est:10 depth:7 max_f:0.1157343390359113': np.float64(0.6337797968090625), 'n_est:40 depth:8 max_f:0.18371681153444983': np.float64(0.6339801848191327), 'n_est:20 depth:7 max_f:0.09921256574801249': np.float64(0.6341360085920194), 'n_est:50 depth:6 max_f:0.13500746736153826': np.float64(0.6341797395522202), 'n_est:20 depth:7 max_f:0.2143109957132682': np.float64(0.6342104913829509), 'n_est:40 depth:5 max_f:0.2143109957132682': np.float64(0.634215789705993), 'n_est:40 depth:8 max_f:0.25': np.float64(0.6342872482424465), 'n_est:30 depth:7 max_f:0.1157343390359113': np.float64(0.6343831713265123), 'n_est:50 depth:6 max_f:0.0625': np.float64(0.634412018019443), 'n_est:50 depth:8 max_f:0.08504937501089857': np.float64(0.6344304677157031), 'n_est:50 depth:8 max_f:0.25': np.float64(0.6344860342244126), 'n_est:20 depth:9 max_f:0.2143109957132682': np.float64(0.6346052468995227), 'n_est:10 depth:7 max_f:0.13500746736153826': np.float64(0.6346571460056909), 'n_est:10 depth:5 max_f:0.1157343390359113': np.float64(0.6347200687566539), 'n_est:30 depth:7 max_f:0.13500746736153826': np.float64(0.6347656270049589), 'n_est:20 depth:8 max_f:0.13500746736153826': np.float64(0.6348350816538127), 'n_est:50 depth:7 max_f:0.25': np.float64(0.6348802751553593), 'n_est:50 depth:9 max_f:0.2143109957132682': np.float64(0.6352007214556241), 'n_est:30 depth:5 max_f:0.18371681153444983': np.float64(0.6352159544686349), 'n_est:20 depth:9 max_f:0.08504937501089857': np.float64(0.6352609933606672), 'n_est:20 depth:9 max_f:0.18371681153444983': np.float64(0.6353401505776324), 'n_est:50 depth:7 max_f:0.0625': np.float64(0.6354032977473322), 'n_est:30 depth:9 max_f:0.0729080649735073': np.float64(0.6354835237927348), 'n_est:50 depth:7 max_f:0.15749013123685915': np.float64(0.6357440904861327), 'n_est:10 depth:6 max_f:0.18371681153444983': np.float64(0.6358345003424645), 'n_est:20 depth:9 max_f:0.13500746736153826': np.float64(0.6359163973585766), 'n_est:40 depth:5 max_f:0.15749013123685915': np.float64(0.6359413501983138), 'n_est:40 depth:9 max_f:0.0625': np.float64(0.6361052020595975), 'n_est:20 depth:6 max_f:0.15749013123685915': np.float64(0.6361519978684071), 'n_est:30 depth:9 max_f:0.13500746736153826': np.float64(0.6362234413137937), 'n_est:30 depth:5 max_f:0.2143109957132682': np.float64(0.6362377305245164), 'n_est:30 depth:9 max_f:0.1157343390359113': np.float64(0.6363095705355755), 'n_est:30 depth:6 max_f:0.25': np.float64(0.6363168005610446), 'n_est:20 depth:9 max_f:0.1157343390359113': np.float64(0.6363824713505781), 'n_est:40 depth:7 max_f:0.2143109957132682': np.float64(0.6364628748085777), 'n_est:40 depth:9 max_f:0.1157343390359113': np.float64(0.6365511479742536), 'n_est:30 depth:9 max_f:0.09921256574801249': np.float64(0.6367343512877405), 'n_est:50 depth:7 max_f:0.18371681153444983': np.float64(0.636748180735267), 'n_est:40 depth:7 max_f:0.0625': np.float64(0.6368219730690922), 'n_est:50 depth:6 max_f:0.1157343390359113': np.float64(0.6368475838038841), 'n_est:30 depth:6 max_f:0.2143109957132682': np.float64(0.6369955113288795), 'n_est:40 depth:5 max_f:0.0625': np.float64(0.637011079768816), 'n_est:50 depth:6 max_f:0.18371681153444983': np.float64(0.6370545284160072), 'n_est:50 depth:8 max_f:0.2143109957132682': np.float64(0.6370656954881186), 'n_est:20 depth:8 max_f:0.25': np.float64(0.6372654887297858), 'n_est:40 depth:6 max_f:0.15749013123685915': np.float64(0.6373284713065372), 'n_est:50 depth:8 max_f:0.1157343390359113': np.float64(0.637337223664928), 'n_est:50 depth:6 max_f:0.15749013123685915': np.float64(0.6373604364910196), 'n_est:40 depth:6 max_f:0.1157343390359113': np.float64(0.6374120005969477), 'n_est:50 depth:8 max_f:0.13500746736153826': np.float64(0.6375370209366847), 'n_est:10 depth:5 max_f:0.13500746736153826': np.float64(0.6376752476607963), 'n_est:10 depth:8 max_f:0.09921256574801249': np.float64(0.6377342657467313), 'n_est:40 depth:6 max_f:0.08504937501089857': np.float64(0.637797345262642), 'n_est:30 depth:8 max_f:0.2143109957132682': np.float64(0.6377984184244383), 'n_est:40 depth:5 max_f:0.1157343390359113': np.float64(0.6379964146428647), 'n_est:10 depth:5 max_f:0.15749013123685915': np.float64(0.6380035346526667), 'n_est:40 depth:7 max_f:0.08504937501089857': np.float64(0.638181920284459), 'n_est:30 depth:7 max_f:0.09921256574801249': np.float64(0.6382717542069873), 'n_est:30 depth:7 max_f:0.0625': np.float64(0.6383087134478198), 'n_est:50 depth:7 max_f:0.13500746736153826': np.float64(0.6383490597721925), 'n_est:40 depth:8 max_f:0.0625': np.float64(0.6385596783268808), 'n_est:40 depth:8 max_f:0.09921256574801249': np.float64(0.6386183600104107), 'n_est:50 depth:5 max_f:0.08504937501089857': np.float64(0.6390907304564802), 'n_est:20 depth:8 max_f:0.1157343390359113': np.float64(0.6391978748741318), 'n_est:40 depth:9 max_f:0.09921256574801249': np.float64(0.6392095178714114), 'n_est:30 depth:8 max_f:0.18371681153444983': np.float64(0.6392954614913495), 'n_est:50 depth:6 max_f:0.09921256574801249': np.float64(0.639736386747231), 'n_est:40 depth:8 max_f:0.08504937501089857': np.float64(0.6399475328108553), 'n_est:50 depth:9 max_f:0.18371681153444983': np.float64(0.6399998472536372), 'n_est:30 depth:8 max_f:0.15749013123685915': np.float64(0.6400398986039881), 'n_est:50 depth:8 max_f:0.0729080649735073': np.float64(0.6401193097645381), 'n_est:10 depth:6 max_f:0.1157343390359113': np.float64(0.6402670686679184), 'n_est:10 depth:5 max_f:0.09921256574801249': np.float64(0.6405361762282453), 'n_est:50 depth:9 max_f:0.09921256574801249': np.float64(0.6406964864226731), 'n_est:20 depth:8 max_f:0.08504937501089857': np.float64(0.6408276164187232), 'n_est:30 depth:5 max_f:0.0625': np.float64(0.6412866813606191), 'n_est:40 depth:5 max_f:0.08504937501089857': np.float64(0.6414390069024305), 'n_est:40 depth:9 max_f:0.15749013123685915': np.float64(0.6417337189758383), 'n_est:50 depth:9 max_f:0.25': np.float64(0.641776891884682), 'n_est:40 depth:5 max_f:0.13500746736153826': np.float64(0.6419909986695468), 'n_est:20 depth:6 max_f:0.2143109957132682': np.float64(0.6421630149892228), 'n_est:50 depth:7 max_f:0.0729080649735073': np.float64(0.6421889690853151), 'n_est:10 depth:7 max_f:0.15749013123685915': np.float64(0.6422943081033867), 'n_est:50 depth:6 max_f:0.08504937501089857': np.float64(0.6424040633569448), 'n_est:50 depth:5 max_f:0.1157343390359113': np.float64(0.6425030994660851), 'n_est:20 depth:8 max_f:0.0729080649735073': np.float64(0.6431174016774581), 'n_est:20 depth:6 max_f:0.09921256574801249': np.float64(0.6432350532603557), 'n_est:40 depth:8 max_f:0.0729080649735073': np.float64(0.6433496675484491), 'n_est:40 depth:8 max_f:0.1157343390359113': np.float64(0.6433514289406677), 'n_est:40 depth:9 max_f:0.13500746736153826': np.float64(0.6434645963144721), 'n_est:30 depth:8 max_f:0.08504937501089857': np.float64(0.6439306544822159), 'n_est:40 depth:7 max_f:0.09921256574801249': np.float64(0.6440543259661669), 'n_est:50 depth:7 max_f:0.1157343390359113': np.float64(0.6440808541182612), 'n_est:50 depth:5 max_f:0.0729080649735073': np.float64(0.6444354019841352), 'n_est:40 depth:9 max_f:0.08504937501089857': np.float64(0.6444936040969194), 'n_est:30 depth:5 max_f:0.15749013123685915': np.float64(0.6448036835815159), 'n_est:50 depth:9 max_f:0.08504937501089857': np.float64(0.6449810492171689), 'n_est:50 depth:5 max_f:0.13500746736153826': np.float64(0.6450272633334357), 'n_est:30 depth:6 max_f:0.15749013123685915': np.float64(0.6454157145482786), 'n_est:20 depth:5 max_f:0.13500746736153826': np.float64(0.6454224694223878), 'n_est:40 depth:7 max_f:0.1157343390359113': np.float64(0.6454473181614335), 'n_est:50 depth:9 max_f:0.15749013123685915': np.float64(0.6455757742533683), 'n_est:40 depth:8 max_f:0.13500746736153826': np.float64(0.6459283740915198), 'n_est:20 depth:6 max_f:0.08504937501089857': np.float64(0.6461146048823653), 'n_est:50 depth:8 max_f:0.0625': np.float64(0.6461628742359129), 'n_est:50 depth:8 max_f:0.15749013123685915': np.float64(0.6465022186492531), 'n_est:40 depth:9 max_f:0.18371681153444983': np.float64(0.6466028588589305), 'n_est:30 depth:5 max_f:0.09921256574801249': np.float64(0.6467533203943729), 'n_est:40 depth:6 max_f:0.09921256574801249': np.float64(0.647048812502595), 'n_est:30 depth:5 max_f:0.13500746736153826': np.float64(0.6472107591860521), 'n_est:50 depth:6 max_f:0.0729080649735073': np.float64(0.6475612089411227), 'n_est:50 depth:5 max_f:0.15749013123685915': np.float64(0.6483471735379779), 'n_est:50 depth:8 max_f:0.09921256574801249': np.float64(0.648584790780468), 'n_est:50 depth:9 max_f:0.1157343390359113': np.float64(0.6490596192793027), 'n_est:40 depth:7 max_f:0.0729080649735073': np.float64(0.64910872054864), 'n_est:30 depth:6 max_f:0.1157343390359113': np.float64(0.6494806416477993), 'n_est:40 depth:5 max_f:0.09921256574801249': np.float64(0.6500852706359073), 'n_est:10 depth:9 max_f:0.09921256574801249': np.float64(0.6516009242168388), 'n_est:50 depth:7 max_f:0.08504937501089857': np.float64(0.6528233933334394), 'n_est:40 depth:6 max_f:0.13500746736153826': np.float64(0.6528818647771517), 'n_est:30 depth:9 max_f:0.08504937501089857': np.float64(0.6539666387749263), 'n_est:10 depth:6 max_f:0.09921256574801249': np.float64(0.6545797289875291), 'n_est:20 depth:8 max_f:0.0625': np.float64(0.654972349477515), 'n_est:30 depth:6 max_f:0.0729080649735073': np.float64(0.6572437499222412), 'n_est:10 depth:6 max_f:0.08504937501089857': np.float64(0.6578124858207752), 'n_est:30 depth:7 max_f:0.08504937501089857': np.float64(0.6650615317676494), 
'n_est:10 depth:7 max_f:0.0729080649735073': np.float64(0.6681468051139976)}

"""