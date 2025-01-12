from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from sklearn.metrics import cohen_kappa_score, f1_score
from sklearn.model_selection import train_test_split
import numpy as np

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
    df.to_csv("submission.csv", index=False)


ROOT_PATH = "../../data/Engineered_features/"
feature_types = ["frequency_domain", "nonlinear", "time_frequency_domain"]
for feature_type in feature_types:
    dfs_by_train_index = [pd.read_csv(f"{ROOT_PATH}eeg_{feature_type}_features_data_{train_index}.csv")
                          for train_index in range(4)]
    df = pd.concat(dfs_by_train_index, axis=0)
    #print(df.head())


    #df = df.drop(columns=["label_ch1", "label_ch2", "label_ch3", "label_ch4", "label_ch5"])
    split_by_channel = [df[df["channel"] == idx] for idx in range(1, 6)]
    labels_by_channel = [split_by_channel[idx - 1][f"label_ch{idx}"] for idx in range(1, 6)]
    split_by_channel_labels_removed = [df.drop(columns=["label_ch1", "label_ch2", "label_ch3", "label_ch4", "label_ch5"])
                                       for df in split_by_channel]

    stacked_features = pd.concat(split_by_channel_labels_removed, axis=0)
    stacked_labels = pd.concat(labels_by_channel, axis=0)

    X_train, X_test, y_train, y_test = train_test_split(stacked_features, stacked_labels, test_size=0.33, shuffle=False)
    print(f"Training performance for {feature_type}:")
    clf = RandomForestClassifier(n_estimators=10, max_depth=8)
    clf.fit(X=X_train, y=y_train)
    preds_train = clf.predict(X=X_train)
    print(clf.score(X=X_train, y=y_train))
    print(cohen_kappa_score(preds_train, y_train))
    print(f1_score(y_train, preds_train))

    print(f"Test performance for {feature_type}:")
    preds_test = clf.predict(X=X_test)
    print(clf.score(X=X_test, y=y_test))
    print(cohen_kappa_score(preds_test, y_test))
    print(f1_score(y_test, preds_test))

    print("-------------")
    preds_total = [clf.predict(pd.read_csv(f"{ROOT_PATH}eeg_{feature_type}_features_data_{test_index}.csv"))
                   for test_index in [4, 5]]
    #for df_test_path in ["eeg_time_frequency_domain_features_data_4.csv",
    #                     "eeg_time_frequency_domain_features_data_5.csv"]:
    #    df = pd.read_csv(f"{ROOT_PATH}{df_test_path}")
    #    preds = clf.predict(X=df)
    #    preds_total.append(preds)
    get_overall_results(preds_total[0], preds_total[1])

