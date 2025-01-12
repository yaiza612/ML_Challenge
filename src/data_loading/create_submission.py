import pandas as pd
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


def create_submission_from_flattened_preds(preds_1, preds_2, submission_name="submission"):
    results = []
    for record_number, pred in enumerate([preds_1, preds_2]):
        pred = np.array(pred)
        pred = np.reshape(pred, (5, len(pred) // 5), order="F")
        record_number += 4
        formatted_preds = format_array_to_target_format(pred,record_number)
        results.extend(formatted_preds)
    df = pd.DataFrame(results)
    df.to_csv(f"{submission_name}.csv", index=False)