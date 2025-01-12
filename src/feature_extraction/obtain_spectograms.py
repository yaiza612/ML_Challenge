import os
import numpy as np
import pandas as pd
from scipy.signal import square, ShortTimeFFT
from scipy.signal.windows import gaussian
from src.data_loading.loader import load_data_as_windows
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


def compute_time_frequency_domain_features(segment_data):
    """
    Extracts time-frequency domain features from EEG data for each segment using STFT.

    Parameters:
    - segment_data: np.ndarray - EEG data for one segment, shape (channels, samples).
    """


    # Process each channel in the segment
    individual_channel_datas = []
    for ch in range(segment_data.shape[0]):
        channel_data = segment_data[ch, :]
        # 250Hz, 2 second signal
        g_std = 8  # standard deviation for Gaussian window in samples
        win = gaussian(15, std=g_std, sym=True)
        SFT = ShortTimeFFT(win, hop=16, fs=250, mfft=31, scale_to='psd')
        Sx2 = SFT.spectrogram(channel_data)
        Sx2 = np.log10(Sx2)
        individual_channel_datas.append(Sx2)

    # merged channels
    for first in range(5):
        for second in range(first):
            channel_data = segment_data[first, :] - segment_data[second, :]
            g_std = 8  # standard deviation for Gaussian window in samples
            win = gaussian(15, std=g_std, sym=True)
            SFT = ShortTimeFFT(win, hop=16, fs=250, mfft=31, scale_to='psd')
            Sx2 = SFT.spectrogram(channel_data)
            Sx2 = np.log10(Sx2)
            individual_channel_datas.append(Sx2)
    merged = np.stack(individual_channel_datas)
    merged = np.array(merged, dtype=np.float32)
    return merged


if __name__ == "__main__":

    path_for_features = "../../data/Engineered_features/"

    for i in range(6):
        filename = f'{path_for_features}eeg_spectrogram_features_data_{i}.npy'
        if not os.path.isfile(filename):
            data, labels = load_data_as_windows(i)
            num_samples = data.shape[1]
            all_segment_features = []
            filename = f'{path_for_features}eeg_spectrogram_features_data_{i}.npy'

            for idx in tqdm(range(data.shape[1]), desc=f"Processing data of index {i} for frequency-time-domain features"):
                segment_data = data[:, idx, :]  # Shape: (5, 500)
                spectrogram_features = compute_time_frequency_domain_features(segment_data)

                all_segment_features.append(spectrogram_features)

            all_segment_features = np.stack(all_segment_features)
            np.save(filename, all_segment_features)
            print(f"Time-frequency features and labels saved to {filename}")
