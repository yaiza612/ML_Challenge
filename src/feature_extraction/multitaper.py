import mne
import numpy as np
from pathlib import Path

from mne.time_frequency import tfr_array_multitaper

from src.data_loading.loader import reshape_array_into_windows

ROOT_PATH = Path("../../data/train")
TEST_PATH = Path("../../data/test")

training_data = [np.load(ROOT_PATH / f"data_{i}.npy") for i in range(4)]
testing_data = [np.load(TEST_PATH / f"data_{i}.npy") for i in range(4, 6)]
full_data = training_data + testing_data


for data_index in range(1, 6):
    data = full_data[data_index]
    reshaped = reshape_array_into_windows(data, 250, 2)
    reshaped_final = np.swapaxes(reshaped, 0, 1)
    mean_power_list = []
    std_power_list = []
    amplitude_power_list = []
    batch_size = 1000
    for i in range(0, reshaped_final.shape[0], batch_size):
        batch_data = reshaped_final[i:i+batch_size]
        sfreq = 250
        freqs = np.arange(0.5, 70.2, 0.5)
        n_cycles = freqs / 2
        time_bandwidth = 3
        power_values = tfr_array_multitaper(batch_data, freqs=freqs, n_cycles=n_cycles,
                                            n_jobs=5, time_bandwidth=time_bandwidth,
                                            use_fft=True, sfreq=sfreq, output="power")
        mean_power_features = np.mean(power_values, axis=-1)
        std_power_features = np.std(power_values, axis=-1)
        amplitude_power_features = np.max(power_values, axis=-1) - np.min(power_values, axis=-1)
        del power_values
        mean_power_list.append(mean_power_features)
        std_power_list.append(std_power_features)
        amplitude_power_list.append(amplitude_power_features)
    stacked_means = np.concatenate(mean_power_list)
    stacked_stds = np.concatenate(std_power_list)
    stacked_amplitudes = np.concatenate(amplitude_power_list)
    np.save(f"avg_power_multitaper_{data_index}.npy", stacked_means)
    np.save(f"std_power_multitaper_{data_index}.npy", stacked_stds)
    np.save(f"amplitude_power_multitaper_{data_index}.npy", stacked_amplitudes)

