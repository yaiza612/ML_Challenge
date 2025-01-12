import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd

y = pd.read_csv("../../data/sample.csv")
print(y.head)

for i in range(3):

    z = np.load(f"../../data/train/data_{i}.npy")
    print(z.shape)
    x = np.load(f"../../data/train/target_{i}.npy")
    print(x.shape)

    num_two_second_samples = x.shape[1]
    num_recordings_per_second = z.shape[1] / num_two_second_samples
    print(num_recordings_per_second)
print((6602015 + 4659937) / 500)
for i in range(4, 6):
    z = np.load(f"../../data/test/data_{i}.npy")
    print(z.shape)
"""
fig, ax = plt.subplots(nrows=5, ncols=1)

for train_dim, axs in zip(z, ax):
    axs.plot(train_dim[:num_recordings_per_second * 10])

plt.show()
"""