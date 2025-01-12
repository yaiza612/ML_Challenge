from pathlib import Path
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import cohen_kappa_score, f1_score
from braindecode.models.util import models_dict
from braindecode.models import EEGNetv1
from src.data_loading.loader import load_train_data_as_windows
from skorch.dataset import ValidSplit
from braindecode import EEGClassifier
import torch
import torch.nn as nn
from torch.nn import Module
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader, TensorDataset
from neural_nets import Net, SimpleNet
from torchinfo import summary
from utils import create_dataloader, SigmoidBinaryKappaLoss, train_one_epoch, validate_model

dataset_size = 800
all_features, all_labels = [], []
for _ in range(4):
    features, labels = load_train_data_as_windows(train_idx=_)
    all_features.append(features[:, :dataset_size])
    all_labels.append(labels[:, :dataset_size])

features = np.concatenate(all_features, axis=1)
labels = np.concatenate(all_labels, axis=1)

print(features.shape)
x = np.random.shuffle(list(range(dataset_size * 4)))
print(labels.shape)

features = np.squeeze(features[:, x, :], axis=1)
labels = np.squeeze(labels[:, x], axis=1)
print(features.shape)
print(labels.shape)
#print(features.shape)  # 5 channels, 15425 samples, 2 seconds @ 250Hz
#print(labels.shape)  # 5 labels (one per channel), 15425 samples
# let us reshape the features so we have it in the following order: num_samples, num_channels, seconds @ Hz
features = np.swapaxes(features, 0, 1)
#print(features.shape)


#print(model)




test_features = features[-dataset_size:, :, :]
train_features = features[:-dataset_size, :, :]
test_labels = labels[:, -dataset_size:]
train_labels = labels[:, :-dataset_size]
#print(train_features.shape)
#train_features = np.expand_dims(train_features, axis=1)
#print(train_features.shape)
# I want the train features to have 1 channel, height 5 and width 500
#print(train_labels.shape)
model_2 = EEGClassifier()
summary(model_2, input_size=(32, 1, 5, 500))
model = Net()
simple_model = SimpleNet()
#summary(model=model, input_size=(32, 5, 500))
#summary(model=simple_model, input_size=(32, 1, 5, 500))

train_loader = create_dataloader(features=train_features, labels=train_labels, batch_size=8)
test_loader = create_dataloader(features=test_features, labels=test_labels, batch_size=8)

optimizer = torch.optim.Adam(params=model.parameters())
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.9)
loss_prob = nn.BCELoss()
loss_log = SigmoidBinaryKappaLoss()



