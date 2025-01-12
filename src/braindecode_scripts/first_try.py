import numpy as np
from sklearn.metrics import cohen_kappa_score
from src.data_loading.loader import load_train_data_as_windows
from skorch.dataset import ValidSplit
from braindecode import EEGClassifier
import torch
import torch.nn as nn

from utils import create_dataloader, SigmoidBinaryKappaLoss, train_for_epochs

all_features, all_labels = [], []
for _ in range(4):
    features, labels = load_train_data_as_windows(train_idx=_)
    all_features.append(features)
    all_labels.append(labels)

features = np.concatenate(all_features, axis=1)
labels = np.concatenate(all_labels, axis=1)

print(features.shape)  # 5 channels, 15425 samples, 2 seconds @ 250Hz  (5, 15425, 500)
print(labels.shape)  # 5 labels (one per channel), 15425 samples
# let us reshape the features so we have it in the following order: num_samples, num_channels, seconds @ Hz  (15425, 5, 500)
features = np.swapaxes(features, 0, 1)
print(features.shape)

indices = np.random.permutation(labels.shape[1])
features = features[indices]
labels = labels[:, indices]

for label_idx in range(5):
    size = 1000
    # let us pretend we only have one label
    labels_selected = labels[label_idx, :]
    test_features = features[-size:, :, :]
    train_features = features[:size, :, :]
    test_labels = labels_selected[-size:]
    train_labels = labels_selected[:size]
    for x in [test_features, test_labels, train_features, train_labels]:
        print(x.shape)

    net = EEGClassifier(
        'EEGITNet',
        #module__final_conv_length='auto',
        max_epochs=30,
        train_split=ValidSplit(0.2),
        batch_size=8,
        # To train a neural network you need validation split, here, we use 20%.
    )

    #net.fit(X=train_features, y=train_labels)
    #preds = net.predict(X=test_features)
    #print(cohen_kappa_score(preds, test_labels))
    #preds_2 = net.predict(X=train_features)
    #print(cohen_kappa_score(preds_2, train_labels))
    train_loader = create_dataloader(features=train_features, labels=train_labels, batch_size=8)
    test_loader = create_dataloader(features=test_features, labels=test_labels, batch_size=8)

    optimizer = torch.optim.Adam(params=net.module_.parameters())
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.9)
    loss_prob = nn.BCELoss()
    loss_log = SigmoidBinaryKappaLoss()
    train_for_epochs(n_epochs=10, train_loader=train_loader, test_loader=test_loader, model=net.module_,
                     optimizer=optimizer, loss_fn=loss_prob, scheduler=scheduler)
