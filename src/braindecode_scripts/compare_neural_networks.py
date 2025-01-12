import numpy as np
from braindecode.models.util import models_dict
from braindecode.models import EEGNetv1, EEGITNet, EEGInception, EEGResNet
from braindecode import EEGClassifier
import torch
import torch.nn as nn
from torch.nn import Module
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader, TensorDataset
from neural_nets import Net, SimpleNet
from torchinfo import summary


model_1 = EEGNetv1(n_chans=5,
    n_times=500,
    n_outputs=5,
)  # this is how to initialize the models
#for x in model_1.named_children():
#    print(x)
print(model_1)
print(model_1(torch.from_numpy(np.array(np.random.random(size=(1, 5, 500)), dtype=np.float32))))


model_1 = EEGITNet(n_chans=5,
    n_times=500,
    n_outputs=5,
)  # this is how to initialize the models
#for x in model_1.named_children():
#    print(x)
print(model_1)
print(model_1(torch.from_numpy(np.array(np.random.random(size=(1, 5, 500)), dtype=np.float32))))

model_1 = EEGInception(n_chans=5,
    n_times=500,
    n_outputs=5,
)  # this is how to initialize the models
#for x in model_1.named_children():
#    print(x)
print(model_1)
print(model_1(torch.from_numpy(np.array(np.random.random(size=(1, 5, 500)), dtype=np.float32))))

model_1 = EEGResNet(n_chans=5,
    n_times=500,
    n_outputs=5,
)  # this is how to initialize the models
#for x in model_1.named_children():
#    print(x)
print(model_1)
print(model_1(torch.from_numpy(np.array(np.random.random(size=(1, 5, 500)), dtype=np.float32))))

#model = Net()
#simple_model = SimpleNet()
#summary(model=model, input_size=(32, 5, 500))
#summary(model=simple_model, input_size=(32, 1, 5, 500))