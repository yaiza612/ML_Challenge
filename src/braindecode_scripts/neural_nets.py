import torch
import torch.nn as nn
import torch.nn.functional as F
from braindecode.models.modules import Ensure4d
from braindecode.models.functions import squeeze_final_output
from einops.layers.torch import Rearrange

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = nn.Conv2d(1, 16, kernel_size=(2, 2), stride=(1, 1), padding="same")
        self.relu_1 = nn.ReLU()
        self.conv_2 = nn.Conv2d(16, 16, kernel_size=(2, 2), stride=(1, 1), padding="same")
        self.relu_2 = nn.ReLU()
        self.conv_3 = nn.Conv2d(16, 16, kernel_size=(2, 2), stride=(1, 1), padding="same")
        self.relu_3 = nn.ReLU()

    def forward(self, x):
        x = self.conv_1(x)
        x = self.relu_1(x)
        x = self.conv_2(x)
        x = self.relu_2(x)
        x = self.conv_3(x)
        x = self.relu_3(x)
        return torch.flatten(x, 1)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.ensuredims = Ensure4d()
        self.conv_1 = nn.Conv2d(5, 16, kernel_size=(1, 1), stride=(1, 1))
        self.bnorm_1 = nn.BatchNorm2d(num_features=16)
        self.elu_1 = nn.ELU()
        self.permute_1 = Rearrange("batch ch t 1 -> batch 1 ch t")
        self.drop_1 = nn.Dropout(p=0.25)
        self.conv_2 = nn.Conv2d(1, 4, kernel_size=(2, 32), stride=(1, 1), padding=(1, 0))
        self.bnorm_2 = nn.BatchNorm2d(num_features=4)
        self.elu_2 = nn.ELU()
        self.pool_2 = nn.MaxPool2d(kernel_size=[2, 4])  # stride and padding missing
        self.drop_2 = nn.Dropout(p=0.25)
        self.conv_3 = nn.Conv2d(4, 4, kernel_size=(8, 4), stride=(1, 1), padding=(4, 0))
        self.bnorm_3 = nn.BatchNorm2d(num_features=4)
        self.elu_3 = nn.ELU()
        self.pool_3 = nn.MaxPool2d(kernel_size=(2, 4))
        self.drop_3 = nn.Dropout(p=0.25)
        self.conv_classifier = nn.Conv2d(4, 5, kernel_size=(4, 28))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.ensuredims(x)
        x = self.elu_1(self.bnorm_1(self.conv_1(x)))
        x = self.bnorm_2(self.conv_2(self.drop_1(self.permute_1(x))))
        x = self.elu_3(self.bnorm_3(self.conv_3(self.drop_2(self.pool_2(self.elu_2(x))))))
        #x = self.sigmoid(self.conv_classifier(self.drop_3(self.pool_3(x))))
        x = self.conv_classifier(self.drop_3(self.pool_3(x)))
        return squeeze_final_output(self.sigmoid(x)), torch.flatten(x)


net = Net()