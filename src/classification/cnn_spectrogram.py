import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import cohen_kappa_score, accuracy_score
from torch import Tensor
from torchvision.models.resnet import conv1x1, conv3x3, Bottleneck, BasicBlock, _log_api_usage_once, resnet18, ResNet18_Weights
from typing import Callable, List, Optional, Type, Union
from tqdm import tqdm


def get_dataloader(idx):
    X = np.load(f"../../data/Engineered_features/eeg_spectrogram_features_data_{idx}.npy")
    global_min = np.min(X[X != -np.inf])
    global_max = np.max(X[X != np.inf])
    X[X < global_min] = global_min
    X[X > global_max] = global_max
    z = X.min((0, 2, 3), keepdims=True)
    y = X.max((0, 2, 3), keepdims=True)
    X = (X - z) / (y - z)  # normalize each channel to have values between 0 and 1
    y = pd.read_csv(f"../../data/Engineered_features/eeg_label_features_{idx}.csv")
    y = np.array(y)
    y = y[::5]  # every fifth row
    batch_size = 32
    dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y).to(torch.float))
    loader = DataLoader(dataset, shuffle=True, batch_size=batch_size)
    return loader, y


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # input_size = 15, 16, 32 (channels, height, width)
        self.conv1 = nn.Conv2d(in_channels=15, out_channels=15, kernel_size=(5, 5), padding="same")
        self.conv2 = nn.Conv2d(in_channels=15, out_channels=15, kernel_size=(5, 5), padding="same")
        self.pool_1 = nn.MaxPool2d(2, 2)  # 8, 16
        self.conv3 = nn.Conv2d(in_channels=15, out_channels=15, kernel_size=(5, 5), padding="same")
        self.conv4 = nn.Conv2d(in_channels=15, out_channels=15, kernel_size=(5, 5), padding="same")
        self.pool_2 = nn.MaxPool2d(2, 2)  # 4, 8
        self.conv5 = nn.Conv2d(in_channels=15, out_channels=15, kernel_size=(3, 3), padding="same")
        self.conv6 = nn.Conv2d(in_channels=15, out_channels=15, kernel_size=(3, 3), padding="same")
        self.fc1 = nn.Linear(15 * 4 * 8, 100)
        self.fc2 = nn.Linear(100, 5)

    def forward(self, x):
        x = self.pool_1(F.relu(self.conv2(F.relu(self.conv1(x)))))
        x = self.pool_2(F.relu(self.conv4(F.relu(self.conv3(x)))))
        x = F.relu(self.conv6(F.relu(self.conv5(x))))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class CustomResNet(nn.Module):
    def __init__(
            self,
            block: Type[Union[BasicBlock, Bottleneck]],
            layers: List[int],
            num_classes: int = 1000,
            zero_init_residual: bool = False,
            groups: int = 1,
            width_per_group: int = 32,
            replace_stride_with_dilation: Optional[List[bool]] = None,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        _log_api_usage_once(self)
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 32
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(15, self.inplanes, kernel_size=(7, 7), padding="same", bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 32, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1])#, stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 32, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 32, layers[3])#, stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(32 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
            self,
            block: Type[Union[BasicBlock, Bottleneck]],
            planes: int,
            blocks: int,
            stride: int = 1,
            dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation,
                norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def run_training(num_iterations):
    #net = Net()
    net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    #net = CustomResNet(block=torchvision.models.resnet.Bottleneck, layers=[3, 3, 3, 3], num_classes=5)
    print(net)
    net.conv1 = nn.Conv2d(in_channels=15, out_channels=64, kernel_size=(7, 7), padding="same")
    net.fc = nn.Linear(in_features=512, out_features=5, bias=True)
    for param in net.parameters():
        param.requires_grad = False
    for param in net.fc.parameters():
        param.requires_grad = True
    for param in net.conv1.parameters():
        param.requires_grad = True

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(net.parameters(), lr=0.001, weight_decay=1e-2)
    for repetition in range(num_iterations):
        for train_idx in range(4):
            loader = get_dataloader(train_idx)
            running_loss = 0.0
            for i, data in enumerate(loader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 20 == 19:    # print every 20 mini-batches
                    print(f'[{repetition} : {i + 1:5d}] loss: {running_loss / 20:.3f}')
                    running_loss = 0.0
                if i == 100:
                    break
        if repetition == 0:
            for param in net.parameters():
                param.requires_grad = True
        torch.save(net.state_dict(), f"resnet_18_{repetition}.pth")

    print('Finished Training')


def test_model(iteration, threshold):
    net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    # net = CustomResNet(block=torchvision.models.resnet.Bottleneck, layers=[3, 3, 3, 3], num_classes=5)
    net.conv1 = nn.Conv2d(in_channels=15, out_channels=64, kernel_size=(7, 7), padding="same")
    net.fc = nn.Linear(in_features=512, out_features=5, bias=True)
    net.load_state_dict(torch.load(f"resnet_18_{iteration}.pth", weights_only=True))
    for idx in range(4):
        loader, labels = get_dataloader(idx)
        # get performance on training data:
        net.eval()
        all_outputs = []
        all_labels = []
        for i, data in tqdm(enumerate(loader, 0)):
            # get the inputs; data is a list of [inputs, labels]
            inputs, temp_labels = data
            temp_labels = np.array(temp_labels)
            all_labels.append(temp_labels.flatten())
            # forward + backward + optimize
            outputs = net(inputs).detach().numpy()
            all_outputs.append(outputs.flatten())

        all_preds = np.hstack(all_outputs)
        labels_to_compare = np.hstack(all_labels)
        y_true_flat = labels_to_compare.flatten()
        y_pred_flat = all_preds.flatten()
        kappa_values = []
        accuracy_values = []
        percentage_points_labeled = []
        for threshold in np.linspace(0.5, 0.9, 100):
            threshold_preds = []
            threshold_labels = []
            abridged_preds_positive = np.ones_like(all_preds)[all_preds > threshold]
            threshold_preds.append(abridged_preds_positive.flatten())
            abridged_labels_positive = labels_to_compare[all_preds > threshold]
            threshold_labels.append(abridged_labels_positive.flatten())
            abridged_preds_negative = np.zeros_like(all_preds)[all_preds < 1 - threshold]
            threshold_preds.append(abridged_preds_negative.flatten())
            abridged_labels_negative = labels_to_compare[all_preds < 1 - threshold]
            threshold_labels.append(abridged_labels_negative.flatten())

            # Step 3: Calculate Cohen's Kappa
            threshold_labels = np.hstack(threshold_labels).flatten()
            threshold_preds = np.hstack(threshold_preds).flatten()
            kappa = cohen_kappa_score(threshold_labels, threshold_preds)
            kappa_values.append(kappa)
            acc = accuracy_score(threshold_labels, threshold_preds)
            accuracy_values.append(acc)
            percentage_labeled = len(threshold_labels) / len(y_true_flat)
            percentage_points_labeled.append(percentage_labeled)
        plt.scatter(kappa_values, percentage_points_labeled)
        plt.show()

if __name__ == "__main__":
    #run_training(80)
    test_model(18, threshold=0.99)

    """483it [00:44, 10.82it/s]
0.0007780830220958102
327it [00:29, 11.01it/s]
0.562176909647746
402it [00:38, 10.56it/s]
0.4627818115555774
426it [00:39, 10.75it/s]
0.4449017457591189"""
