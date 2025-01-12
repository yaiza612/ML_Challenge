import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from src.data_loading.loader import load_train_data_as_windows, load_test_data_as_windows
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from torchinfo import summary
import matplotlib.pyplot as plt
from tqdm import tqdm


# Define the CNN Autoencoder architecture
class CNNAutoencoder(nn.Module):
    def __init__(self):
        super(CNNAutoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            #nn.Conv1d(5, 16, kernel_size=3, stride=1, padding=1),
            #nn.ReLU(),
            nn.Conv1d(5, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            #nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1),
            #nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=32, out_channels=1, kernel_size=1)
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(1, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(negative_slope=0.01),
            nn.ConvTranspose1d(16, 5, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def get_data(index):
    if index < 4:
        data, _ = load_train_data_as_windows(index)
    else:
        data = load_test_data_as_windows(index)
    data_reshaped = data.reshape(-1, 1)
    scaler = StandardScaler()
    data_normalized = scaler.fit_transform(data_reshaped)
    reshaped_data = data_normalized.reshape(data.shape)
    return reshaped_data



if __name__ == "__main__":
    path_for_features = ("../../data/Engineered_features/")
    # load data
    all_data = []
    z = np.load(f"{path_for_features}eeg_autoencoder_features_data_{2}.npy")
    for i in range(6):
        all_data.append(get_data(i))


    all_data = np.concatenate(all_data, axis=1)
    print(all_data.shape)
    all_data = np.swapaxes(all_data, 0, 1)

    data_tensor = torch.tensor(all_data, dtype=torch.float32)

    # Split data into train/val
    train_size = int(0.8 * len(data_tensor))
    val_size = len(data_tensor) - train_size
    train_data, val_data = torch.utils.data.random_split(data_tensor, [train_size, val_size])

    # Initialize the autoencoder
    autoencoder = CNNAutoencoder()
    summary(autoencoder, input_size=(1, 5, 500))

    # Set up optimizer and loss function
    optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    # Prepare DataLoader
    batch_size = 64
    train_dataset = TensorDataset(train_data.dataset[train_data.indices], train_data.dataset[train_data.indices])
    val_dataset = TensorDataset(val_data.dataset[val_data.indices], val_data.dataset[val_data.indices])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    best_val_loss = float('inf')
    best_model_state = None
    # Training the autoencoder
    epochs = 50
    for epoch in range(epochs):
        autoencoder.train()
        for data in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False):
            inputs, _ = data
            optimizer.zero_grad()
            outputs = autoencoder(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()

        # print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

        # Validation loss
        autoencoder.eval()
        val_loss = 0
        with torch.no_grad():
            for val_data in val_loader:
                val_inputs, _ = val_data
                val_outputs = autoencoder(val_inputs)
                val_loss += criterion(val_outputs, val_inputs).item()
        val_loss /= len(val_loader)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = autoencoder.state_dict()

        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}')

    if best_model_state is not None:
        autoencoder.load_state_dict(best_model_state)
        print("Best model loaded")

    # Extract features using the encoder
    with torch.no_grad():
        for i in range(6):
            data = get_data(i)
            data = np.swapaxes(data, 0, 1)

            data_tensor = torch.tensor(data, dtype=torch.float32)
            csv_dataset = TensorDataset(data_tensor, data_tensor)
            csv_loader = DataLoader(csv_dataset, shuffle=False, batch_size=64)
            features = []
            for batch in tqdm(csv_loader, desc=f"Creating csv for index {i+1}/6", leave=False):
                batch_in, _ = batch
                batch_out = autoencoder.encoder(batch_in).numpy()
                features.append(batch_out)
            features = np.concatenate(features, axis=0)
            #features = autoencoder.encoder(data_tensor).numpy()  # shape: num_samples, 1, autoenc_dim
            features = np.repeat(features, repeats=5, axis=0)
            features = features.reshape(features.shape[0], features.shape[2])
            print(features.shape)
            column_names = [f"autoenc_{i+1}" for i in range(features.shape[1])]
            df = pd.DataFrame(features, columns=column_names)
            df.to_csv(f"{path_for_features}eeg_autoencoder_features_data_{i}.csv", index=False)
            # reshape it to num_samples * 5, autoenc_dim
            #np.save(f"{path_for_features}eeg_autoencoder_features_data_{i}.npy", features)
