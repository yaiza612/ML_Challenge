import torch
import torch.nn as nn
from torch.nn import Module
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import LRScheduler
from sklearn.metrics import cohen_kappa_score
import numpy as np



def create_dataloader(features, labels, batch_size):
    tensor_x = torch.tensor(features)
    tensor_y = torch.tensor(labels.T)
    tensor_y = tensor_y.to(torch.float32)
    dataset = TensorDataset(tensor_x, tensor_y)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return data_loader


class SigmoidBinaryKappaLoss(nn.Module):
    def __init__(self):
        super(SigmoidBinaryKappaLoss, self).__init__()
        self.weights = torch.tensor([[0, 1],
                                     [1, 0]], dtype=torch.float32)  # Quadratic weights for binary case

    def forward(self, logits, labels):
        batch_size = logits.size(0)

        # Convert logits to probabilities using sigmoid
        prob = torch.sigmoid(logits)  # Shape: (batch_size,)

        # Construct the predicted probabilities for both classes (0 and 1)
        prob = torch.stack([1 - prob, prob], dim=1)  # Shape: (batch_size, 2)

        # One-hot encode the true labels
        true_dist = torch.zeros(batch_size, 2, device=logits.device)
        true_dist.scatter_(1, labels.unsqueeze(1).type(torch.int64), 1)
        # Shape of true_dist: (batch_size, 2)

        # Compute the confusion matrix (soft version)
        conf_matrix = torch.matmul(true_dist.T, prob)  # Shape: (2, 2)

        # Compute row sums and column sums for the expected agreement matrix
        row_sums = conf_matrix.sum(1).unsqueeze(1)  # Shape: (2, 1)
        col_sums = conf_matrix.sum(0).unsqueeze(0)  # Shape: (1, 2)

        # Expected matrix is computed from the row and column sums
        expected_matrix = torch.matmul(row_sums, col_sums) / batch_size
        # Shape of expected_matrix: (2, 2)

        # Apply quadratic weights to the confusion matrix and expected matrix
        weighted_conf_matrix = self.weights.to(logits.device) * conf_matrix
        weighted_expected_matrix = self.weights.to(logits.device) * expected_matrix

        # Compute the final Kappa loss: 1 - (observed agreement / expected agreement)
        kappa_loss = (weighted_conf_matrix.sum() / weighted_expected_matrix.sum())

        return kappa_loss


def train_one_epoch(
        dataloader: DataLoader, model: Module, loss_fn, optimizer,
        scheduler: LRScheduler, epoch: int, device, prob=True
):
    model.train()  # Set the model to training mode
    train_loss, correct = 0, 0


    for batch_idx, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        pred_prob, logits = model(X)
        # pred_prob.shape, logits.shape and y.shape is (8, 5)
        loss = loss_fn(pred_prob, y)
        loss.backward()
        optimizer.step()  # update the model weights
        optimizer.zero_grad()
        train_loss += np.abs(loss.item())
        correct += ((pred_prob > 0.5) == y).sum().item()

    # Update the learning rate
    scheduler.step()

    correct /= (len(dataloader.dataset) * 5)
    return train_loss / len(dataloader), correct


def validate_model(dataloader: DataLoader, model: Module, device="cpu"):
    model.eval()
    all_preds = []
    all_labels = []
    for batch_idx, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        pred_prob, logits = model(X)
        all_preds.append(pred_prob.detach())
        all_labels.append(y.detach())
        # pred_prob.shape, logits.shape and y.shape is (8, 5)

    return np.array(all_preds), np.array(all_labels)

def train_for_epochs(n_epochs, train_loader, test_loader, model, optimizer, loss_fn, scheduler):
    for epoch in range(1, n_epochs + 1):
        train_loss, train_accuracy = train_one_epoch(dataloader=train_loader, model=model, loss_fn=loss_fn,
                                                     optimizer=optimizer, scheduler=scheduler,
                                                     epoch=epoch, device="cpu", prob=False)

        kappas = []
        for loader in [train_loader, test_loader]:
            predictions, logits = validate_model(dataloader=loader, model=model, device="cpu")

            predictions = predictions.flatten()
            logits = logits.flatten()
            pred_binary = np.where(predictions > 0.5, 1, 0)
            kappa = cohen_kappa_score(pred_binary, logits)
            kappas.append(kappa)
        print(
            f"Train Accuracy: {100 * train_accuracy:.2f}%, "
            f"Average Train Loss: {train_loss:.6f}",
            f"Kappa on train dataset: {kappas[0]:.3f}",
            f"Kappa on validation dataset: {kappas[1]:.3f}"
        )