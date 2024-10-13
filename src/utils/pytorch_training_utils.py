from datetime import datetime
from torch import nn
import torch

from utils.config import *


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        # Convolutional layer 1
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1
        )

        # Max pooling layer to reduce the size after convolutions
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Convolutional layer 2
        self.conv2 = nn.Conv2d(
            in_channels=16, out_channels=64, kernel_size=3, stride=1, padding=1
        )

        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(64 * 25 * 25, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 2),
        )

    def forward(self, x):
        # Pass through convolutional layers with ReLU activation and pooling
        x = self.pool(nn.LeakyReLU()(self.conv1(x)))
        x = self.pool(nn.LeakyReLU()(self.conv2(x)))

        # Flatten the output from conv layers before feeding into fully connected layers
        x = self.flatten(x)

        # Pass through the fully connected layers
        logits = self.linear_relu_stack(x)
        return logits


def train_loop(dataloader, model, loss_fn, optimizer, device, batch_size, start_time):
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    size = len(dataloader.dataset)

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f'loss: {loss:>7f}  [{current:>5d}/{size:>5d}]')
    
    time_components = []
    elapsed_time = datetime.now() - start_time
    numeric_elapsed_time = elapsed_time.total_seconds()

    days = elapsed_time.days
    hours, remainder = divmod(elapsed_time.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    if days > 0:
        time_components.append(f"{days} days")
    if hours > 0:
        time_components.append(f"{hours} hrs")
    if minutes > 0:
        time_components.append(f"{minutes} mins")
    if seconds > 0:
        time_components.append(f"{seconds} secs")

    formatted_difference = ', '.join(time_components)
    return formatted_difference, numeric_elapsed_time


def test_loop(dataloader, model, loss_fn, device):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(
        f'Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n'
    )
    return correct, test_loss
