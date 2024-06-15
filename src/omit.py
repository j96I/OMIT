from torchvision.transforms import ToTensor, Lambda
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from torch import nn
import numpy as np
import torch
import os

from utils.pytorch_utils import NeuralNetwork, test_loop, train_loop
from utils.config import *


def useGPU(tensor):
    if torch.cuda.is_available():
        tensor = tensor.to("cuda")


# Downloads and displays mnist dataset
def fashion_mnist_display():
    labels_map = {
        0: "T-Shirt",
        1: "Trouser",
        2: "Pullover",
        3: "Dress",
        4: "Coat",
        5: "Sandal",
        6: "Shirt",
        7: "Sneaker",
        8: "Bag",
        9: "Ankle Boot",
    }

    training_data = datasets.FashionMNIST(
        root="data", train=True, download=True, transform=ToTensor()
    )
    train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)

    test_data = datasets.FashionMNIST(
        root="data", train=False, download=True, transform=ToTensor()
    )
    test_dataloader = DataLoader(test_data, batch_size=64)

    # Display image and label.
    train_features, train_labels = next(iter(train_dataloader))

    print(f"Feature batch shape: {train_features.size()}")
    print(f"Labels batch shape: {train_labels.size()}")

    img = train_features[0].squeeze()
    label = train_labels[0]
    print(f"Label: {label}")

    plt.imshow(img, cmap="gray")
    plt.show()

    # 8x8 image display
    # figure = plt.figure(figsize=(8, 8))
    # cols, rows = 3, 3
    # for i in range(1, cols * rows + 1):
    #     sample_idx = torch.randint(len(training_data), size=(1,)).item()
    #     img, label = training_data[sample_idx]
    #     figure.add_subplot(rows, cols, i)
    #     plt.title(labels_map[label])
    #     plt.axis("off")
    #     plt.imshow(img.squeeze(), cmap="gray")
    # plt.show()


def fashion_mnist_transform():
    ds = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor(),
        target_transform=Lambda(
            lambda y: torch.zeros(10, dtype=torch.float).scatter_(
                0, torch.tensor(y), value=1
            )
        ),
    )

    target_transform = Lambda(
        lambda y: torch.zeros(10, dtype=torch.float).scatter_(
            dim=0, index=torch.tensor(y), value=1
        )
    )


def useNN():
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")

    model = NeuralNetwork().to(device)
    print(model)

    X = torch.rand(1, 28, 28, device=device)
    logits = model(X)
    pred_probab = nn.Softmax(dim=1)(logits)
    y_pred = pred_probab.argmax(1)
    print(f"Predicted class: {y_pred}")


if __name__ == "__main__":
    training_data = datasets.FashionMNIST(
        root="data", train=True, download=True, transform=ToTensor()
    )
    train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)

    test_data = datasets.FashionMNIST(
        root="data", train=False, download=True, transform=ToTensor()
    )
    test_dataloader = DataLoader(test_data, batch_size=64)

    model = NeuralNetwork()

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    epochs = 10
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer)
        test_loop(test_dataloader, model, loss_fn)
    print("Done!")
