from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from random import randrange
import torch

from utils.pytorch_dataset_utils import CustomImageDataset, revert_grayscale_to_rgb
from utils.pytorch_training_utils import NeuralNetwork, test_loop, train_loop
from utils.config import *


def data_init():
    dataset = CustomImageDataset(img_dir='data/custom_dataset')

    # Determine the sizes for train and test splits
    dataset_size = len(dataset)
    train_size = int(0.7 * dataset_size)  # 70% for training
    test_size = dataset_size - train_size  # Remaining 30% for testing

    train_data, test_data = torch.utils.data.random_split(
        dataset, [train_size, test_size]
    )

    train_dataloader = DataLoader(
        dataset=train_data, batch_size=batch_size, shuffle=True
    )
    test_dataloader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)

    return train_dataloader, test_dataloader


def train_model(retrain=False):
    # Get testing and training data
    train_dataloader, test_dataloader = data_init()

    # Load in premade or Create new model
    model = torch.load(model_path) if retrain else NeuralNetwork()

    # Set to GPU processing
    model.to(device)

    # Creates a criterion that measures the mean absolute error (MAE)
    loss_fn = torch.nn.CrossEntropyLoss()

    # Optimizer object which holds the current state
    # & will update the parameters based on the computed gradients
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # Per epoch, refine model
    for t in range(epochs):
        print(f'Epoch {t+1}\n-------------------------------')
        train_loop(train_dataloader, model, loss_fn, optimizer, device)
        test_loop(test_dataloader, model, loss_fn, device)

    torch.save(model, model_path)

    print('Done!')


def use_model():
    img_index = randrange(10)
    _, test_dataloader = data_init()

    first_batch = next(iter(test_dataloader))
    image_tensor, label_index = first_batch[0][img_index], first_batch[1][img_index]
    img = image_tensor.squeeze()

    model = torch.load(model_path)
    model.eval()

    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        pred = model(image_tensor)

        predicted, actual = (
            labels_map[pred[0].argmax(0).item()],
            labels_map[label_index.item()],
        )
        prediction = f'Predicted: "{predicted}", Actual: "{actual}"'
        print(prediction)

    img_rgb = revert_grayscale_to_rgb(img)

    plt.imshow(img_rgb)
    plt.title(prediction)
    plt.axis('off')
    plt.show()


# train_model()
use_model()
