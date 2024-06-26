import matplotlib.pyplot as plt
import torch

from utils.pytorch_dataset_utils import CustomImageDataset
from utils.pytorch_training_utils import NeuralNetwork, test_loop, train_loop
from utils.config import *


def data_init():
    dataset = CustomImageDataset(
        annotations_file='src\data\custom\data.csv',
        img_dir='src\data\custom',
        transform=ToTensor(),
    )

    train_data, test_data = torch.utils.data.random_split(dataset, [2000, 5000])

    train_dataloader = DataLoader(
        dataset=train_data, batch_size=batch_size, shuffle=True
    )
    test_dataloader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)

    return train_dataloader, test_dataloader


def train_model():
    # Load in premade model
    # model = torch.load(model_path)

    train_dataloader, test_dataloader = data_init()

    # Create model, set to GPU processing
    model = NeuralNetwork().to(device)

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

    print(model)
    print('Done!')


def use_model(img_index):
    train_dataloader, test_dataloader = data_init()

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
        print(f'Predicted: "{predicted}", Actual: "{actual}"')

    plt.imshow(img.squeeze(), cmap='gray')
    plt.title(actual)
    plt.axis('off')
    plt.show()


train_model()
use_model(0)
