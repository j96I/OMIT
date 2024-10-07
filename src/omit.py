import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from datetime import datetime
from PIL import Image
import pandas as pd
import torch
import os

from utils.pytorch_training_utils import NeuralNetwork, test_loop, train_loop
from utils.pytorch_dataset_utils import CustomImageDataset
from utils.config import *


def data_init(img_dir):
    dataset = CustomImageDataset(img_dir)

    # Determine the sizes for train and test splits
    dataset_size = len(dataset)

    # Batch size relative to dataset
    batch_size = round(dataset_size * (image_sample_percentage/100))

    train_size = int(0.8 * dataset_size)  # 80% for training
    test_size = dataset_size - train_size  # Remaining 20% for testing

    train_data, test_data = torch.utils.data.random_split(
        dataset, [train_size, test_size]
    )

    train_dataloader = DataLoader(
        dataset=train_data, batch_size=batch_size, shuffle=True
    )
    test_dataloader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)

    return train_dataloader, test_dataloader, batch_size


def train_model(retrain=False, img_dir=training_data_path):
    # Get testing and training data
    train_dataloader, test_dataloader, batch_size = data_init(img_dir)

    # Load in premade or Create new model
    model = torch.load(model_path) if retrain else NeuralNetwork()

    # Set to GPU processing
    model.to(device)

    # Creates a criterion that measures the mean absolute error (MAE)
    loss_fn = torch.nn.CrossEntropyLoss()

    # Optimizer object which holds the current state
    # & will update the parameters based on the computed gradients
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    start_time = datetime.now()
    results = []

    # Per epoch, refine model
    for t in range(epochs):
        print(f'Epoch {t+1}\n-------------------------------')
        elapsed_timestamp = train_loop(train_dataloader, model, loss_fn, optimizer, device, batch_size, start_time)
        accuracy, avg_loss = test_loop(test_dataloader, model, loss_fn, device)

        result = {'Accuracy %': accuracy*100, 'Avg Loss': avg_loss, 'Elapsed timestamp': elapsed_timestamp}
        results.append(result)

    training_stats = pd.DataFrame(results)

    # Plot the Accuracy % over time
    plt.figure(figsize=(10, 6))
    plt.xticks(rotation=45)
    plt.plot(
        training_stats['Elapsed timestamp'],
        training_stats['Accuracy %'],
        marker='o', linestyle='-', color='b', label='Accuracy')

    # Title & labels
    plt.title('Model Accuracy over Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Accuracy (%)')

    # Configuration label
    config_text = (
        f"Accuracy: {training_stats['Accuracy %'].iloc[-1]:>0.1f}%\n\n"
        f"Epochs: {epochs}\n"
        f"Sample size: {image_sample_percentage}%\n"
        f"Learning rate: {learning_rate:.1e}")
    plt.text(1.05, 0.5, config_text, transform=plt.gca().transAxes, fontsize=12, verticalalignment='center', bbox=dict(facecolor='white', alpha=0.5))

    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    end_time = datetime.now().strftime('%d-%m-%Y %H-%M')
    plt.savefig(f'data/Model_AoT {end_time}.jpg')


    torch.save(model, model_path)
    print('Training Complete!')


def predict_image(jpeg_image, img_dir=training_data_path):

    if (os.path.isdir(training_data_path)):
        # first look for the training data path as it was passed.
        image_folders_list = sorted(os.listdir(img_dir))
    
    # Load the model and move it to the appropriate device
    model = torch.load(model_path, map_location=device, weights_only=False)
    model.to(device)
    model.eval()
    
    # Define the transformation pipeline
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((100, 100)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # Open the image
    image = Image.open(jpeg_image).convert('L')
    image_tensor = transform(image).unsqueeze(0)
    
    # Move the tensor to the same device as the model
    image_tensor = image_tensor.to(device)
    
    # Make the prediction
    with torch.no_grad():
        output = model(image_tensor)
        return image_folders_list[output[0].argmax(0).item()]
    