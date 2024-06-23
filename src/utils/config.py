from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torchvision import datasets
import torch

model_path = 'omit_model.pth'

learning_rate = 1e-3
batch_size = 64
epochs = 5

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ---------------------------------------------------------------------

# Import training data
training_data = datasets.FashionMNIST(
    root='data', train=True, download=True, transform=ToTensor()
)
train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)

# Import testing data
test_data = datasets.FashionMNIST(
    root='data', train=False, download=True, transform=ToTensor()
)
test_dataloader = DataLoader(test_data, batch_size=64)

labels_map = {
    0: 'T-Shirt',
    1: 'Trouser',
    2: 'Pullover',
    3: 'Dress',
    4: 'Coat',
    5: 'Sandal',
    6: 'Shirt',
    7: 'Sneaker',
    8: 'Bag',
    9: 'Ankle Boot',
}

# ---------------------------------------------------------------------

omit_labels_map = {
    0: 'Clean',
    1: 'Dirty',
}
