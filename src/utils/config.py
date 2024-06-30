from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torchvision import datasets
import torch

model_path = 'omit_model.pth'

learning_rate = 1e-3
batch_size = 64
epochs = 5

device = 'cuda' if torch.cuda.is_available() else 'cpu'

labels_map = {
    0: 'Clean',
    1: 'Dirty',
}
