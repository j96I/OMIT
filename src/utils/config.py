import torch

model_path = 'data/omit_model.pth'
training_data_path = 'data/custom_dataset'

learning_rate = 1e-3
batch_size = 5
epochs = 100

device = 'cuda' if torch.cuda.is_available() else 'cpu'

labels_map = {
    0: 'Clean',
    1: 'Messy',
}
