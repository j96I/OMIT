import torch

model_path = 'data/omit_model.pth'
training_data_path = 'data/custom_dataset'

learning_rate = 1e-3

epochs = 150
image_sample_percentage = 2

device = 'cuda' if torch.cuda.is_available() else 'cpu'
