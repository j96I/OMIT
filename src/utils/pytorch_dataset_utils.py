from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os


class CustomImageDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform or transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize((28, 28)),  # Resize images to 28x28
                transforms.ToTensor(),
            ]
        )
        self.img_labels = []
        self.class_to_idx = {}

        self._prepare_dataset()

    def _prepare_dataset(self):
        # Create a class to index mapping
        for idx, class_name in enumerate(sorted(os.listdir(self.img_dir))):
            class_path = os.path.join(self.img_dir, class_name)
            if os.path.isdir(class_path):
                self.class_to_idx[class_name] = idx
                # Collect image paths and labels
                for img_name in os.listdir(class_path):
                    img_path = os.path.join(class_path, img_name)
                    self.img_labels.append((img_path, idx))

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path, label = self.img_labels[idx]
        # Use PIL to open the image
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label
