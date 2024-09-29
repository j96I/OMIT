from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os

class CustomImageDataset(Dataset):
    def __init__(self, img_dir):
        self.img_dir = img_dir
        self.transform = transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize((100, 100)),
                transforms.ToTensor(),
            ]
        )
        self.img_labels = []
        self.class_to_idx = {}

        self._prepare_dataset()

    def _prepare_dataset(self):

        # if the images are not in the expected directory, they might be one level up.
        if (os.path.isdir(self.img_dir) is False):
            self.img_dir = os.path.join('..', self.img_dir)
        
        # use the image folder names as labels, but ignore the directory called .git!
        if (os.path.isdir(self.img_dir)):
          # first look for the training data path as it was passed.
          image_folders_list = sorted(os.listdir(self.img_dir))
    
          if '.git' in image_folders_list:
              image_folders_list.remove('.git')  
              
        else:
          print(f"ERROR - couldnt find the input data on path: {self.img_dir}")
          exit(1)

        # Create a class to index mapping
        for idx, class_name in enumerate(image_folders_list):
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