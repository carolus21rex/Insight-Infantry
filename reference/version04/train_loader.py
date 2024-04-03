# Importing libraries
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

"""
Imported Required Packages:
    os - Functions for interacting with the OS | Used for traversing the dataset
    Image - Provides image processing capabilities | Used for loading and manipulating images
    torch - Main library for PyTorch | Used to access torch dataloader/datasets to feed Pytorch model
    Dataset - Abstract class for a dataset | Used as a base class for the custom dataset classes
    transforms - provides common image transformations | Used for preprocessing img before feeding to model
"""


# finds img + annotations pairs in dataset and applies proper
# transformations to each img to load into Pytorch DataLoader
class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = self._get_image_paths()

    def _get_image_paths(self):
        image_paths = []
        for root, _, files in os.walk(self.root_dir):
            for file in files:
                if file.endswith(".jpg"):
                    image_paths.append(os.path.join(root, file))
        return image_paths

    def _load_bboxes(self, txt_path):
        with open(txt_path, "r") as file:
            bbox_info = [line.split() for line in file]
        bbox_tensor = torch.tensor([[float(info) for info in bbox] for bbox in bbox_info])
        return bbox_tensor

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            return None

        txt_path = img_path.replace(".jpg", ".txt")
        try:
            bbox_tensor = self._load_bboxes(txt_path)
        except Exception as e:
            print(f"Error loading bounding boxes for {img_path}: {e}")
            return None

        # Apply transformations
        if self.transform:
            img = self.transform(img)

        return img, bbox_tensor


# defines the root for finding the images
data_root = os.path.join(os.getcwd(), '../../src/app', '..', '..', 'data', 'train_images')
print(data_root)
# resizes image for input & converts image to Pytorch tensors
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# instance of CustomDataset created passing in root and transform
dataset = CustomDataset(root_dir=data_root, transform=transform)
# Use DataLoader to load data in batches using dataset as input
data_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)