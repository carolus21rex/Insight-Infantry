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
    # initialize dataset with data_root and transforms defined below, calls image_paths
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = self._get_image_paths()

    # traversing through root_dir & retrieves image files with jpg extension
    def _get_image_paths(self):
        image_paths = []
        for root, dirs, files in os.walk(self.root_dir):
            for file in files:
                if file.endswith(".jpg"):  # Adjust file extension as needed
                    image_paths.append(os.path.join(root, file))
        return image_paths

    # pads the bounding box tensors to zero
    def _pad_boxes(self, boxes, max_num_boxes):
        padded_boxes = torch.zeros((max_num_boxes, boxes.shape[1]))
        padded_boxes[:boxes.shape[0], :] = boxes
        return padded_boxes

    # returns the total number of images in the dataset
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # loads an image with proper format
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("RGB")

        # Extract bounding box information from the corresponding text file
        txt_path = img_path.replace(".jpg", ".txt")
        with open(txt_path, "r") as file:
            bbox_info = [line.split() for line in file]

        # Convert bounding box coordinates to tensor
        bbox_tensor = torch.tensor([[float(info) for info in bbox] for bbox in bbox_info])

        # sets max tensors boxes and calls pad_boxes to pad with 0
        max_num_boxes = 10  # Max number of coordinate groups found in txt file
        padded_bbox_tensor = self._pad_boxes(bbox_tensor, max_num_boxes)

        # applies transform to image if possible
        if self.transform:
            img = self.transform(img)

        # returns image and padded bounding box tensor
        return img, padded_bbox_tensor


# defines the root for finding the images
data_root = r"C:\Users\Jonah Dalton\PycharmProjects\DataForProjects\train_images"
# resizes image for input & converts image to Pytorch tensors
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# instance of CustomDataset created passing in root and transform
dataset = CustomDataset(root_dir=data_root, transform=transform)
# Use DataLoader to load data in batches using dataset as input
data_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
