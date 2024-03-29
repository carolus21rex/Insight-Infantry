import os
import glob
import pandas as pd
import numpy as np
import cv2
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torch.utils.data import DataLoader, Dataset
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2


# Define the dataset class
class GunDataset(Dataset):
    def __init__(self, dataframe, image_dir, transforms=None):
        super().__init__()

        self.image_ids = dataframe['image_id'].unique()
        self.df = dataframe
        self.image_dir = image_dir
        self.transforms = transforms

    def __getitem__(self, index: int):
        image_id = self.image_ids[index]
        records = self.df[self.df['image_id'] == image_id]

        image_path = os.path.join(self.image_dir, f'{image_id}.jpg')
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0

        boxes = records[['minx', 'miny', 'width', 'height']].values
        boxes[:, 2] += boxes[:, 0]  # Convert width to xmax
        boxes[:, 3] += boxes[:, 1]  # Convert height to ymax
        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        area = torch.as_tensor(area, dtype=torch.float32)

        labels = torch.tensor(records['label'].values, dtype=torch.int64)

        target = {}
        target['boxes'] = torch.as_tensor(boxes, dtype=torch.float32)
        target['labels'] = labels
        target['image_id'] = torch.tensor([index])
        target['area'] = area
        target['iscrowd'] = torch.zeros((records.shape[0],), dtype=torch.int64)

        if self.transforms:
            sample = {
                'image': image,
                'bboxes': target['boxes'],
                'labels': labels
            }
            sample = self.transforms(**sample)
            image = sample['image']

            target['boxes'] = torch.stack(tuple(map(torch.tensor, zip(*sample['bboxes'])))).permute(1, 0)

        return image, target

    def __len__(self) -> int:
        return self.image_ids.shape[0]


# Define transformations
def get_transform(train):
    if train:
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            ToTensorV2(p=1.0)
        ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})
    else:
        return A.Compose([
            ToTensorV2(p=1.0)
        ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})


# Load the dataset
def load_dataset(csv_file, img_dir, train=True):
    df = pd.read_csv(csv_file)
    dataset = GunDataset(dataframe=df, image_dir=img_dir, transforms=get_transform(train))
    return dataset


# Define the model
def create_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


# Training function
def train_model(model, data_loader, optimizer, device, epochs=10):
    model.to(device)
    model.train()

    for epoch in range(epochs):
        for images, targets in data_loader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

        print(f'Epoch {epoch + 1}/{epochs}, Loss: {losses.item()}')


# Main training script
def main():
    # Set device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Load train dataset
    train_dataset = load_dataset(csv_file='train_annotations.csv', img_dir='train_images', train=True)
    train_data_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

    # Create model
    num_classes = 3  # 2 classes (Human, Gun) + background
    model = create_model(num_classes)

    # Define optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)

    # Train the model
    train_model(model, train_data_loader, optimizer, device, epochs=10)

    # Save the trained model
    torch.save(model.state_dict(), 'gun_detection_model.pth')


if __name__ == '__main__':
    main()
