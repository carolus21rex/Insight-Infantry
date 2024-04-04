from torchvision.transforms import transforms
from torch.utils.data import Dataset
from PIL import Image


class MyDataset(Dataset):
    def __init__(self, image_paths, target_paths, transform=None):
        self.image_paths = image_paths
        self.target_paths = target_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx])
        target = load_yolo_label(self.target_paths[idx])

        if self.transform:
            image = self.transform(image)

        return image, target


def load_yolo_label(label_path):
    labels = []
    with open(label_path, 'r') as f:
        for line in f.readlines():
            class_id, x, y, width, height = map(float, line.split())
            labels.append({'class_id': int(class_id), 'bbox': [x, y, width, height]})
    return labels
