import torch
import torchvision
import os
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import transforms
from src.app.train.data_loader import get_dataloader
from src.app.train.dataset import MyDataset
from src.app.train.image_helper import get_image_paths
from src.app.train.target_helper import get_target_paths


def main(num_samples):
    device = get_device()
    model = create_model(num_classes=3)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize all images to have size 224x224
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image_dir = os.path.abspath(os.path.join(os.getcwd(), '..', '..', '..', 'data', 'train_images'))
    all_image_paths = get_image_paths(image_dir)
    all_target_paths = get_target_paths(image_dir)
    image_paths = all_image_paths[:num_samples]
    target_paths = all_target_paths[:num_samples]
    dataset = MyDataset(image_paths, target_paths, transform)
    data_loader = get_dataloader(dataset, batch_size=4, num_workers=0, shuffle=True)
    train_model(model, data_loader, device, num_epochs=10)
    torch.save(model.state_dict(), 'detection_model.pth')


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def create_model(num_classes):
    model = fasterrcnn_resnet50_fpn()
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    return model


def convert_yolo_format_to_corners(bbox):
    center_x, center_y, width, height = bbox.unbind(dim=-1)
    xmin = center_x - width / 2
    ymin = center_y - height / 2
    xmax = center_x + width / 2
    ymax = center_y + height / 2
    return torch.stack([xmin, ymin, xmax, ymax], dim=-1)


def train_model(model, data_loader, device, num_epochs):
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    for epoch in range(num_epochs):
        model.train()
        i = 0
        for images, targets in data_loader:
            images = list(image.to(device) for image in images)

            new_targets = []
            for t in targets:
                boxes = []
                labels = []
                for box in t:
                    boxes.append(convert_yolo_format_to_corners(torch.tensor(box['bbox']).to(device)))
                    label = box['class_id'] - 14
                    labels.append(torch.tensor(label).to(device))
                new_targets.append({"boxes": torch.stack(boxes), "labels": torch.tensor(labels)})
            targets = new_targets
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            i += 1
            if i % 10 == 0:
                print(f"Iteration: {i}, Loss: {losses.item()}")


if __name__ == '__main__':
    main(1310)
