import cv2
import os
import numpy as np
import torch
import torchvision
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from PIL import Image


def prepare_image(image_path):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = transform(image).unsqueeze(0)  # the model requires a batch dimension
    return image


def predict_opencv(model, device, image_path, threshold=0.1):
    image = prepare_image(image_path)
    image = image.to(device)

    model.eval()

    with torch.no_grad():
        prediction = model(image)

    above_threshold_indices = (prediction[0]['scores'] > threshold)
    prediction[0]['boxes'] = prediction[0]['boxes'][above_threshold_indices]
    prediction[0]['labels'] = prediction[0]['labels'][above_threshold_indices]
    prediction[0]['scores'] = prediction[0]['scores'][above_threshold_indices]
    for index, score in enumerate(prediction[0]['scores']):
        if score > 0.2:
            print(prediction[0]['boxes'][index], prediction[0]['labels'][index], score)

    return prediction


def draw_boxes_on_image(image_path, prediction):
    image = cv2.imread(image_path)
    boxes = prediction[0]['boxes'].data.cpu().numpy().astype(np.int32)

    for box in boxes:
        # print(f"Using box with coordinates: {box}")
        cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), color=(0, 255, 0), thickness=2)

    cv2.imshow('Image with Boxes', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def create_model(num_classes):
    model = fasterrcnn_resnet50_fpn()
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features,
                                                                                                   num_classes)
    return model


if __name__ == "__main__":
    # Set the device
    device = get_device()

    # Initialize the model
    model = create_model(num_classes=3)

    # Load the trained model
    model.load_state_dict(torch.load("detection_model.pth"))
    model.to(device)

    # Path of the image you want to use
    image_path = os.path.abspath(os.path.join(os.getcwd(), '..', '..', '..', 'data', 'train_images', 'image_0001.jpg'))

    # Get model prediction
    pred = predict_opencv(model, device, image_path)

    # Draw bounding boxes on the image
    draw_boxes_on_image(image_path, pred)
