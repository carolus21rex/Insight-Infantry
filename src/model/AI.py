import torch
from torchvision.models import detection
import cv2
import torchvision.transforms as transforms
import numpy as np


# Preprocess the image
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img = transform(img)
    return img, img.unsqueeze(0)  # Add batch dimension


# Load your object detection model
def load_detection_model():
    model = detection.fasterrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    return model


# Function to assign threat scores
def assign_threat_score(detections):
    # Example scoring logic
    score = 0
    for detection in detections:
        label = detection['label']
        if label == 'Adult' and 'AK-47' in detection['objects']:
            score += 10
        elif label == 'Child' and 'Hand Pistol' in detection['objects']:
            score += 5
        # Add more rules as needed
    return score


# Mock function for detection - replace with your actual detection logic
def detect_objects(image, model):
    # This function should return a list of detections with their labels and confidence
    # For example: [{'label': 'Adult', 'objects': ['AK-47'], 'confidence': 0.9}]
    # You'll need to implement actual detection logic based on your models
    model.eval()
    with torch.no_grad():
        output = model(image)
    # return [{'label': 'Adult', 'objects': ['AK-47'], 'confidence': 0.9}]
    print(output[0])
    # probabilities = torch.nn.functional.softmax(output[0], dim=0)
    return output[0]


# Main processing function
def process_image(image_path):
    out, img = preprocess_image(image_path)
    model = load_detection_model()
    output = detect_objects(img, model)  # You need to implement this based on your model
    # score = assign_threat_score(detections)
    # return score, detections, confidence
    display_image(out, output['boxes'])
    return output


def display_image(image, boxes):
    # Convert the image to a NumPy array
    image_np = image.permute(1, 2, 0).numpy()

    # Convert image to uint8 and scale to 0-255
    image_np = (image_np * 255).astype(np.uint8)

    # Ensure image is in BGR format (OpenCV expects BGR)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    # Draw a box for each tensor
    for box in boxes:
        # Extract coordinates from tensor
        x_min, y_min, x_max, y_max = box.tolist()

        # Draw rectangle on image
        cv2.rectangle(image_np, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)

    # Display the image with boxes
    cv2.imshow('Image with Boxes', image_np)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # Display the image with boxes
    cv2.imshow('Image with Boxes', image_np)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Example usage
image_path = 'D:\\insightInfantry\\data\\test2.jpg'
boxes = process_image(image_path)
for ele in boxes['boxes']:
    print(ele)
