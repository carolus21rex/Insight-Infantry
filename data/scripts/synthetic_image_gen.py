import os
import random
from PIL import Image
import cv2
import numpy as np

# Directory setup
dataset_dir = 'synthetic_human_gun_dataset'
human_images_dir = 'path_to_human_images'
gun_images_dir = 'path_to_gun_images'

os.makedirs(dataset_dir, exist_ok=True)

# Load lists of images
human_images = [os.path.join(human_images_dir, file) for file in os.listdir(human_images_dir) if file.endswith('.png')]
gun_images = [os.path.join(gun_images_dir, file) for file in os.listdir(gun_images_dir) if file.endswith('.png')]

def composite_image(human_img_path, gun_img_path):
    """Composite an image of a human holding a gun."""
    human_img = Image.open(human_img_path)
    gun_img = Image.open(gun_img_path)

    # Assuming gun image needs to be resized and rotated to fit the hand position
    gun_img = gun_img.resize((50, 20))  # Resize gun image
    gun_img = gun_img.rotate(45, expand=1)  # Rotate gun to simulate holding

    # Calculate position to place the gun (this is simplistic and will likely need adjusting)
    gun_position = (100, human_img.height // 2)  # Example position

    human_img.paste(gun_img, gun_position, gun_img)
    
    # Save composite image
    output_path = os.path.join(dataset_dir, f'composite_{os.path.basename(human_img_path)}')
    human_img.save(output_path)

    # Generate YOLO label for the gun
    generate_yolo_label(output_path, gun_img, gun_position, human_img.size)

def generate_yolo_label(img_path, gun_img, gun_position, img_size):
    """ Generate a YOLO label file for the gun in the image. """
    x_min, y_min = gun_position
    x_max, y_max = x_min + gun_img.width, y_min + gun_img.height

    yolo_format = convert_to_yolo_format((x_min, y_min, x_max, y_max), img_size)
    label_path = img_path.replace('.png', '.txt')
    with open(label_path, 'w') as file:
        file.write(f"0 {yolo_format}\n")  # Assuming '0' is the class ID for 'gun'

def convert_to_yolo_format(box, img_dim):
    """ Convert bounding box coordinates to YOLO format. """
    dw = 1. / img_dim[0]
    dh = 1. / img_dim[1]
    x = (box[0] + box[2]) / 2.0
    y = (box[1] + box[3]) / 2.0
    w = box[2] - box[0]
    h = box[3] - box[1]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return f"{x:.6f} {y:.6f} {w:.6f} {h:.6f}"

# Generate composite images
for human_img_path in human_images:
    gun_img_path = random.choice(gun_images)
    composite_image(human_img_path, gun_img_path)
