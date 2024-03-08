from PIL import Image
import os

# Replace 'your/actual/folder/path' with the actual path to your folder containing JPG images
folder_path = os.getcwd()

# Dictionary to hold the filename and its dimensions
image_dimensions = {}

# Iterate through each file in the folder
for filename in os.listdir(folder_path):
    # Check if the file is a jpg file
    if filename.lower().endswith('.jpg'):
        # Open the image file
        with Image.open(os.path.join(folder_path, filename)) as img:
            # Get the dimensions of the image
            width, height = img.size
            # Store the dimensions in the dictionary
            image_dimensions[filename] = (width, height)

expected_x = 1280
expected_y = 720

# Print the dimensions for each image
for filename, dimensions in image_dimensions.items():
    if dimensions[0] != expected_x or dimensions[1] != expected_y:
        print(f'{filename}: {dimensions[0]}x{dimensions[1]}')
