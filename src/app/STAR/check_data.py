import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from load_data import CustomDataset

"""
Imported Required Packages:
    plt - Provides plotting capabilities | Used to visualize images with bounding boxes
    torch - Main library for PyTorch | Used to create DataLoader and perform tensor operations
    transforms - Provides common image transformations | Used for preprocessing img before feeding to model
    CustomDataset - Custom class for dataset handling | Accesses the created CustomDataset from the load_data file
"""

# Create a dataset instance by calling CustomDataset & defining transform and data_root
data_root = r"C:\Users\Jonah Dalton\PycharmProjects\DataForProjects\train_images"
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])
dataset = CustomDataset(root_dir=data_root, transform=transform)

# Create a DataLoader to iterate over the dataset
data_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)


# Plot images with the bounding boxes obtained from the CustomDataset
def plot_images_with_boxes(images, boxes):
    # get num of images in the batch
    num_images = len(images)
    #  creating a grid of subplots to display multiple images along with their bounding boxes
    fig, axes = plt.subplots(nrows=1, ncols=num_images, figsize=(10 * num_images, 10))

    for i in range(num_images):
        # Convert from tensor to PIL image format
        img = images[i].permute(1, 2, 0)
        # display the img
        axes[i].imshow(img)
        for box in boxes[i]:
            # Ensure there are exactly four values in the bounding box
            if len(box) == 4:
                # unpack cords
                x, y, w, h = box
                # create rectangle and add it to plot
                rect = plt.Rectangle((x, y), w, h, linewidth=2, edgecolor='red', facecolor='none')
                axes[i].add_patch(rect)
        # turn off axis and show plot
        axes[i].axis('off')
    plt.show()


# Get a batch of data
images, boxes = next(iter(data_loader))

# Plot the images with bounding boxes
plot_images_with_boxes(images, boxes)

# Print some sample data from the first 5 samples
for i in range(5):
    print("Sample", i + 1)
    print("Image shape:", images[i].shape)
    print("Bounding boxes:")
    for box in boxes[i]:
        print("  ", box)
    print()
