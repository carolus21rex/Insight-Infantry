import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

# Define the backbone
backbone = torchvision.models.mobilenet_v2(pretrained=False).features
backbone.out_channels = 1280

# Define the anchor generator
anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                   aspect_ratios=((0.5, 1.0, 2.0),))

# Define the ROI pooler
roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
                                                output_size=7,
                                                sampling_ratio=2)

# Load the trained model
model = FasterRCNN(backbone,
                    num_classes=3,
                    rpn_anchor_generator=anchor_generator,
                    box_roi_pool=roi_pooler)
model.load_state_dict(torch.load('trained_model.pth'))
model.eval()

# Example input for deployment
x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]

# Make predictions
predictions = model(x)
print(predictions)
