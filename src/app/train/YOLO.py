import torch
import os
import shutil
import subprocess

# Define your file locations
yolov5_loc = os.path.join(os.getcwd(), "yolov5")
data_loc = os.path.join(os.getcwd(), "data")
models_loc = os.path.join(os.getcwd(), "models")

# Make sure 'yolov5s' is downloaded
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Define your training command
train_cmd = f"python {os.path.join(yolov5_loc, 'train.py')} --img 640 --batch 16 --epochs 5"
# Point to your custom data configuration, model configuration, weights, etc.
train_cmd += f" --data {os.path.join(data_loc, 'data.yaml')} --cfg {os.path.join(yolov5_loc, 'models', 'yolov5s.yaml')} --weights '' --name yolov5s_results"

# Run the training command
subprocess.run(train_cmd, shell=True)

# Your trained model will be saved under `runs/train/yolov5s_results`
# Afterwards, you can copy it to the location you want:
trained_model_src = os.path.join(yolov5_loc, "runs", "train", "yolov5s_results", "weights", "best.pt")

# Be sure models_loc directory exists
if not os.path.exists(models_loc):
    os.makedirs(models_loc)
shutil.copy(trained_model_src, models_loc)

# Print the best model name
print('The best model is:', os.path.join(models_loc, 'best.pt'))
