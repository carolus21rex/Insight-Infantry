import torch
import os
import shutil
import subprocess

# Define your file locations
yolov5_loc = os.path.join(os.getcwd(), "yolov5")
data_loc = os.path.join(os.getcwd(), "data")
models_loc = os.path.join(os.getcwd(), "models")
hyp_yaml = os.path.join(os.getcwd(), 'hyp.scratch.yaml')  # Make sure that hyp.scratch.yaml is properly edited

# Here we check if CUDA is available and set our device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Loading the model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Move the model to the device
model = model.to(device)

# Remove the loop over folds as you're not doing cross validation anymore
data_yaml = os.path.join(data_loc, 'data.yaml')  # assuming data.yaml is your data file
train_cmd = (f"python {os.path.join(yolov5_loc, 'train.py')} --img 640 --batch 16 --epochs 5 "
             f"--data {data_yaml} --cfg {os.path.join(yolov5_loc, 'models', 'yolov5s.yaml')} "
             f"--weights '' --hyp {hyp_yaml} --lr 0.001 --name yolov5s_results")
subprocess.run(train_cmd, shell=True)

# Your trained model will be saved under `runs/train/yolov5s_results`
# Afterwards, you can copy it to the location you want:
trained_model_src = os.path.join(yolov5_loc, "runs", "train", "yolov5s_results")

# Be sure models_loc directory exists
if not os.path.exists(models_loc):
    os.makedirs(models_loc)

shutil.move(trained_model_src, models_loc)
