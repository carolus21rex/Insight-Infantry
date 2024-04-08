# EGGIS AI - Expendable Geospatial Guardian Intelligence System v0.6

--------
##  Usage:

--------
### Requirements:
- git: used for cloning yolov5
- torch: used to train and deploy the AI
- shutil: used for automated scripting in `src/app/train/YOLO.py` and `src/app/deploy/deploy.py`
- opencv: used in `src/app/deploy/deploy.py` to get a camera feed, mostly for demonstration purposes
- numpy: used by torch to make tensors, a structure that is AI specific and intended for cuda (GPU) usage
- dataset: used to train the AI, place in `src/app/train/data`. The source of the dataset is `https://www.kaggle.com/code/ugorjiir/gundetect`, the modified dataset can be found at `https://drive.google.com/file/d/1pxaRp6xoXetNCNZ5A2MjNqESCjyD867I/view?usp=drive_link`
- best.pth: If you intend to replicate the demonstration video, the model used can be found at: `https://drive.google.com/file/d/1_ozhd9txtZKq7TjH7xP_ixfCzRAz4tq4/view?usp=drive_link`
  
--------
### Training:
1. Run src/app/train/RUNME.py as administrator/su
   - This will clone yolov5 into `cwd/yolov5`
2. Copy the dataset from the google drive, seen in requirements, into `src/app/train/data`
3. Run YOLO.py as administrator/su
4. After training (expect it to take a few hours) your model can be found in `src/app/train/models` as well as `src/app/train/yolov5/runs/train/yolo5s_results(X)/weights`

--------
### Deployment:
1. Copy best.pth into `src/app/deploy` either from step 4 of training or `https://drive.google.com/file/d/1_ozhd9txtZKq7TjH7xP_ixfCzRAz4tq4/view?usp=drive_link`
2. Run `src/app/deploy/deploy.py` with camera enabled
