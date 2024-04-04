import os
import subprocess

# URL of the github repository
repo_url = "https://github.com/ultralytics/yolov5.git"

# The location where the repository to be cloned
location = f"{os.getcwd()}\\yolov5"

# Clone the repository
subprocess.run(["git", "clone", repo_url, location], shell=True)
