#Function to get image name
import os
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from PIL import Image, ImageDraw
import numpy as np
import glob
from pathlib import Path
from shapely.geometry import Polygon, Point
import matplotlib.patches as patches

import cv2


def imname(path):
    file = os.path.basename(path)
    file_name = file[:-4]
    return file_name

# function to get values to plot
def img_ann(imname):
    imr = cv2.imread(f'{os.getcwd()}\\{imname}.jpg')
#    imr = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    df = pd.read_csv(f'{os.getcwd()}\\{imname}.txt', sep=' ', names=['class', 'x1', 'y1', 'w', 'h'])

    # rescale coordinates for visualization
    df_scaled = df.iloc[:, 1:]
    df_scaled[['x1', 'w']] = df_scaled[['x1', 'w']] * imr.shape[1]
    df_scaled[['y1', 'h']] = df_scaled[['y1', 'h']] * imr.shape[0]
    return imr, df_scaled

#Function to plot images
def draw_annot(img, df):
    # create figure and axes
    fig,ax = plt.subplots(1, figsize=(15,15))
    # display the image
    ax.imshow(img)
    for box in df.values:
        # create a Rectangle patch
        rect = patches.Rectangle((box[0]-(box[2]/2),box[1]-(box[3]/2)),box[2],box[3],linewidth=3,edgecolor='g',facecolor='none')
    # add the patch to the axes
        ax.add_patch(rect)
    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    im, df_s = img_ann("0a1cfb1bb8e135c2")
    draw_annot(im, df_s)