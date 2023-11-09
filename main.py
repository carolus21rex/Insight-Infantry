import cv2
import numpy as np
from src.app import camera

# Create a window and display the image with rectangles
cv2.namedWindow("Image with Rectangles")


# Initialize the camera capture
cap = cv2.VideoCapture(0)
mode = 0
while True:

    if camera.build_camera_image(cap) == -1:
        break


cv2.waitKey(1)
cv2.destroyAllWindows()
