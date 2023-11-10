import cv2
from src.app import camera

# Create a window and display the image with rectangles
cv2.namedWindow("Image with Rectangles")


while True:

    if camera.build_camera_image() == -1:
        break


cv2.waitKey(1)
cv2.destroyAllWindows()
