import unittest
import cv2
import numpy as np
from src.app import camera


class TestCameraFunctions(unittest.TestCase):
    def test_globals(self):
        self.assertIsInstance(camera.mode, int)

        self.assertIsNone(camera.capture)

        self.assertIsInstance(camera.modes, list)
        for mode in camera.modes:
            self.assertIsInstance(mode, str)



    def test_init_capture(self):
        camera.init_capture()

        self.assertIsNotNone(camera.capture)

        self.assertIsInstance(camera.capture, cv2.VideoCapture)

    def test_add_rectangles_to_image_green_box(self):
        pass

    def test_add_rectangles_to_image_red_highlight(self):
        # Create a blank white image as a test input
        width, height = 640, 480
        image = np.ones((height, width, 3), dtype=np.uint8) * 255

        # Set a specific mode as selected
        camera.mode = 1  # Set the second mode as selected (0-based index)

        # Call the add_rectangles_to_image function
        result_image = camera.add_rectangles_to_image(image)

        # Define the expected coordinates for the selected box (adjust as needed)
        selected_box_x = 160
        selected_box_y = height - 40
        selected_box_width = 160
        selected_box_height = 40

        # Check if the selected box is highlighted (contains red pixels)
        selected_box = result_image[selected_box_y:selected_box_y+selected_box_height, selected_box_x:selected_box_x+selected_box_width]
        red_pixel_count = cv2.countNonZero(cv2.inRange(selected_box, (0, 0, 200), (100, 100, 255)))  # Adjust the color range as needed
        self.assertGreater(red_pixel_count, 0)

        # Check for at least one white pixel in the selected box
        white_pixel_count = cv2.countNonZero(cv2.inRange(selected_box, (200, 200, 200), (255, 255, 255)))
        self.assertGreater(white_pixel_count, 0)

        # Check for the green box outline (unselected)
        unselected_box_x = 0
        unselected_box_y = height - 40
        unselected_box_width = selected_box_x
        unselected_box_height = 40

        unselected_box = result_image[unselected_box_y:unselected_box_y+unselected_box_height, unselected_box_x:unselected_box_width]
        green_pixel_count = cv2.countNonZero(cv2.inRange(unselected_box, (0, 200, 0), (100, 255, 100)))  # Adjust the color range as needed
        self.assertGreater(green_pixel_count, 0)

    def test_add_rectangles_to_image_white_text(self):
        pass


if __name__ == '__main__':
    unittest.main()
