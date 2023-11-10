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
        # Create a blank white image as a test input
        width, height = 640, 480
        image = np.ones((height, width, 3), dtype=np.uint8) * 255

        result_image = camera.add_rectangles_to_image(image)

        green_pixel_count = cv2.countNonZero(
            cv2.inRange(result_image[0:height, 0:width], (0, 200, 0), (100, 255, 100)))
        self.assertGreater(green_pixel_count, 0)

    def test_add_rectangles_to_image_red_highlight(self):
        # Create a blank white image as a test input
        width, height = 640, 480
        image = np.ones((height, width, 3), dtype=np.uint8) * 255
        x = 0
        while x + 1 < len(camera.modes):
            camera.mode = x - 1
            x += 1

            # Call the add_rectangles_to_image function
            result_image = camera.add_rectangles_to_image(image)

            self.assertIsInstance(result_image, np.ndarray)  # Check if the result is a NumPy array
            self.assertFalse(np.all(result_image == [255, 255, 255]))

            # Define the expected coordinates for the selected box (adjust as needed)
            selected_box_x = 160
            selected_box_y = height - 40
            selected_box_width = 160
            selected_box_height = 40
            if camera.mode >= 0:
                # Check if the selected box is highlighted (contains red pixels)
                selected_box = result_image[selected_box_y:selected_box_y + selected_box_height,
                               selected_box_x:selected_box_x + selected_box_width]
                red_pixel_count = cv2.countNonZero(
                    cv2.inRange(selected_box, (0, 0, 200), (100, 100, 255)))  # Adjust the color range as needed
                self.assertGreater(red_pixel_count, 0)

    def test_add_rectangles_to_image_white_text(self):
        # Create a blank white image as a test input
        width, height = 640, 480
        image = np.ones((height, width, 3), dtype=np.uint8) * 255
        x = 0
        while x + 1 < len(camera.modes):
            camera.mode = x - 1
            x += 1

            # Call the add_rectangles_to_image function
            result_image = camera.add_rectangles_to_image(image)

            self.assertIsInstance(result_image, np.ndarray)  # Check if the result is a NumPy array
            self.assertFalse(np.all(result_image == [255, 255, 255]))

            # Define the expected coordinates for the selected box (adjust as needed)
            selected_box_x = 160
            selected_box_y = height - 40
            selected_box_width = 160
            selected_box_height = 40

            if camera.mode >= 0:
                # Check for at least one white pixel in the selected box
                selected_box = result_image[selected_box_y:selected_box_y + selected_box_height,
                               selected_box_x:selected_box_x + selected_box_width]

                white_pixel_count = cv2.countNonZero(cv2.inRange(selected_box, (200, 200, 200), (255, 255, 255)))
                self.assertGreater(white_pixel_count, 0)

    def test_interpret_keypress_as_mode(self):
        camera.mode = -1

        # NaN for keyboards
        self.assertEqual(camera.interpret_keypress_as_mode(3), -1)

        # mode 1
        self.assertEqual(camera.interpret_keypress_as_mode(49), 0)

    def test_init_capture(self):
        self.assertIsNone(camera.capture)
        camera.init_capture()
        self.assertIsNotNone(camera.capture)

    def test_build_camera_image(self):
        camera.capture = 1
        self.assertEqual(camera.build_camera_image(), -1)
        camera.capture = None
        self.assertIsInstance(camera.build_camera_image(), np.ndarray)


if __name__ == '__main__':
    unittest.main()
