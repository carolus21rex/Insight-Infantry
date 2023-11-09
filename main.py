import cv2
import numpy as np


def add_rectangles_to_image(image, selected_rectangle=1):
    # Draw rectangles at the bottom
    rectangle1_color = (0, 0, 255)  # Red
    rectangle2_color = (0, 255, 0)  # Green
    height, width, _ = image.shape
    rectangle1_bottom = (50, height - 50)
    rectangle2_bottom = (width - 50, height - 50)

    cv2.rectangle(image, (rectangle1_bottom[0], rectangle1_bottom[1] - 150),
                  (rectangle1_bottom[0] + 150, rectangle1_bottom[1]), rectangle2_color,
                  2)  # not selected
    cv2.rectangle(image, (rectangle2_bottom[0] - 150, rectangle2_bottom[1] - 150),
                  (rectangle2_bottom[0], rectangle2_bottom[1]), rectangle2_color,
                  2)  # not selected

    # selected
    if selected_rectangle == 1:
        cv2.rectangle(image, (rectangle1_bottom[0], rectangle1_bottom[1] - 150),
                      (rectangle1_bottom[0] + 150, rectangle1_bottom[1]), rectangle1_color,
                      -1)
    elif selected_rectangle == 2:

        cv2.rectangle(image, (rectangle2_bottom[0] - 150, rectangle2_bottom[1] - 150),
                      (rectangle2_bottom[0], rectangle2_bottom[1]), rectangle1_color,
                      -1)

    cv2.putText(image, "Button 1", (rectangle1_bottom[0] + 10, rectangle1_bottom[1] - 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.putText(image, "Button 2", (rectangle2_bottom[0] - 140, rectangle2_bottom[1] - 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    return image


def interpret_keypress_as_mode(mode, key):
    # shift for readability
    key -= 48

    print(f"check: {key}")
    # if the user presses a value that we don't care about (aka not a number) do nothing
    if 0 > key or key >= 10:
        return mode
    return int(key)


# Create a window and display the image with rectangles
cv2.namedWindow("Image with Rectangles")
width, height = 640, 480
selected_mode = 1

# blank screen for debugging purposes
blank = np.ones((height, width, 3), dtype=np.uint8) * 255

# Initialize the camera capture
cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    if not ret:
        print("Error: Couldn't read a frame.")
        break

    # Call the function to add rectangles to the image with a selected rectangle (1 or 2)
    image = add_rectangles_to_image(frame, selected_mode)

    # Create a window and display the image with rectangles
    cv2.imshow("Image with Rectangles", image)

    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break

    if key != -1:
        selected_mode = interpret_keypress_as_mode(selected_mode, key)

    # Wait for a key press and then close the window
cv2.waitKey(1)
cv2.destroyAllWindows()
