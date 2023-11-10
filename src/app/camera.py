import cv2

# Global variables: List of mode strings, current selected mode
modes = ["Mode 1", "Mode 2", "Mode 3"]
mode = -1

# this is the camera
capture = None


def add_rectangles_to_image(image):
    # Define colors
    unselected_color = (0, 255, 0)  # Green
    selected_color = (0, 0, 255)  # Red

    # Get the number of modes and calculate button properties
    num_modes = len(modes)
    button_width = image.shape[1] // num_modes
    button_height = 40
    button_y = image.shape[0] - button_height

    for i, mode_text in enumerate(modes):
        # Determine button position
        button_x = i * button_width

        # Draw the button (unselected)
        cv2.rectangle(image, (button_x, button_y), (button_x + button_width, button_y + button_height),
                      unselected_color, 2)

        # Draw the button (selected) based on the selected mode
        if i == mode:
            cv2.rectangle(image, (button_x, button_y), (button_x + button_width, button_y + button_height),
                          selected_color, -1)

        # Add mode labels
        label_x = button_x + 10
        label_y = button_y + 25
        cv2.putText(image, mode_text, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    return image


def interpret_keypress_as_mode(key):
    global mode
    # shift for readability
    key -= 48

    print(f"check: {key} : {mode}")
    # if the user presses a value that we don't care about (aka not a number) do nothing
    if 0 > key or key >= 10:
        return mode
    return int(key-1)


def init_capture():
    global capture
    capture = cv2.VideoCapture(0)


# noinspection PyUnresolvedReferences
def build_camera_image():
    global mode, capture
    if capture is None:
        init_capture()
    ret, frame = capture.read()

    # prevent rubbish input
    if not ret:
        print("Error: Couldn't read a frame.")
        return -1

    # handle camera mode functionality
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        return -1

    # no key is -1
    if key != -1:
        mode = interpret_keypress_as_mode(key)

    image = add_rectangles_to_image(frame)

    # Create a window and display the image with rectangles
    cv2.imshow("Image with Rectangles", image)
