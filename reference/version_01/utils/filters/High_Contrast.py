import cv2

def modify_image(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply GaussianBlur to reduce noise and improve edge detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply Canny edge detector with lower thresholds
    edges = cv2.Canny(blurred, 20, 60)  # Adjust these thresholds as needed

    # Find contours in the edges
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw a red outline around the detected contours
    result_image = image.copy()
    cv2.drawContours(result_image, contours, -1, (0, 0, 255), 2)

    return result_image


