import cv2
import numpy as np
import os

net = cv2.dnn.readNetFromCaffe(
    os.path.join(os.getcwd(), "MobileNetSSD_deploy.prototxt"),
    os.path.join(os.getcwd(), "MobileNetSSD_deploy.caffemodel")
)


CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]
#CLASSES = ["background", "boomstick", "person"]
cap = cv2.VideoCapture(0)

# Set the camera frame dimensions to 300 x 300
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 300)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 300)

while True:
    ret, frame = cap.read()

    # Check if the frame is not empty
    if not ret or frame is None:
        print("Error: Unable to capture frame")
        break

    blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.2:
            class_id = int(detections[0, 0, i, 1])
            label = CLASSES[class_id]

            box = detections[0, 0, i, 3:7] * np.array([300, 300, 300, 300])
            (startX, startY, endX, endY) = box.astype("int")

            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("Object Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
