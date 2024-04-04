import cv2
import torch
import numpy as np

# Load your trained YOLO model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')

# Initiate camera feed
cap = cv2.VideoCapture(0)  # if you have multiple webcams, the parameter can be 0 or 1 or 2 ...

try:
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        init_frame = frame

        # Our operations on the frame come here
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model(frame)  # apply YOLO on image

        # Get box parameters
        for *box, score, class_id in results.xyxy[0].tolist():
            x1, y1, x2, y2 = map(int, box)
            if class_id == 0:
                cv2.rectangle(init_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            else:
                cv2.rectangle(init_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)


        # Display the resulting frame
        cv2.imshow('Processing Video Feed', init_frame)

        # Exit loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
except Exception as e:
    # Handling any exceptions
    print(f"An error occurred: {e}")
finally:
    # When everything done, release the capture
    cap.release()
    # Destroy all windows
    cv2.destroyAllWindows()