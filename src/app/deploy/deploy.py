import torch
import cv2
import numpy as np
from torchvision.transforms import ToTensor
from SelecSLS60_B import Net

# Load the trained weights
model = Net(nClasses=1000, config='SelecSLS60_B')
model.load_state_dict(torch.load('detection_model.pth'))
model.eval()

# Initialize the web camera
cap = cv2.VideoCapture(0)


# Define any necessary pre-processing function
def pre_process(f):
    # This is just an example, you would need to adjust this for your model
    f = cv2.resize(f, (224, 224))  # Replace with your input size
    # Convert from BGR to RGB
    f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
    f = ToTensor()(f).unsqueeze(0)  # Convert to tensor and add batch dimension
    return f


while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Pre-process the captured frame
    input_data = pre_process(frame)

    # Perform the model inference
    boxes = model(input_data)

    for box in boxes:
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('frame', frame)

    # If 'q' key is pressed, break the loop and close the application
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture and destroy all windows
cap.release()
cv2.destroyAllWindows()