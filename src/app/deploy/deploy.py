import os.path
import cv2
import torch


def init_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = 'best.pt'
    model = torch.hub.load(os.path.join(os.getcwd(), 'yolov5'), 'custom', path=model_path, source='local')
    model.to(device).eval()
    return model


def pass_model(model, frame):
    init_frame = frame
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model(frame)  # apply YOLO on image

    # Get box parameters
    for *box, score, class_id in results.xyxy[0].tolist():
        x1, y1, x2, y2 = map(int, box)
        if class_id == 0:
            cv2.rectangle(init_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        else:
            cv2.rectangle(init_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

    return init_frame
