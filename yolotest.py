import cv2
from ultralytics import YOLO

# Load the YOLOv8 model (replace 'test.pt' with your trained model)
model = YOLO("test.pt")

# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run inference
    results = model(frame)[0]  # Get first result

    # Draw detections
    if results.boxes is not None:
        for box in results.boxes:
            # Get coordinates in xyxy format
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            conf = box.conf[0].item()
            cls_id = int(box.cls[0].item())

            # Draw rectangle
            cv2.rectangle(frame, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (255, 0, 0), 2)
            label = f"{model.names[cls_id]} {conf:.2f}"
            cv2.putText(frame, label, (xyxy[0], xyxy[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Show result
    cv2.imshow('YOLOv8 Detection', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
