import cv2
from deepface import DeepFace

def capture_face_embedding(image_path: str = "temp_capture.jpg") -> list:
    import cv2
    from deepface import DeepFace

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")
    
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise RuntimeError("Failed to capture image from webcam.")

    cv2.imwrite(image_path, frame)
    cv2.imshow("Captured Image", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    embedding = DeepFace.represent(img_path=image_path, model_name="Facenet", enforce_detection=False)[0]["embedding"]

    return embedding
