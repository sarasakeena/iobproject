import cv2
from cvzone.FaceDetectionModule import FaceDetector

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Use CAP_DSHOW if on Windows
detector = FaceDetector()

while True:
    success, img = cap.read()
    if not success:
        print("❌ Failed to capture frame")
        break  # Stop if the camera feed is not working

    img, bboxs = detector.findFaces(img, draw=False)

    if bboxs:
        for bbox in bboxs:
            center = bbox["center"]
            cv2.circle(img, center, 5, (255, 0, 255), cv2.FILLED)

    cv2.imshow("Image", img)

    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
