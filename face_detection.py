import cv2
import numpy as np
import dlib

def preprocess_image(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    equalized_image = cv2.equalizeHist(gray_image)
    gamma = 1.5
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype(np.uint8)
    return cv2.LUT(equalized_image, table)

def detect_face(image):
    detector = dlib.get_frontal_face_detector()
    preprocessed_image = preprocess_image(image)
    return detector(preprocessed_image)
