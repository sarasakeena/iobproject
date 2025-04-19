from time import time
import cv2
import cvzone
import numpy as np
import os
import shutil
from cvzone.FaceDetectionModule import FaceDetector
import mediapipe as mp

# =============== Configuration ===============
DATASET_ROOT = 'datasett'
RAW_FOLDER = os.path.join(DATASET_ROOT, 'datacollect')
PROCESSED_FOLDER = os.path.join(DATASET_ROOT, 'processed')
TRAIN_RATIO = 0.8
CLASS_ID = 0  # 0 for fake, 1 for real
CONFIDENCE_THRESHOLD = 0.8
BLUR_THRESHOLD = 35
USER_ID_PREFIX = 'I743'
CAM_WIDTH, CAM_HEIGHT = 640, 480

# =============== Setup Directories ===============
os.makedirs(RAW_FOLDER, exist_ok=True)
os.makedirs(os.path.join(PROCESSED_FOLDER, 'train', 'real'), exist_ok=True)
os.makedirs(os.path.join(PROCESSED_FOLDER, 'train', 'fake'), exist_ok=True)
os.makedirs(os.path.join(PROCESSED_FOLDER, 'test', 'real'), exist_ok=True)
os.makedirs(os.path.join(PROCESSED_FOLDER, 'test', 'fake'), exist_ok=True)

# =============== Initialize Components ===============
cap = cv2.VideoCapture(0)
cap.set(3, CAM_WIDTH)
cap.set(4, CAM_HEIGHT)
detector = FaceDetector()

class DatasetManager:
    def __init__(self, processed_dir=PROCESSED_FOLDER):
        self.processed_dir = processed_dir

    def get_split_counts(self):
        counts = {}
        for split in ['train', 'test']:
            counts[split] = {
                'real': len(os.listdir(os.path.join(self.processed_dir, split, 'real'))),
                'fake': len(os.listdir(os.path.join(self.processed_dir, split, 'fake')))
            }
        return counts

    def balance_splits(self, min_samples=10):
        counts = self.get_split_counts()
        for class_name in ['real', 'fake']:
            if counts['test'][class_name] < min_samples:
                self._transfer_samples('train', 'test', class_name, min_samples - counts['test'][class_name])
            if counts['train'][class_name] < min_samples:
                self._transfer_samples('test', 'train', class_name, min_samples - counts['train'][class_name])

    def _transfer_samples(self, src_split, dst_split, class_name, num_samples):
        src_dir = os.path.join(self.processed_dir, src_split, class_name)
        dst_dir = os.path.join(self.processed_dir, dst_split, class_name)
        files = os.listdir(src_dir)[:num_samples]
        for f in files:
            shutil.move(os.path.join(src_dir, f), os.path.join(dst_dir, f))

def organize_dataset():
    """Organize raw data into train/test splits"""
    for file in os.listdir(RAW_FOLDER):
        if file.endswith('.txt'):
            base_name = file.split('.')[0]
            txt_path = os.path.join(RAW_FOLDER, file)
            img_path = os.path.join(RAW_FOLDER, f"{base_name}.jpg")

            with open(txt_path, 'r') as f:
                class_id = int(f.read().split()[0])

            folder_type = 'train' if np.random.rand() < TRAIN_RATIO else 'test'
            class_folder = 'fake' if class_id == 0 else 'real'
            dest_dir = os.path.join(PROCESSED_FOLDER, folder_type, class_folder)

            shutil.move(img_path, os.path.join(dest_dir, f"{base_name}.jpg"))
            shutil.move(txt_path, os.path.join(dest_dir, file))

def generate_user_id():
    """Generate ID in I743XXXXXXXXX format"""
    timestamp = str(int(time() * 1000))
    return f"{USER_ID_PREFIX}{timestamp[-9:]}"

# =============== Main Loop ===============

try:
    while True:
        success, img = cap.read()
        if not success:
            break
        img_out = img.copy()
        img, bboxs = detector.findFaces(img, draw=False)
        valid_detections = []

        for bbox in bboxs:
            x, y, w, h = [int(val) for val in bbox["bbox"]]  # Convert to integers
            score = bbox["score"][0]

            if score > CONFIDENCE_THRESHOLD:
                # Convert offsets to integers before calculations
                offset_w = int(0.1 * w)
                offset_h = int(0.2 * h)

                x = max(0, x - offset_w)
                y = max(0, y - offset_h * 3)
                w = int(w + offset_w * 2)
                h = int(h + offset_h * 3.5)

                # Ensure final coordinates are integers
                x, y, w, h = map(int, (x, y, w, h))

                # Now safe to slice
                face_roi = img[y:y + h, x:x + w]  # Fixed indentation
                blur_value = int(cv2.Laplacian(face_roi, cv2.CV_64F).var())

                valid_detections.append({
                        "bbox": (x, y, w, h),
                        "blur": blur_value,
                        "score": score
                })

                # Draw UI elements
                cv2.rectangle(img_out, (x, y, w, h), (255, 0, 0), 3)
                cvzone.putTextRect(img_out,
                                   f'Score: {int(score * 100)}% Blur: {blur_value}',
                                    (x, y - 10), scale=2, thickness=3)

        # Save valid captures
        if valid_detections and all(d["blur"] > BLUR_THRESHOLD for d in valid_detections):
            user_id = generate_user_id()
            img_path = os.path.join(RAW_FOLDER, f"{user_id}.jpg")
            label_path = os.path.join(RAW_FOLDER, f"{user_id}.txt")

            cv2.imwrite(img_path, img)
            with open(label_path, 'w') as f:
                for det in valid_detections:
                    x_center = (det["bbox"][0] + det["bbox"][2]/2) / CAM_WIDTH
                    y_center = (det["bbox"][1] + det["bbox"][3]/2) / CAM_HEIGHT
                    width = det["bbox"][2] / CAM_WIDTH
                    height = det["bbox"][3] / CAM_HEIGHT
                    f.write(f"{CLASS_ID} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

            organize_dataset()
            print(f"Saved and organized: {user_id}")

        cv2.imshow("Data Collection", img_out)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    # Balance dataset after collection
    DatasetManager().balance_splits(min_samples=10)