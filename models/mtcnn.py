import cv2
import numpy as np
import torch
import dlib
from facenet_pytorch import MTCNN, InceptionResnetV1
from transformers import DPTImageProcessor, DPTForDepthEstimation
from scipy.spatial import distance as dist
from imutils import face_utils
import time
from skimage.feature import local_binary_pattern

# ========== Device Setup ==========
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ========== Model Initialization ==========
mtcnn = MTCNN(keep_all=True, device=device)
inception = InceptionResnetV1(pretrained='vggface2').eval().to(device)
processor = DPTImageProcessor.from_pretrained("Intel/dpt-hybrid-midas")
model = DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas").eval().to(device)

# ========== Dlib Predictor ==========
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(r"C:\iobproject\models\shape_predictor_68_face_landmarks.dat")

# ========== Constants ==========
EYE_AR_THRESH = 0.20
EYE_AR_CONSEC_FRAMES = 1
BLINK_THRESHOLD = 1
DEPTH_STD_THRESH = 2.0
MOTION_THRESHOLD = 1.0
CONFIDENCE_THRESHOLD = 0.4
LIVENESS_TIME_WINDOW = 1.0

# ========== Face Memory ==========
known_face_encodings = []
known_face_names = []

# ========== Functions ==========
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def enhanced_lighting_adjustment(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    gamma = 0.7
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    cl = cv2.LUT(cl, table)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

def analyze_texture(face_roi):
    gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, 8, 1, method='uniform')
    hist, _ = np.histogram(lbp, bins=10, range=(0, 10))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)
    return -np.sum(hist * np.log(hist + 1e-7))

# ========== Liveness Class ==========
class LivenessDetector:
    def __init__(self):
        self.blink_count = 0
        self.last_blink_time = time.time()
        self.face_movement = 0
        self.prev_face_pos = None
        self.liveness_start_time = time.time()
    
    def update_blink(self):
        self.blink_count += 1
        self.last_blink_time = time.time()
    
    def update_movement(self, current_pos):
        if self.prev_face_pos is not None:
            movement = np.linalg.norm(np.array(current_pos) - np.array(self.prev_face_pos))
            self.face_movement += movement
        self.prev_face_pos = current_pos
    
    def check_liveness(self):
        time_elapsed = time.time() - self.liveness_start_time
        if time_elapsed < LIVENESS_TIME_WINDOW:
            return False
        has_blinked = self.blink_count >= BLINK_THRESHOLD
        has_moved = self.face_movement > 10
        recent_blink = (time.time() - self.last_blink_time) < LIVENESS_TIME_WINDOW
        return has_blinked and has_moved and recent_blink

# ========== Main ==========
liveness_detector = LivenessDetector()
video_capture = cv2.VideoCapture(0)
blink_counter = 0

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    frame = enhanced_lighting_adjustment(frame)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    boxes, _ = mtcnn.detect(rgb_frame)
    faces = mtcnn(rgb_frame)

    if boxes is not None and faces is not None:
        for i, (box, face) in enumerate(zip(boxes, faces)):
            if face is None:
                continue

            x_min, y_min, x_max, y_max = map(int, box)
            h, w, _ = rgb_frame.shape
            x_min, y_min = max(0, x_min), max(0, y_min)
            x_max, y_max = min(w, x_max), min(h, y_max)

            face_center = ((x_min + x_max) // 2, (y_min + y_max) // 2)
            liveness_detector.update_movement(face_center)

            face_embedding = inception(face.unsqueeze(0).to(device))
            distances = [np.linalg.norm(face_embedding.detach().cpu().numpy() - known_face)
                         for known_face in known_face_encodings]
            min_distance = np.min(distances) if distances else 1.0
            confidence = 1 - min_distance
            name = "Unknown"
            if confidence > CONFIDENCE_THRESHOLD:
                name = known_face_names[np.argmin(distances)]

            face_roi = rgb_frame[y_min:y_max, x_min:x_max]
            if face_roi.shape[0] == 0 or face_roi.shape[1] == 0:
                continue

            inputs = processor(images=face_roi, return_tensors="pt").to(device)
            with torch.no_grad():
                depth_map = model(**inputs).predicted_depth[0].cpu().numpy()

            depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-7)
            depth_std = np.std(depth_map)
            texture_score = analyze_texture(frame[y_min:y_max, x_min:x_max])
            is_live = liveness_detector.check_liveness()
            spoofed_status = "Real" if is_live else "Spoofed"

            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), 
                          (0, 255, 0) if is_live else (0, 0, 255), 2)
            cv2.putText(frame, f"{name} ({confidence:.2f}) - {spoofed_status}", 
                        (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                        (0, 255, 0) if is_live else (0, 0, 255), 2)
            cv2.putText(frame, f"Blinks: {liveness_detector.blink_count}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(frame, f"Movement: {liveness_detector.face_movement:.1f}", (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(frame, f"Texture: {texture_score:.2f}", (10, 90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    #rects = detector(gray)
    #for rect in rects:
        #shape = predictor(gray, rect)
        #shape = face_utils.shape_to_np(shape)
        #left_ear = eye_aspect_ratio(shape[36:42])
        #right_ear = eye_aspect_ratio(shape[42:48])
        #ear = (left_ear + right_ear) / 2.0

        #if ear < EYE_AR_THRESH:
            #blink_counter += 1
        #else:
            #if blink_counter >= EYE_AR_CONSEC_FRAMES:
                #liveness_detector.update_blink()
            #blink_counter = 0

    cv2.imshow("Enhanced Face Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
