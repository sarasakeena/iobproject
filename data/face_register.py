import sys
import os
import cv2
import torch
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf8')



# Load models
mtcnn = MTCNN(keep_all=False)
resnet = InceptionResnetV1(pretrained='vggface2').eval()

# Get name and password
name = sys.argv[1]
password = sys.argv[2]

# Create user folder
user_folder = f'users/{name}'
os.makedirs(user_folder, exist_ok=True)
image_path = os.path.join(user_folder, 'face.jpg')

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Webcam not accessible!")
    exit(1)

print("[INFO] Press 'c' to capture your face.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Capture Face", frame)
    key = cv2.waitKey(1)

    if key == ord('c'):
        cv2.imwrite(image_path, frame)
        break

cap.release()
cv2.destroyAllWindows()

# Load and embed face
img = cv2.imread(image_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

face = mtcnn(img_rgb)
if face is not None:
    embedding = resnet(face.unsqueeze(0)).detach().numpy()

    # Save embedding and password
    emb_path = f'embeddings/{name}.npz'
    os.makedirs('embeddings', exist_ok=True)
    np.savez(emb_path, embedding=embedding, password=password)

    print(f"[✅] Registered {name} successfully.")
    
else:
    print("[❌] No face detected.")