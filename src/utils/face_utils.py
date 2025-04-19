# src/utils/face_utils.py

import torch
import numpy as np
from PIL import Image
import cv2

from facenet_pytorch import MTCNN, InceptionResnetV1
from scipy.spatial.distance import cosine

# Load models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mtcnn = MTCNN(keep_all=True, device=device)
inception = InceptionResnetV1(pretrained='vggface2').eval().to(device)


def get_face_embedding(image: np.ndarray):
    """Detect face and return 512D embedding"""
    img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    faces = mtcnn(img)

    if faces is None or len(faces) == 0:
        raise ValueError("No face detected")

    face_tensor = faces[0].unsqueeze(0).to(device)
    embedding = inception(face_tensor).detach().cpu().numpy()[0]
    return embedding


def compare_embeddings(known_embeddings, captured_embedding, threshold=0.10):
    """Compare embeddings using cosine distance"""
    distances = [cosine(captured_embedding, known) for known in known_embeddings]
    min_dist = min(distances)
    return min_dist, min_dist < threshold
