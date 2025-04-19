# src/utils/mtcnn_utils.py

import cv2
import torch
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize models
# Initialize models with lower thresholds
mtcnn = MTCNN(keep_all=True, device=device, thresholds=[0.2, 0.4, 0.6])
inception = InceptionResnetV1(pretrained='vggface2').eval().to(device)


def extract_face_embedding(image_np):
    """
    Takes a numpy image, detects face using MTCNN, and returns embedding.
    """
    img = Image.fromarray(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))
    faces = mtcnn(img)

    if faces is None or len(faces) == 0:
        raise ValueError("No face detected")

    face_embedding = inception(faces[0].unsqueeze(0).to(device)).detach().cpu().numpy()[0]
    return face_embedding
