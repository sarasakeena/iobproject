import os

# Database configuration
DB_CONFIG = {
    "dbname": os.getenv("DB_NAME", "face_auth_db"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", "sakeena123"),
    "host": os.getenv("DB_HOST", "localhost")
}

# Reference face image path (update this)
REF_IMAGE = os.path.join(os.path.dirname(__file__), "ref_image.jpg")

class Config:
    DATABASE_PATH = os.path.join(os.path.dirname(__file__), 'face_hashes.db')
    LSH_HASH_SIZE = 16       # Number of bits in the LSH hash
    LSH_NUM_HASHTABLES = 4   # Number of hash tables for LSH
    EMBEDDING_DIM = 128      # Dimension of face embeddings (e.g., FaceNet)
    HAMMING_THRESHOLD = 4       # Max allowed Hamming distance for matches