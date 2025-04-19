# lsh/lsh_manager.py
import numpy as np
from lshash import LSHash  # type: ignore
from config import Config

# Initialize LSH
lsh = LSHash(
    hash_size=Config.LSH_HASH_SIZE,
    input_dim=Config.EMBEDDING_DIM,
    num_hashtables=Config.LSH_NUM_HASHTABLES
)

def embedding_to_hash(embedding: np.ndarray) -> str:
    """Convert face embedding to LSH hash (hex string)."""
    hash_bin = lsh.index(embedding.flatten(), extra_data=None)[0][0][0]
    return hash_bin.hex()

def hamming_distance(hash1: str, hash2: str) -> int:
    """Compute Hamming distance between two hex hashes."""
    xor = int(hash1, 16) ^ int(hash2, 16)
    return bin(xor).count('1')