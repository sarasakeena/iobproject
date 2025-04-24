# save_img_hash.py
import os
import hashlib
import numpy as np # type: ignore
import face_recognition
from database.db_utils import save_face_hash  # Make sure this works with your DB

def compute_sha512_hash(image_path):
    """Compute the SHA-512 hash of an image file."""
    try:
        with open(image_path, 'rb') as f:
            image_data = f.read()
            return hashlib.sha512(image_data).hexdigest()
    except Exception as e:
        print(f"Error hashing {image_path}: {e}")
        return None

def process_image(image_path, encoding_dir="face_data"):
    """Process a single image: hash + face encode + save."""
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    
    # 1. Compute SHA-512 hash
    image_hash = compute_sha512_hash(image_path)
    if not image_hash:
        return

    # 2. Load and encode face
    image = face_recognition.load_image_file(image_path)
    encodings = face_recognition.face_encodings(image)

    if encodings:
        # 3. Save encoding as .npz
        os.makedirs(encoding_dir, exist_ok=True)
        npz_path = os.path.join(encoding_dir, f"{image_name}.npz")
        np.savez(npz_path, encoding=encodings[0])
        print(f"✅ Saved encoding to {npz_path}")

        # 4. Save hash in DB
        save_face_hash(image_path, image_hash)
        print(f"🔐 Saved SHA-512 hash for {image_name}")
    else:
        print(f"❌ No face found in {image_path}")

def main():
    image_dir = "ref_images"  # Can be a folder or single file

    if os.path.isfile(image_dir):
        process_image(image_dir)
    else:
        for filename in os.listdir(image_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(image_dir, filename)
                process_image(image_path)

if __name__ == "__main__":
    main()
