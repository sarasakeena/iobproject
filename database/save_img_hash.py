# save_img_hash.py
import os
from PIL import Image  # For image processing (install with `pip install Pillow`)
import imagehash  # For hashing (install with `pip install imagehash`)
from database.db_utils import save_face_hash  # Absolute import


def compute_image_hash(image_path):
    """Compute the perceptual hash of an image."""
    try:
        with Image.open(image_path) as img:
            # Example: Use Average Hash (you can choose other algorithms)
            return str(imagehash.average_hash(img))
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None


def main():
    # Directory containing images to process
    image_dir = "ref_image.jpeg"  # Replace with your directory

    # Process all images in the directory
    for filename in os.listdir(image_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(image_dir, filename)
            image_hash = compute_image_hash(image_path)

            if image_hash:
                save_face_hash(image_path, image_hash)
                print(f"Saved hash for {filename}")

