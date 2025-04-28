# scripts/preprocessing.py

import os
import cv2
import numpy as np
from tqdm import tqdm
import shutil

def normalize_patch(img):
    """Normalize pixel intensities to 0-255 range."""
    img_norm = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    return img_norm

def is_tile_informative(img, threshold=10):
    """Check if a tile has enough texture/information."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return np.std(gray) > threshold  # higher std dev = more texture

def preprocess_tiles(input_dir, output_dir, threshold=10):
    """Normalize and filter tiles, saving cleaned versions."""
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)  # Clean slate
    os.makedirs(output_dir, exist_ok=True)

    class_folders = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]

    for class_name in tqdm(class_folders, desc="Preprocessing classes"):
        class_input_path = os.path.join(input_dir, class_name)
        class_output_path = os.path.join(output_dir, class_name)
        os.makedirs(class_output_path, exist_ok=True)

        tile_files = os.listdir(class_input_path)

        for tile_file in tqdm(tile_files, desc=f"Processing {class_name}", leave=False):
            tile_path = os.path.join(class_input_path, tile_file)
            img = cv2.imread(tile_path)

            if img is None:
                continue  # skip bad files

            if is_tile_informative(img, threshold=threshold):
                img_norm = normalize_patch(img)
                save_path = os.path.join(class_output_path, tile_file)
                cv2.imwrite(save_path, img_norm)
