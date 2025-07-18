#scripts/preprocessing.py
import os
import cv2
import numpy as np
from tqdm import tqdm
import shutil

def is_tile_informative(img, threshold=10):
    """Check if a tile has enough texture/information."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return np.std(gray) > threshold

def normalize_patch(img):
    """Normalize pixel intensities to 0-255 range."""
    img_norm = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    return img_norm

def augment_and_save(img, save_base_path, file_basename, n_augments=5, transform=None):
    for i in range(n_augments):
        aug_img = transform(img)
        aug_img_np = aug_img.mul(255).byte().numpy().transpose(1, 2, 0)
        aug_filename = f"{file_basename}_aug{i}.png"
        cv2.imwrite(os.path.join(save_base_path, aug_filename), cv2.cvtColor(aug_img_np, cv2.COLOR_RGB2BGR))

def preprocess_tiles(input_dir, output_dir, threshold=10, n_augments=5, transform=None):
    """Normalize, augment, and filter tiles, saving augmented versions."""
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    class_folders = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]

    total_in = 0
    total_kept = 0

    for class_name in tqdm(class_folders, desc="Preprocessing classes"):
        class_input_path = os.path.join(input_dir, class_name)
        class_output_path = os.path.join(output_dir, class_name)
        os.makedirs(class_output_path, exist_ok=True)

        tile_files = os.listdir(class_input_path)
        total_in += len(tile_files)

        for tile_file in tqdm(tile_files, desc=f"Processing {class_name}", leave=False):
            tile_path = os.path.join(class_input_path, tile_file)
            img = cv2.imread(tile_path)

            if img is None:
                continue

            if is_tile_informative(img, threshold=threshold):
                img_norm = normalize_patch(img)
                img_rgb = cv2.cvtColor(img_norm, cv2.COLOR_BGR2RGB)
                file_basename = os.path.splitext(tile_file)[0]

                # Save the original normalized tile
                base_save_path = os.path.join(class_output_path, f"{file_basename}_orig.png")
                cv2.imwrite(base_save_path, img_norm)
                total_kept += 1

                # Save augmented versions
                if transform is not None:
                    augment_and_save(img_rgb, class_output_path, file_basename, n_augments=n_augments, transform=transform)
                    total_kept += n_augments

    print(f"\n✅ Preprocessing complete. Kept {total_kept}/{total_in} tiles (including augmentations)")

def augment_and_save(img, save_base_path, file_basename, n_augments=5, transform=None):
    for i in range(n_augments):
        aug_img = transform(img)
        aug_img_np = aug_img.mul(255).byte().numpy().transpose(1, 2, 0)
        aug_filename = f"{file_basename}_aug{i}.png"
        cv2.imwrite(os.path.join(save_base_path, aug_filename), cv2.cvtColor(aug_img_np, cv2.COLOR_RGB2BGR))


