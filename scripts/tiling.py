# scripts/tiling.py

import os
from PIL import Image
from tqdm import tqdm

def tile_flat_image(image_path, output_folder, tile_size=512, overlap=0):
    """Tile a flat (normal) image into patches and save."""
    try:
        img = Image.open(image_path)
        width, height = img.size
    except Exception as e:
        print(f"❌ Error opening {image_path}: {e}")
        return

    os.makedirs(output_folder, exist_ok=True)
    count = 0

    for y in range(0, height, tile_size - overlap):
        for x in range(0, width, tile_size - overlap):
            box = (x, y, x + tile_size, y + tile_size)
            tile = img.crop(box)

            # Only save tiles of full size (ignore partial edge tiles)
            if tile.size[0] == tile_size and tile.size[1] == tile_size:
                tile.save(os.path.join(output_folder, f'tile_{count}_x{x}_y{y}.png'))
                count += 1

    print(f"✅ Tiled {count} patches from {os.path.basename(image_path)}")

def tile_images_from_folder(raw_folder, output_folder, tile_size=224, overlap=0):
    """Tile all images from a folder structure like /raw/classname/*.tif."""
    allowed_extensions = ('.tif', '.tiff', '.png', '.jpg', '.jpeg')
    
    classes = [d for d in os.listdir(raw_folder) if os.path.isdir(os.path.join(raw_folder, d))]

    for class_name in tqdm(classes, desc="Tiling classes"):
        class_path = os.path.join(raw_folder, class_name)
        class_output = os.path.join(output_folder, class_name)
        os.makedirs(class_output, exist_ok=True)

        image_files = [f for f in os.listdir(class_path) if f.lower().endswith(allowed_extensions)]

        for image_file in tqdm(image_files, desc=f"Tiling images in {class_name}", leave=False):
            image_path = os.path.join(class_path, image_file)
            tile_flat_image(image_path, class_output, tile_size=tile_size, overlap=overlap)
