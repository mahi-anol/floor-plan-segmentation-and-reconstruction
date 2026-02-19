import os
import cv2
import glob
import shutil
import random
import numpy as np
from tqdm import tqdm
from pathlib import Path


INPUT_ROOT = "./artifacts/processed-data"
OUTPUT_ROOT = "./artifacts/augmented"
AUGMENTATIONS_PER_IMAGE = 3  # Each original gets 3 separate folders
SUPPORTED_EXT = (".png", ".jpg", ".jpeg", ".tif", ".tiff")

# Optional palette (as RGB).
PALETTE_RGB = {
    "Background": (0, 0, 0),
    "Wall": (255, 0, 0),
}

def rgb_to_bgr_palette(palette_rgb):
    """Convert a name->(R,G,B) palette into (B,G,R) values for OpenCV."""
    return {name: (col[2], col[1], col[0]) for name, col in palette_rgb.items()}

PALETTE_BGR = rgb_to_bgr_palette(PALETTE_RGB)

def is_image_file(fname):
    return fname.lower().endswith(SUPPORTED_EXT)

def find_image_and_mask(folder_path):
    """Return (img_path, mask_path) or (None, None) if not found."""
    files = sorted([f for f in os.listdir(folder_path) if is_image_file(f)])
    mask_candidates = [f for f in files if "_gt_" in f.lower()]
    img_candidates = [f for f in files if "_gt_" not in f.lower()]

    if not img_candidates or not mask_candidates:
        return None, None

    return os.path.join(folder_path, img_candidates[0]), os.path.join(folder_path, mask_candidates[0])

def read_image_keep_alpha(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise IOError(f"Could not read image: {path}")
    return img

def ensure_3channel_for_pixel_ops(img):
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img

def apply_pixel_modifications(image):
    single_channel = (image.ndim == 2)
    working = ensure_3channel_for_pixel_ops(image.copy())

    choice = random.random()
    if choice < 0.3:
        kernel_size = random.choice([3, 5])
        working = cv2.GaussianBlur(working, (kernel_size, kernel_size), 0)
    elif choice < 0.6:
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        working = cv2.filter2D(working, -1, kernel)

    if single_channel:
        return cv2.cvtColor(working, cv2.COLOR_BGR2GRAY)
    return working

def rotate_and_expand(image, angle, is_mask=False):
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)

    cos = abs(M[0, 0])
    sin = abs(M[0, 1])
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_CUBIC

    if image.ndim == 2:
        border_val = 0
    else:
        try:
            val = image[0, 0].tolist()
            border_val = [int(v) for v in val]
        except Exception:
            border_val = [0] * image.shape[2]

    return cv2.warpAffine(image, M, (nW, nH), flags=interp, borderValue=border_val)

def apply_random_geometry(image, mask):
    if random.random() > 0.5:
        image, mask = cv2.flip(image, 1), cv2.flip(mask, 1)
    if random.random() > 0.5:
        image, mask = cv2.flip(image, 0), cv2.flip(mask, 0)

    angle = random.uniform(0, 360)
    image = rotate_and_expand(image, angle, is_mask=False)
    mask = rotate_and_expand(mask, angle, is_mask=True)
    image = apply_pixel_modifications(image)
    return image, mask

def process_set(input_dir, output_dir, set_name):
    """Helper to apply augmentation to a specific subset (train or test)."""
    if not os.path.exists(input_dir):
        print(f"{set_name} directory not found! Skipping.")
        return 0

    sample_folders = sorted([f for f in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, f))])
    
    new_folder_id = 0
    missing_count = 0

    for folder in tqdm(sample_folders, desc=f"Augmenting {set_name} Set"):
        folder_path = os.path.join(input_dir, folder)
        img_path, mask_path = find_image_and_mask(folder_path)

        if img_path is None or mask_path is None:
            missing_count += 1
            continue

        try:
            img_ptr = read_image_keep_alpha(img_path)
            mask_ptr = read_image_keep_alpha(mask_path)
        except Exception:
            continue

        # 1. Save Original
        current_out_path = os.path.join(output_dir, str(new_folder_id))
        os.makedirs(current_out_path, exist_ok=True)
        base_img_name = os.path.basename(img_path)
        base_mask_name = os.path.basename(mask_path)
        cv2.imwrite(os.path.join(current_out_path, base_img_name), img_ptr)
        cv2.imwrite(os.path.join(current_out_path, base_mask_name), mask_ptr)
        new_folder_id += 1

        # 2. Save Augmented
        for _ in range(AUGMENTATIONS_PER_IMAGE):
            aug_img, aug_mask = apply_random_geometry(img_ptr.copy(), mask_ptr.copy())
            aug_out_path = os.path.join(output_dir, str(new_folder_id))
            os.makedirs(aug_out_path, exist_ok=True)
            cv2.imwrite(os.path.join(aug_out_path, base_img_name), aug_img)
            cv2.imwrite(os.path.join(aug_out_path, base_mask_name), aug_mask)
            new_folder_id += 1
            
    return new_folder_id

def run_augmentation():
    # Define paths
    train_in = os.path.join(INPUT_ROOT, "train")
    test_in = os.path.join(INPUT_ROOT, "test")
    train_out = os.path.join(OUTPUT_ROOT, "train")
    test_out = os.path.join(OUTPUT_ROOT, "test")

    # Clean previous output
    if os.path.exists(OUTPUT_ROOT):
        shutil.rmtree(OUTPUT_ROOT)

    # Process Train
    train_count = process_set(train_in, train_out, "Train")
    
    # Process Test (Now applies augmentation instead of just copying)
    test_count = process_set(test_in, test_out, "Test")

    print(f"\n--- Processing Complete ---")
    print(f"Train folders created: {train_count}")
    print(f"Test folders created: {test_count}")

if __name__ == "__main__":
    run_augmentation()