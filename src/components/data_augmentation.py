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

# Optional palette you mentioned (as RGB). We'll convert to OpenCV BGR internally if used.
PALETTE_RGB = {
    "Background": (0, 0, 0),
    "Wall": (255, 0, 0),
}

def rgb_to_bgr_palette(palette_rgb):
    """Convert a name->(R,G,B) palette into (B,G,R) values for OpenCV comparisons."""
    return {name: (col[2], col[1], col[0]) for name, col in palette_rgb.items()}

PALETTE_BGR = rgb_to_bgr_palette(PALETTE_RGB)

def is_image_file(fname):
    return fname.lower().endswith(SUPPORTED_EXT)

def find_image_and_mask(folder_path):
    """Return (img_path, mask_path) or (None, None) if not found."""
    files = sorted([f for f in os.listdir(folder_path) if is_image_file(f)])
    # mask filenames contain '_gt_' (based on your comment)
    mask_candidates = [f for f in files if "_gt_" in f.lower()]
    img_candidates = [f for f in files if "_gt_" not in f.lower()]

    if not img_candidates or not mask_candidates:
        return None, None

    # pick first appropriate file of each list
    return os.path.join(folder_path, img_candidates[0]), os.path.join(folder_path, mask_candidates[0])

def read_image_keep_alpha(path):
    """Read image preserving channels (color or alpha)"""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise IOError(f"Could not read image: {path}")
    return img

def ensure_3channel_for_pixel_ops(img):
    """If grayscale, convert to 3-channel so pixel filters behave consistently."""
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def apply_pixel_modifications(image):
    """Randomly applies a small amount of blur or sharpness.
       Works with color or grayscale. Returns same dtype/channels as input."""
    # If single-channel, convert for kernel ops then convert back
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
    """Rotates image (or mask) and expands canvas so no portion is cut.
       For masks we use INTER_NEAREST; for images INTER_CUBIC.
       borderValue automatically matches channel count."""
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

    # Determine border value matching channels
    if image.ndim == 2:
        border_val = 0
    else:
        # use top-left pixel to preserve background-ish color; fallback to black list
        try:
            # image[0,0] might have 3 or 4 channels (B,G,R[,A])
            val = image[0, 0].tolist()
            # ensure python ints
            border_val = [int(v) for v in val]
        except Exception:
            border_val = [0] * image.shape[2]

    return cv2.warpAffine(image, M, (nW, nH), flags=interp, borderValue=border_val)

def apply_random_geometry(image, mask):
    """Applies flips, rotation, and pixel effects. Returns (image, mask)."""
    if random.random() > 0.5:
        image, mask = cv2.flip(image, 1), cv2.flip(mask, 1)
    if random.random() > 0.5:
        image, mask = cv2.flip(image, 0), cv2.flip(mask, 0)

    angle = random.uniform(0, 360)
    image = rotate_and_expand(image, angle, is_mask=False)
    mask = rotate_and_expand(mask, angle, is_mask=True)
    image = apply_pixel_modifications(image)
    return image, mask


def convert_mask_colors_to_index(mask_bgr, palette_bgr):
    """
    Convert a BGR color-coded mask to integer class indices.
    palette_bgr: dict name->(B,G,R)
    Returns a single-channel uint8 or uint16 index mask.
    """
    h, w = mask_bgr.shape[:2]
    out = np.zeros((h, w), dtype=np.uint8)
    # Build reverse lookup: color tuple -> index
    color_to_idx = {tuple(v): i for i, v in enumerate(palette_bgr.values())}
    # If mask has 3 channels
    if mask_bgr.ndim == 3:
        for color, idx in color_to_idx.items():
            # create boolean mask
            matches = np.all(mask_bgr == color, axis=2)
            out[matches] = idx
    else:
        # single-channel mask - no conversion
        out = mask_bgr.copy().astype(np.uint8)
    return out


def run_augmentation():
    train_in = os.path.join(INPUT_ROOT, "train")
    test_in = os.path.join(INPUT_ROOT, "test")
    train_out = os.path.join(OUTPUT_ROOT, "train")
    test_out = os.path.join(OUTPUT_ROOT, "test")

    # Copy test data exactly as is
    if os.path.exists(test_in):
        if os.path.exists(test_out):
            shutil.rmtree(test_out)
        shutil.copytree(test_in, test_out)
        print(f"Test data copied to {test_out}")

    if not os.path.exists(train_in):
        print("Train directory not found!"); return

    sample_folders = sorted([f for f in os.listdir(train_in) if os.path.isdir(os.path.join(train_in, f))])

    new_folder_id = 0
    missing_count = 0

    for folder in tqdm(sample_folders, desc="Augmenting Train Sets"):
        folder_path = os.path.join(train_in, folder)
        img_path, mask_path = find_image_and_mask(folder_path)

        if img_path is None or mask_path is None:
            missing_count += 1
            print(f"  [WARN] Skipping '{folder_path}' — couldn't find image or mask (expected supported extensions {SUPPORTED_EXT}, mask containing '_gt_').")
            continue

        try:
            img_ptr = read_image_keep_alpha(img_path)
            mask_ptr = read_image_keep_alpha(mask_path)
        except Exception as e:
            print(f"  [ERROR] Reading files in {folder_path}: {e}")
            continue

        # Create output root for this sample (original)
        current_out_path = os.path.join(train_out, str(new_folder_id))
        os.makedirs(current_out_path, exist_ok=True)

        # Save original files (preserve original file names)
        base_img_name = os.path.basename(img_path)
        base_mask_name = os.path.basename(mask_path)

        cv2.imwrite(os.path.join(current_out_path, base_img_name), img_ptr)
        cv2.imwrite(os.path.join(current_out_path, base_mask_name), mask_ptr)
        new_folder_id += 1

        # Augmented versions
        for _ in range(AUGMENTATIONS_PER_IMAGE):
            aug_img, aug_mask = apply_random_geometry(img_ptr.copy(), mask_ptr.copy())

            aug_out_path = os.path.join(train_out, str(new_folder_id))
            os.makedirs(aug_out_path, exist_ok=True)

            cv2.imwrite(os.path.join(aug_out_path, base_img_name), aug_img)
            cv2.imwrite(os.path.join(aug_out_path, base_mask_name), aug_mask)
            new_folder_id += 1

    print(f"\nAugmentation complete. Total folders created: {new_folder_id}. Skipped folders: {missing_count}")

if __name__ == "__main__":
    run_augmentation()
