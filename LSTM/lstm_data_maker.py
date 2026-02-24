# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# from PIL import Image
# from torchvision import transforms
# import pickle
# import os

# # Your model loader
# from src.components.legacy_models.model_mod_3 import get_model


# # ---------------------------------------------------
# # Utilities
# # ---------------------------------------------------

# def load_color_to_id(pickle_path):
#     """Load RGB -> class_id mapping"""
#     if not os.path.exists(pickle_path):
#         raise FileNotFoundError(pickle_path)

#     with open(pickle_path, "rb") as f:
#         color_to_class = pickle.load(f)

#     return {tuple(color): idx for idx, color in enumerate(color_to_class.keys())}


# def preprocess_image(image_path, device):
#     """Prepare input image for model"""
#     tf = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor()
#     ])

#     img = Image.open(image_path).convert("RGB")
#     tensor = tf(img).unsqueeze(0).to(device)
#     return img, tensor


# def preprocess_gt_mask(mask_path, color_to_id):
#     """Convert RGB GT mask → class-id mask"""
#     mask = Image.open(mask_path).convert("RGB")
#     mask = mask.resize((224, 224), Image.NEAREST)
#     mask_np = np.array(mask)

#     h, w, _ = mask_np.shape
#     gt_id_mask = np.zeros((h, w), dtype=np.int64)

#     for color, idx in color_to_id.items():
#         match = np.all(mask_np == np.array(color), axis=-1)
#         gt_id_mask[match] = idx

#     return gt_id_mask


# def find_image_and_gt(folder_path):
#     """
#     Finds the input image and GT mask inside a folder.
#     GT image must contain 'gt' in filename.
#     """
#     image_path = None
#     gt_path = None

#     for file in os.listdir(folder_path):
#         file_lower = file.lower()

#         if not file_lower.endswith((".png", ".jpg", ".jpeg")):
#             continue

#         full_path = os.path.join(folder_path, file)

#         if "gt" in file_lower:
#             gt_path = full_path
#         else:
#             image_path = full_path

#     return image_path, gt_path


# # ---------------------------------------------------
# # Folder Inference
# # ---------------------------------------------------

# def run_folder_inference(
#     root_folder,
#     model_ckpt_path,
#     mapping_pickle_path
# ):
#     device = "cuda" if torch.cuda.is_available() else "cpu"

#     # Load class mapping
#     color_to_id = load_color_to_id(mapping_pickle_path)
#     num_classes = len(color_to_id)
#     print(f"[INFO] Number of classes: {num_classes}")

#     # Load model
#     model = get_model(image_channel=3, number_of_class=num_classes)
#     checkpoint = torch.load(model_ckpt_path, map_location=device)

#     if isinstance(checkpoint, dict) and "model" in checkpoint:
#         model.load_state_dict(checkpoint["model"])
#     else:
#         model.load_state_dict(checkpoint)

#     model.to(device).eval()

#     # Iterate through child folders
#     for child in os.listdir(root_folder):
#         child_path = os.path.join(root_folder, child)

#         if not os.path.isdir(child_path):
#             continue

#         image_path, gt_path = find_image_and_gt(child_path)

#         if image_path is None or gt_path is None:
#             print(f"[WARN] Skipping {child_path} (missing image or gt)")
#             continue

#         print(f"[INFO] Processing: {child}")

#         # Prepare inputs
#         original_img, input_tensor = preprocess_image(image_path, device)
#         gt_mask = preprocess_gt_mask(gt_path, color_to_id)

#         # Inference
#         with torch.no_grad():
#             logits = model(input_tensor)
#             pred_mask = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()

#         # ---------------------------------------------------
#         # Visualization
#         # ---------------------------------------------------
#         fig, ax = plt.subplots(1, 3, figsize=(18, 6))

#         ax[0].imshow(original_img.resize((224, 224)))
#         ax[0].set_title("Input Image")
#         ax[0].axis("off")

#         ax[1].imshow(gt_mask, cmap="tab20")
#         ax[1].set_title("Ground Truth Mask")
#         ax[1].axis("off")

#         ax[2].imshow(pred_mask, cmap="tab20")
#         ax[2].set_title("Predicted Mask")
#         ax[2].axis("off")

#         plt.tight_layout()
#         plt.show()


# # ---------------------------------------------------
# # Entry Point
# # ---------------------------------------------------

# if __name__ == "__main__":
#     CONFIG = {
#         "root_folder": r"E:\floor-plan-segmentation-and-reconstruction\artifacts\processed-data\train",
#         "model": r"E:\floor-plan-segmentation-and-reconstruction\training_results\trial-2\Best.pt",
#         "pickle": r"./artifacts/processed-data/color_to_class.pkl"
#     }

#     run_folder_inference(
#         CONFIG["root_folder"],
#         CONFIG["model"],
#         CONFIG["pickle"]
#     )

#### lstm data save
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import pickle
import os

# Your model loader
from src.components.legacy_models.model_mod_3 import get_model

# ---------------------------------------------------
# Utilities
# ---------------------------------------------------

def load_color_to_id(pickle_path):
    """Load RGB -> class_id mapping"""
    if not os.path.exists(pickle_path):
        raise FileNotFoundError(f"Pickle file not found at: {pickle_path}")

    with open(pickle_path, "rb") as f:
        color_to_class = pickle.load(f)

    return {tuple(color): idx for idx, color in enumerate(color_to_class.keys())}


def preprocess_image(image_path, device):
    """Prepare input image for model"""
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    img = Image.open(image_path).convert("RGB")
    tensor = tf(img).unsqueeze(0).to(device)
    return img, tensor


def preprocess_gt_mask(mask_path, color_to_id):
    """Convert RGB GT mask -> class-id mask"""
    mask = Image.open(mask_path).convert("RGB")
    mask = mask.resize((224, 224), Image.NEAREST)
    mask_np = np.array(mask)

    h, w, _ = mask_np.shape
    gt_id_mask = np.zeros((h, w), dtype=np.int64)

    for color, idx in color_to_id.items():
        match = np.all(mask_np == np.array(color), axis=-1)
        gt_id_mask[match] = idx

    return gt_id_mask


def decode_segmap(mask, id_to_color):
    """
    Convert a class-id mask (0, 1, 2...) back to an RGB image 
    using the original color mapping.
    """
    h, w = mask.shape
    rgb_image = np.zeros((h, w, 3), dtype=np.uint8)

    for idx, color in id_to_color.items():
        # idx is the class ID (integer)
        # color is the RGB tuple (r, g, b)
        rgb_image[mask == idx] = color

    return rgb_image


def find_image_and_gt(folder_path):
    image_path = None
    gt_path = None

    for file in os.listdir(folder_path):
        file_lower = file.lower()
        if not file_lower.endswith((".png", ".jpg", ".jpeg")):
            continue

        full_path = os.path.join(folder_path, file)
        if "gt" in file_lower:
            gt_path = full_path
        else:
            image_path = full_path

    return image_path, gt_path


# ---------------------------------------------------
# Folder Inference Logic
# ---------------------------------------------------

def run_folder_inference(
    root_folder,
    model_ckpt_path,
    mapping_pickle_path,
    results_root="inference_results"
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")

    # 1. Load class mapping
    color_to_id = load_color_to_id(mapping_pickle_path)
    # Create the reverse mapping: ID -> RGB
    id_to_color = {v: k for k, v in color_to_id.items()}
    
    num_classes = len(color_to_id)
    print(f"[INFO] Number of classes: {num_classes}")

    # 2. Load model
    model = get_model(image_channel=3, number_of_class=num_classes)
    checkpoint = torch.load(model_ckpt_path, map_location=device)

    if isinstance(checkpoint, dict) and "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    else:
        model.load_state_dict(checkpoint)

    model.to(device).eval()

    os.makedirs(results_root, exist_ok=True)

    # 3. Iterate through child folders
    for child in os.listdir(root_folder):
        child_path = os.path.join(root_folder, child)

        if not os.path.isdir(child_path):
            continue

        image_path, gt_path = find_image_and_gt(child_path)

        if image_path is None or gt_path is None:
            continue

        print(f"[INFO] Processing: {child}")

        save_dir = os.path.join(results_root, child)
        os.makedirs(save_dir, exist_ok=True)

        # Preprocess
        original_img, input_tensor = preprocess_image(image_path, device)
        gt_mask = preprocess_gt_mask(gt_path, color_to_id)

        # Inference
        with torch.no_grad():
            logits = model(input_tensor)
            pred_mask = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()

        # ---------------------------------------------------
        # Reconstruct RGB Masks
        # ---------------------------------------------------
        # Convert IDs back to RGB colors
        pred_rgb = decode_segmap(pred_mask, id_to_color)
        gt_rgb = decode_segmap(gt_mask, id_to_color)

        # ---------------------------------------------------
        # Saving Results
        # ---------------------------------------------------
        
        # 1. Save side-by-side comparison using the CORRECT colors
        fig, ax = plt.subplots(1, 3, figsize=(18, 6))

        ax[0].imshow(original_img.resize((224, 224)))
        ax[0].set_title("Input Image")
        ax[0].axis("off")

        ax[1].imshow(gt_rgb) # Now using the real RGB colors
        ax[1].set_title("Ground Truth (Reconstructed)")
        ax[1].axis("off")

        ax[2].imshow(pred_rgb) # Now using the real RGB colors
        ax[2].set_title("Predicted Mask")
        ax[2].axis("off")

        plt.tight_layout()
        # plt.savefig(os.path.join(save_dir, "comparison_plot.png"))
        plt.close(fig)

        # 2. Save the RGB masks as standalone images
        Image.fromarray(gt_rgb).save(os.path.join(save_dir, "gt_mask_color.png"))
        Image.fromarray(pred_rgb).save(os.path.join(save_dir, "pred_mask_color.png"))
        original_img.save(os.path.join(save_dir, "original.png"))

    print(f"\n[SUCCESS] Results saved to: {os.path.abspath(results_root)}")


if __name__ == "__main__":
    CONFIG = {
        "root_folder": r"E:\floor-plan-segmentation-and-reconstruction\artifacts\processed-data\train",
        "model": r"E:\floor-plan-segmentation-and-reconstruction\training_results\trial-2\Best.pt",
        "pickle": r"./artifacts/processed-data/color_to_class.pkl",
        "results_folder": r"./inference_results" 
    }

    run_folder_inference(
        CONFIG["root_folder"],
        CONFIG["model"],
        CONFIG["pickle"],
        CONFIG["results_folder"]
    )