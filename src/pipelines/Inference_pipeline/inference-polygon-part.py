# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.patches as mpatches
# from PIL import Image
# from torchvision import transforms
# import pickle
# import os
# import cv2

# # Your model loader
# from src.components.model_mod_3 import get_model

# # ---------------------------------------------------
# # Utilities
# # ---------------------------------------------------

# def load_color_to_id(pickle_path):
#     """Load RGB -> class_id mapping"""
#     if not os.path.exists(pickle_path):
#         raise FileNotFoundError(pickle_path)

#     with open(pickle_path, "rb") as f:
#         color_to_class = pickle.load(f)

#     id_to_color = {i: k for i, k in enumerate(color_to_class.keys())}
#     return {tuple(color): idx for idx, color in enumerate(color_to_class.keys())}, id_to_color

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

# # ---------------------------------------------------
# # Refined Polygon Extraction
# # ---------------------------------------------------

# def refine_and_extract_polygons(pred_mask, upscale_factor=4, min_area=20, epsilon_coeff=0.002):
#     """
#     Refines the mask by upscaling and smoothing, then extracts polygons.
    
#     Args:
#         pred_mask (np.array): 224x224 Class ID mask.
#         upscale_factor (int): Multiplier for resolution (4x = 896x896). 
#                               Higher = smoother lines for small objects.
#         min_area (int): Minimum area (in upscaled pixels) to keep. 
#                         Lower this to keep smaller objects.
#         epsilon_coeff (float): Approximation strictness. 
#                                Lower (e.g., 0.002) = keeps more detail/corners.
#                                Higher (e.g., 0.01) = simplifies to rectangles.
#     """
#     polygons = []
#     unique_classes = np.unique(pred_mask)
    
#     # 1. Upscale the mask immediately to get higher precision for small features
#     h, w = pred_mask.shape
#     new_h, new_w = h * upscale_factor, w * upscale_factor
    
#     # Use INTER_NEAREST to keep class IDs integers (don't interpolate classes!)
#     upscaled_mask = cv2.resize(pred_mask.astype(np.uint8), (new_w, new_h), interpolation=cv2.INTER_NEAREST)

#     for cls_id in unique_classes:
#         if cls_id == 0: continue # Skip background

#         # Create binary mask for this class
#         class_mask = np.uint8(upscaled_mask == cls_id) * 255
        
#         # ---------------------------------------------------------
#         # A. Mask Refinement (Morphological Ops)
#         # ---------------------------------------------------------
#         # Kernel size depends on upscale factor. 
#         # For 896px image, a 3x3 or 5x5 kernel is appropriate to smooth jagged edges 
#         # without deleting small objects.
#         kernel = np.ones((3, 3), np.uint8)
        
#         # Closing: Connects small gaps (e.g. broken walls)
#         refined = cv2.morphologyEx(class_mask, cv2.MORPH_CLOSE, kernel)
        
#         # Opening: Removes isolated pixel noise. 
#         # WARNING: If your 'small polygons' are tiny dots, Opening might remove them.
#         # We use a very gentle opening here.
#         refined = cv2.morphologyEx(refined, cv2.MORPH_OPEN, kernel)

#         # ---------------------------------------------------------
#         # B. Contour Extraction
#         # ---------------------------------------------------------
#         contours, _ = cv2.findContours(refined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#         for cnt in contours:
#             # Filter by area (adjusting for the upscale factor)
#             # If we scaled up by 4, area scales by 16.
#             if cv2.contourArea(cnt) < (min_area * (upscale_factor**2)):
#                 continue

#             # ---------------------------------------------------------
#             # C. Polygon Approximation
#             # ---------------------------------------------------------
#             peri = cv2.arcLength(cnt, True)
#             epsilon = epsilon_coeff * peri
#             approx = cv2.approxPolyDP(cnt, epsilon, True)

#             if len(approx) >= 3:
#                 # D. Rescale points back to original 224x224 space
#                 # (Or keep them high-res if you prefer, but here we normalize to input)
#                 points_upscaled = approx.reshape(-1, 2)
#                 points_original = points_upscaled / upscale_factor
                
#                 polygons.append({
#                     "class_id": cls_id,
#                     "points": points_original
#                 })
                
#     return polygons, upscaled_mask


# # ---------------------------------------------------
# # Inference Function
# # ---------------------------------------------------

# def run_inference(image_path, gt_mask_path, model_ckpt_path, mapping_pickle_path):
#     device = "cuda" if torch.cuda.is_available() else "cpu"

#     # Load resources
#     color_to_id, id_to_color = load_color_to_id(mapping_pickle_path)
#     num_classes = len(color_to_id)
    
#     model = get_model(image_channel=3, number_of_class=num_classes)
#     checkpoint = torch.load(model_ckpt_path, map_location=device)
#     if isinstance(checkpoint, dict) and "model" in checkpoint:
#         model.load_state_dict(checkpoint["model"])
#     else:
#         model.load_state_dict(checkpoint)
#     model.to(device).eval()

#     # Preprocess
#     original_img_pil, input_tensor = preprocess_image(image_path, device)
#     display_img = original_img_pil.resize((224, 224))
#     gt_mask = preprocess_gt_mask(gt_mask_path, color_to_id)

#     # 1. Inference
#     print("[INFO] Running Inference...")
#     with torch.no_grad():
#         logits = model(input_tensor)
#         pred_mask = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()

#     # 2. Refine & Extract
#     print("[INFO] Refining Mask & Extracting Polygons...")
#     # NOTE: min_area is low (10) to ensure small polygons appear
#     polygons, refined_mask_vis = refine_and_extract_polygons(
#         pred_mask, 
#         upscale_factor=4, 
#         min_area=10, 
#         epsilon_coeff=0.003
#     )
    
#     print(f"[INFO] Found {len(polygons)} polygon shapes.")

#     # ---------------------------------------------------
#     # Visualization
#     # ---------------------------------------------------
#     fig, ax = plt.subplots(1, 4, figsize=(24, 6))

#     # Input
#     ax[0].imshow(display_img)
#     ax[0].set_title("Input Image")
#     ax[0].axis("off")

#     # Raw Prediction
#     ax[1].imshow(pred_mask, cmap="tab20")
#     ax[1].set_title("Raw Model Prediction")
#     ax[1].axis("off")

#     # Refined Mask (Visualizing the upscaled version)
#     ax[2].imshow(refined_mask_vis, cmap="tab20")
#     ax[2].set_title(f"Refined Mask (Upscaled)")
#     ax[2].axis("off")

#     # Polygons
#     ax[3].imshow(display_img)
#     ax[3].set_title(f"Final Polygons (Count: {len(polygons)})")
#     ax[3].axis("off")

#     # Draw Polygons
#     for poly in polygons:
#         points = poly["points"]
#         cls_id = poly["class_id"]
        
#         cmap = plt.get_cmap("tab20")
#         color = cmap(cls_id / num_classes)

#         # Polygon Fill
#         patch = mpatches.Polygon(
#             points, 
#             closed=True, 
#             edgecolor='white',  
#             facecolor=color,    
#             alpha=0.7,          
#             linewidth=1.5
#         )
#         ax[3].add_patch(patch)
        
#         # Vertices (dots) to check corner quality
#         ax[3].scatter(points[:, 0], points[:, 1], c='red', s=5, zorder=10)

#     plt.tight_layout()
#     plt.show()

#     return polygons

# # ---------------------------------------------------
# # Entry Point
# # ---------------------------------------------------
# if __name__ == "__main__":
#     CONFIG = {
#         "image": r"E:\floor-plan-segmentation-and-reconstruction\artifacts\processed-data\train\55\IIa_AP3401.png",
#         "mask":  r"E:\floor-plan-segmentation-and-reconstruction\artifacts\processed-data\train\55\IIa_AP3401_gt_5.png",
#         "model": "E:/floor-plan-segmentation-and-reconstruction/checkpoints/cross_entropy_best.pt",
#         "pickle": "./artifacts/processed-data/color_to_class.pkl"
#     }

#     polys = run_inference(
#         CONFIG["image"],
#         CONFIG["mask"],
#         CONFIG["model"],
#         CONFIG["pickle"]
#     )


import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
from torchvision import transforms
import pickle
import os
import cv2

# Your model loader
# Make sure this path is correct relative to where you run the script
try:
    from src.components.model_mod_3 import get_model
except ImportError:
    # Fallback for standalone testing if structure is different
    # Replace with your actual model definition if needed
    print("[WARNING] Could not import get_model. Define it here if running standalone.")
    pass


# ---------------------------------------------------
# Utilities
# ---------------------------------------------------

def load_color_to_id(pickle_path):
    """Load RGB -> class_id mapping"""
    if not os.path.exists(pickle_path):
        raise FileNotFoundError(pickle_path)

    with open(pickle_path, "rb") as f:
        color_to_class = pickle.load(f)

    id_to_color = {i: k for i, k in enumerate(color_to_class.keys())}
    return {tuple(color): idx for idx, color in enumerate(color_to_class.keys())}, id_to_color

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
    """Convert RGB GT mask → class-id mask"""
    mask = Image.open(mask_path).convert("RGB")
    mask = mask.resize((224, 224), Image.NEAREST)
    mask_np = np.array(mask)

    h, w, _ = mask_np.shape
    gt_id_mask = np.zeros((h, w), dtype=np.int64)

    for color, idx in color_to_id.items():
        match = np.all(mask_np == np.array(color), axis=-1)
        gt_id_mask[match] = idx

    return gt_id_mask

# ---------------------------------------------------
# Refined Polygon Extraction
# ---------------------------------------------------

def refine_and_extract_polygons(pred_mask, upscale_factor=4, min_area=20, epsilon_coeff=0.002):
    """
    Refines the mask by upscaling and smoothing, then extracts polygons.
    """
    polygons = []
    unique_classes = np.unique(pred_mask)
    
    # 1. Upscale
    h, w = pred_mask.shape
    new_h, new_w = h * upscale_factor, w * upscale_factor
    upscaled_mask = cv2.resize(pred_mask.astype(np.uint8), (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    # Placeholder for visualizing the refined mask
    refined_mask_vis = np.zeros_like(upscaled_mask)

    for cls_id in unique_classes:
        if cls_id == 0: continue # Skip background

        # Create binary mask for this class
        class_mask = np.uint8(upscaled_mask == cls_id) * 255
        
        # ---------------------------------------------------------
        # A. Mask Refinement (Morphological Ops)
        # ---------------------------------------------------------
        kernel = np.ones((3, 3), np.uint8)
        refined = cv2.morphologyEx(class_mask, cv2.MORPH_CLOSE, kernel)
        refined = cv2.morphologyEx(refined, cv2.MORPH_OPEN, kernel)

        # Add to visualization mask
        refined_mask_vis[refined > 0] = cls_id

        # ---------------------------------------------------------
        # B. Contour Extraction
        # ---------------------------------------------------------
        contours, _ = cv2.findContours(refined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            # Filter by area (adjusting for the upscale factor)
            if cv2.contourArea(cnt) < (min_area * (upscale_factor**2)):
                continue

            # ---------------------------------------------------------
            # C. Polygon Approximation
            # ---------------------------------------------------------
            peri = cv2.arcLength(cnt, True)
            epsilon = epsilon_coeff * peri
            approx = cv2.approxPolyDP(cnt, epsilon, True)

            if len(approx) >= 3:
                # D. Rescale points back to original 224x224 space
                points_upscaled = approx.reshape(-1, 2)
                points_original = points_upscaled / upscale_factor
                
                polygons.append({
                    "class_id": cls_id,
                    "points": points_original
                })
                
    return polygons, refined_mask_vis


# ---------------------------------------------------
# Inference Function
# ---------------------------------------------------

def run_inference(image_path, gt_mask_path, model_ckpt_path, mapping_pickle_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load resources
    color_to_id, id_to_color = load_color_to_id(mapping_pickle_path)
    num_classes = len(color_to_id)
    
    model = get_model(image_channel=3, number_of_class=num_classes)
    checkpoint = torch.load(model_ckpt_path, map_location=device)
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    else:
        model.load_state_dict(checkpoint)
    model.to(device).eval()

    # Preprocess
    original_img_pil, input_tensor = preprocess_image(image_path, device)
    display_img = original_img_pil.resize((224, 224))
    gt_mask = preprocess_gt_mask(gt_mask_path, color_to_id)

    # 1. Inference
    print("[INFO] Running Inference...")
    with torch.no_grad():
        logits = model(input_tensor)
        pred_mask = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()

    # 2. Refine & Extract
    print("[INFO] Refining Mask & Extracting Polygons...")
    polygons, refined_mask_vis = refine_and_extract_polygons(
        pred_mask, 
        upscale_factor=4, 
        min_area=10, 
        epsilon_coeff=0.003
    )
    
    print(f"[INFO] Found {len(polygons)} polygon shapes.")

    # ---------------------------------------------------
    # Visualization
    # ---------------------------------------------------
    fig, ax = plt.subplots(1, 4, figsize=(24, 6))

    # Input
    ax[0].imshow(display_img)
    ax[0].set_title("Input Image")
    ax[0].axis("off")

    # Raw Prediction
    ax[1].imshow(pred_mask, cmap="tab20")
    ax[1].set_title("Raw Model Prediction")
    ax[1].axis("off")

    # Refined Mask
    ax[2].imshow(refined_mask_vis, cmap="tab20")
    ax[2].set_title(f"Refined Mask (Upscaled Process)")
    ax[2].axis("off")

    # Polygons
    ax[3].imshow(display_img)
    ax[3].set_title(f"Final Polygons (Count: {len(polygons)})")
    ax[3].axis("off")

    # Draw Polygons in YELLOW
    for poly in polygons:
        points = poly["points"]
        # cls_id = poly["class_id"] # Not used for coloring anymore

        # Polygon Fill
        patch = mpatches.Polygon(
            points, 
            closed=True, 
            edgecolor='red',    # Red outline for contrast
            facecolor='yellow', # <--- ALL POLYGONS YELLOW
            alpha=0.7,          
            linewidth=2
        )
        ax[3].add_patch(patch)
        
        # Vertices (dots) in black for contrast against yellow fill
        ax[3].scatter(points[:, 0], points[:, 1], c='black', s=10, zorder=10)

    plt.tight_layout()
    plt.show()

    return polygons

# ---------------------------------------------------
# Entry Point
# ---------------------------------------------------
if __name__ == "__main__":
    # Ensure paths are correct for your environment
    CONFIG = {
        "image": r"E:\floor-plan-segmentation-and-reconstruction\artifacts\processed-data\test\16\Ic_CA0501_sommaire.png",
        "mask":  r"E:\floor-plan-segmentation-and-reconstruction\artifacts\processed-data\train\55\IIa_AP3401_gt_5.png",
        "model": "E:/floor-plan-segmentation-and-reconstruction/checkpoints/cross_entropy_best.pt",
        "pickle": "./artifacts/processed-data/color_to_class.pkl"
    }

    polys = run_inference(
        CONFIG["image"],
        CONFIG["mask"],
        CONFIG["model"],
        CONFIG["pickle"]
    )