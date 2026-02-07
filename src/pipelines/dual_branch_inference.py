import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
from torchvision import transforms
import pickle
import os
import cv2
import trimesh
from shapely.geometry import Polygon as ShapelyPoly

# ---------------------------------------------------
# Configuration & Imports
# ---------------------------------------------------

# Try importing your NEW model
try:
    from src.components.dev_models.Novel_v1.model import get_model
except ImportError:
    print("[WARNING] Could not import get_model. Ensure you are in the correct directory.")

# ---------------------------------------------------
# Utilities
# ---------------------------------------------------

def load_color_to_id(pickle_path):
    """Load RGB -> class_id mapping"""
    if not os.path.exists(pickle_path):
        raise FileNotFoundError(f"Pickle not found: {pickle_path}") 

    with open(pickle_path, "rb") as f:
        color_to_class = pickle.load(f)

    # Create ID -> Color map for visualization later
    id_to_color = {i: k for i, k in enumerate(color_to_class.keys())}
    
    # Create Color -> ID map
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

# ---------------------------------------------------
# Refined Polygon Extraction
# ---------------------------------------------------

def refine_and_extract_polygons(pred_mask, upscale_factor=4, min_area=20, epsilon_coeff=0.002):
    polygons = []
    unique_classes = np.unique(pred_mask)
    
    h, w = pred_mask.shape
    new_h, new_w = h * upscale_factor, w * upscale_factor
    upscaled_mask = cv2.resize(pred_mask.astype(np.uint8), (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    refined_mask_vis = np.zeros_like(upscaled_mask)

    for cls_id in unique_classes:
        if cls_id == 0: continue # Skip background

        class_mask = np.uint8(upscaled_mask == cls_id) * 255
        
        # Morphological refinement
        kernel = np.ones((3, 3), np.uint8)
        refined = cv2.morphologyEx(class_mask, cv2.MORPH_CLOSE, kernel)
        refined = cv2.morphologyEx(refined, cv2.MORPH_OPEN, kernel)

        refined_mask_vis[refined > 0] = cls_id

        contours, _ = cv2.findContours(refined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            if cv2.contourArea(cnt) < (min_area * (upscale_factor**2)):
                continue

            peri = cv2.arcLength(cnt, True)
            epsilon = epsilon_coeff * peri
            approx = cv2.approxPolyDP(cnt, epsilon, True)

            if len(approx) >= 3:
                points_upscaled = approx.reshape(-1, 2)
                points_original = points_upscaled / upscale_factor
                
                polygons.append({
                    "class_id": cls_id,
                    "points": points_original
                })
                
    return polygons, refined_mask_vis

# ---------------------------------------------------
# 3D Reconstruction Logic
# ---------------------------------------------------

def generate_3d_scene(polygons, output_filename="output_floorplan.obj", extrusion_heights=None):
    print(f"[INFO] Generating 3D Scene...")
    
    meshes = []
    
    if extrusion_heights is None:
        extrusion_heights = {
            1: 40.0, 
            2: 15.0, 
            3: 10.0, 
            4: 5.0   
        }

    for poly in polygons:
        cls_id = poly["class_id"]
        points = poly["points"]

        # 1. Coordinate System Adjustment (Flip Y)
        points_3d = points.copy()
        points_3d[:, 1] = -points_3d[:, 1] 

        # 2. Create Shapely Polygon (Validation)
        try:
            shapely_poly = ShapelyPoly(points_3d)
            if not shapely_poly.is_valid:
                shapely_poly = shapely_poly.buffer(0) 
        except Exception as e:
            print(f"[WARN] Skipping invalid polygon: {e}")
            continue

        # 3. Determine Height
        height = extrusion_heights.get(cls_id, 10.0) 

        # 4. Extrude
        try:
            mesh = trimesh.creation.extrude_polygon(shapely_poly, height)
            # Random color for visualization
            mesh.visual.face_colors = trimesh.visual.random_color()
            meshes.append(mesh)
        except Exception as e:
             print(f"[WARN] Failed to extrude polygon class {cls_id}: {e}")

    if not meshes:
        print("[WARNING] No meshes created.")
        return

    # 5. Combine and Export
    scene = trimesh.Scene(meshes)
    scene.export(output_filename)
    print(f"[SUCCESS] 3D model saved to: {output_filename}")


# ---------------------------------------------------
# Inference Pipeline (Updated for Dual Branch)
# ---------------------------------------------------

def run_pipeline(config):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Setup Resources
    color_to_id, id_to_color = load_color_to_id(config["pickle"])
    num_classes = len(color_to_id)
    
    # Define Heights (Example: Class 0 is background, 1 is Wall)
    HEIGHT_MAP = {i: 20.0 for i in range(num_classes)}
    HEIGHT_MAP[1] = 50.0 

    # 2. Load Model
    model = get_model(image_channel=3, number_of_class=num_classes)
    
    # Load checkpoint safely
    try:
        checkpoint = torch.load(config["model"], map_location=device)
        state_dict = checkpoint["model"] if (isinstance(checkpoint, dict) and "model" in checkpoint) else checkpoint
        model.load_state_dict(state_dict)
    except Exception as e:
        print(f"[ERROR] Failed to load weights: {e}")
        return

    model.to(device).eval()

    # 3. Preprocess
    print(f"[INFO] Processing: {config['image']}")
    original_img_pil, input_tensor = preprocess_image(config['image'], device)

    # 4. Inference (Dual Branch)
    with torch.no_grad():
        # Get both outputs
        seg_logits, edge_logits = model(input_tensor)
        
        # Process Semantic Segmentation
        seg_prob = torch.softmax(seg_logits, dim=1)
        seg_mask = torch.argmax(seg_prob, dim=1).squeeze(0).cpu().numpy()
        
        # Process Edge Detection
        edge_prob = torch.sigmoid(edge_logits).squeeze(0).squeeze(0).cpu().numpy()
        # Binary edge mask (threshold 0.5)
        edge_mask = (edge_prob > 0.5).astype(np.float32)

    # 5. FUSION LOGIC
    # We use the edge mask to refine the segmentation.
    # Areas where edge is high are likely boundaries.
    # Simple strategy: If edge is detected, force that pixel to background (0) or separate objects.
    # Here, we will just use the edge mask to visualize for now, 
    # OR we can subtract edges from the mask to create gaps between touching objects.
    
    print("[INFO] Fusing Segmentation and Edge masks...")
    
    # Strategy: Mask out pixels that are strong edges to separate touching walls
    # (Assuming 0 is background)
    fused_mask = seg_mask.copy()
    fused_mask[edge_mask == 1] = 0 

    # 6. Polygon Extraction (Using Fused Mask)
    polygons, refined_vis = refine_and_extract_polygons(
        fused_mask, 
        upscale_factor=4, 
        min_area=10, 
        epsilon_coeff=0.003
    )
    print(f"[INFO] Extracted {len(polygons)} polygons.")

    # 7. Visualization (2D)
    # Updated to show Edge Branch output
    fig, ax = plt.subplots(1, 4, figsize=(24, 6))
    
    ax[0].imshow(original_img_pil.resize((224, 224)))
    ax[0].set_title("Input Image")
    
    ax[1].imshow(seg_mask, cmap="tab20")
    ax[1].set_title("Raw Segmentation Output")

    ax[2].imshow(edge_mask, cmap="gray")
    ax[2].set_title("Edge Branch Output")
    
    # Overlay polygons on original image
    ax[3].imshow(original_img_pil.resize((224, 224)))
    ax[3].set_title("Fused Result & Polygons")
    for poly in polygons:
        p = mpatches.Polygon(poly["points"], fill=True, facecolor='yellow', edgecolor='red', alpha=0.5)
        ax[3].add_patch(p)
    
    plt.tight_layout()
    plt.show()

    # 8. 3D Generation
    output_obj_path = config["output_3d"]
    generate_3d_scene(polygons, output_obj_path, extrusion_heights=HEIGHT_MAP)

# ---------------------------------------------------
# Entry Point
# ---------------------------------------------------
if __name__ == "__main__":
    CONFIG = {
        "image": r"E:\floor-plan-segmentation-and-reconstruction\ttest_floorplan\png-transparent-floor-plan-paper-line-2d-floor-plan-angle-text-rectangle.png",
        "model": "E:/floor-plan-segmentation-and-reconstruction/checkpoints/best.pt",
        "pickle": "./artifacts/processed-data/color_to_class.pkl",
        "output_3d": "my_floorplan_3d.obj" 
    }

    run_pipeline(CONFIG)