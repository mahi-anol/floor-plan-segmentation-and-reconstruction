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

# Try importing your model
try:
    from src.components.legacy_models.model_mod_3 import get_model
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

def preprocess_gt_mask(mask_path, color_to_id):
    """Convert RGB GT mask -> class-id mask"""
    if not os.path.exists(mask_path):
        return None 
        
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
# 3D Reconstruction Logic (New)
# ---------------------------------------------------

def generate_3d_scene(polygons, output_filename="output_floorplan.obj", extrusion_heights=None):
    """
    Converts 2D polygons into a 3D scene using Trimesh.
    
    Args:
        polygons: List of dicts with 'class_id' and 'points'.
        output_filename: Path to save the .obj file.
        extrusion_heights: Dict mapping class_id -> height (float).
    """
    print(f"[INFO] Generating 3D Scene...")
    
    meshes = []
    
    # Default heights if not provided (Adjust these based on your specific class IDs)
    # Example: Class 1 (Wall) -> 50 units, Class 2 (Window) -> 20 units
    if extrusion_heights is None:
        extrusion_heights = {
            1: 40.0, # Example: Wall
            2: 15.0, # Example: Window
            3: 10.0, # Example: Door
            4: 5.0   # Example: Object
        }

    for poly in polygons:
        cls_id = poly["class_id"]
        points = poly["points"]

        # 1. Coordinate System Adjustment
        # Images have (0,0) at top-left. 3D usually has Y up.
        # We flip Y so the floorplan doesn't look mirrored/upside-down in 3D tools.
        points_3d = points.copy()
        points_3d[:, 1] = -points_3d[:, 1] 

        # 2. Create Shapely Polygon (Validation)
        try:
            shapely_poly = ShapelyPoly(points_3d)
            if not shapely_poly.is_valid:
                shapely_poly = shapely_poly.buffer(0) # Attempt self-repair
        except Exception as e:
            print(f"[WARN] Skipping invalid polygon: {e}")
            continue

        # 3. Determine Height
        # Default to 10.0 if class not in config
        height = extrusion_heights.get(cls_id, 10.0) 

        # 4. Extrude
        # Trimesh makes this very easy
        mesh = trimesh.creation.extrude_polygon(shapely_poly, height)
        
        # 5. Styling
        # Assign a random color or specific color based on class
        # (Here we just use random for distinctness)
        mesh.visual.face_colors = trimesh.visual.random_color()
        
        meshes.append(mesh)

    if not meshes:
        print("[WARNING] No meshes created. Check polygon extraction.")
        return

    # 6. Combine all meshes into one scene
    scene = trimesh.Scene(meshes)
    
    # 7. Export
    # You can export as 'obj', 'glb', 'stl', etc.
    scene.export(output_filename)
    print(f"[SUCCESS] 3D model saved to: {output_filename}")
    
    return scene

# ---------------------------------------------------
# Inference Pipeline
# ---------------------------------------------------

def run_pipeline(config):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Setup Resources
    color_to_id, id_to_color = load_color_to_id(config["pickle"])
    num_classes = len(color_to_id)
    
    # Define Heights based on your specific classes (Update this map!)
    # Look at your id_to_color to know which ID is which object
    # For now, I'm setting heuristic values
    HEIGHT_MAP = {
        i: 20.0 for i in range(num_classes)
    }
    # Example: If you know ID 1 is Wall, make it taller
    HEIGHT_MAP[1] = 50.0 

    # 2. Load Model
    model = get_model(image_channel=3, number_of_class=num_classes)
    checkpoint = torch.load(config["model"], map_location=device)
    state_dict = checkpoint["model"] if (isinstance(checkpoint, dict) and "model" in checkpoint) else checkpoint
    model.load_state_dict(state_dict)
    model.to(device).eval()

    # 3. Preprocess
    print(f"[INFO] Processing: {config['image']}")
    original_img_pil, input_tensor = preprocess_image(config['image'], device)

    # 4. Inference
    with torch.no_grad():
        logits = model(input_tensor)
        pred_mask = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()

    # 5. Polygon Extraction
    polygons, refined_vis = refine_and_extract_polygons(
        pred_mask, 
        upscale_factor=4, 
        min_area=10, 
        epsilon_coeff=0.003
    )
    print(f"[INFO] Extracted {len(polygons)} polygons.")

    # 6. Visualization (2D)
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    ax[0].imshow(original_img_pil.resize((224, 224)))
    ax[0].set_title("Input")
    ax[1].imshow(refined_vis, cmap="tab20")
    ax[1].set_title("Refined Segmentation")
    
    # Visualizing Polygons
    ax[2].imshow(original_img_pil.resize((224, 224)))
    ax[2].set_title("Polygons")
    for poly in polygons:
        p = mpatches.Polygon(poly["points"], fill=True, facecolor='yellow', edgecolor='red', alpha=0.5)
        ax[2].add_patch(p)
    plt.show()

    # 7. 3D Generation (The new part)
    output_obj_path = config["output_3d"]
    generate_3d_scene(polygons, output_obj_path, extrusion_heights=HEIGHT_MAP)

# ---------------------------------------------------
# Entry Point
# ---------------------------------------------------
if __name__ == "__main__":
    # UPDATE THESE PATHS
    CONFIG = {
        "image": r"E:\floor-plan-segmentation-and-reconstruction\ttest_floorplan\png-transparent-floor-plan-paper-line-2d-floor-plan-angle-text-rectangle.png",
        "model": r"E:\floor-plan-segmentation-and-reconstruction\training_results\trial-2\Best.pt",
        "pickle": "./artifacts/processed-data/color_to_class.pkl",
        "output_3d": "my_floorplan_3d.obj" # Output file name
    }

    run_pipeline(CONFIG)