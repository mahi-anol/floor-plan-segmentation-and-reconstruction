# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# from PIL import Image
# from torchvision import transforms
# from scipy.spatial import cKDTree

# # Import from your existing files
# from src.components.model_mod_3 import get_model
# from src.utils import load_pickle

# # Configuration for paths (Mirroring your DatasetConfig)
# COLOR_MAPPING_PATH = './artifacts/processed-data/color_to_class.pkl'

# def process_ground_truth(mask_path, color_to_class_mapping, color_to_id_mapping):
#     """
#     Replicates the exact logic from CVC_FP_dataset.load_image_and_mask
#     to convert an RGB mask into an ID-based mask.
#     """
#     mask = Image.open(mask_path).convert("RGB")
#     mask_np = np.array(mask)
    
#     unique_colors = list(color_to_class_mapping.keys())
#     numpyed_unique_color = np.array(unique_colors)
    
#     # KDTree for closest color matching
#     kd_search_tree = cKDTree(numpyed_unique_color)
#     _, points = kd_search_tree.query(mask_np)
#     fixed_mask = numpyed_unique_color[points]
    
#     mask_height, mask_width = fixed_mask.shape[:-1]
#     refined_mask = np.zeros((mask_height, mask_width), dtype=np.int64)
    
#     for color, id in color_to_id_mapping.items():
#         refined_mask[np.all(fixed_mask == color, axis=-1)] = int(id)
    
#     # Apply the same Resize transform as in data_pipeline.py
#     mask_tf = transforms.Compose([
#         transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.NEAREST),
#     ])
    
#     # Convert back to PIL to resize, then to numpy for plotting
#     final_gt = np.array(mask_tf(Image.fromarray(refined_mask.astype(np.uint8))))
#     return final_gt

# def run_inference(image_path, mask_path, model_path, num_classes=8):
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
#     # 1. Load Mappings
#     color_to_class_mapping = load_pickle(COLOR_MAPPING_PATH)
#     color_to_id_mapping = {color: id for id, color in enumerate(color_to_class_mapping.keys())}

#     # 2. Load Model
#     model = get_model(image_channel=3, number_of_class=num_classes)
#     checkpoint = torch.load(model_path, map_location=device)
#     if isinstance(checkpoint, dict) and "model" in checkpoint:
#         model.load_state_dict(checkpoint["model"])
#     else:
#         model.load_state_dict(checkpoint)
#     model.to(device).eval()

#     # 3. Preprocess Image for Model
#     img_tf = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#     ])
#     original_img = Image.open(image_path).convert("RGB")
#     input_tensor = img_tf(original_img).unsqueeze(0).to(device)

#     # 4. Process Ground Truth
#     gt_mask_processed = process_ground_truth(mask_path, color_to_class_mapping, color_to_id_mapping)

#     # 5. Model Prediction
#     with torch.inference_mode():
#         logits = model(input_tensor)
#         prediction = torch.argmax(torch.softmax(logits, dim=1), dim=1)
#         prediction = prediction.squeeze(0).cpu().numpy()

#     # 6. Visualization
#     plt.figure(figsize=(18, 6))

#     # Subplot 1: Original Image
#     plt.subplot(1, 3, 1)
#     plt.imshow(original_img.resize((224, 224)))
#     plt.title("Original Image (Resized)")
#     plt.axis("off")

#     # Subplot 2: Ground Truth
#     plt.subplot(1, 3, 2)
#     plt.imshow(gt_mask_processed, cmap='viridis')
#     plt.title("Ground Truth Mask (Processed)")
#     plt.axis("off")

#     # Subplot 3: Prediction
#     plt.subplot(1, 3, 3)
#     plt.imshow(prediction, cmap='viridis')
#     plt.title("Predicted Mask")
#     plt.axis("off")

#     plt.tight_layout()
#     plt.show()

# if __name__ == "__main__":
#     # --- UPDATE THESE PATHS ---
#     IMAGE_FILE = r"E:/floor-plan-segmentation-and-reconstruction/artifacts/processed-data/train/1/image024.png"
#     MASK_FILE = r"E:/floor-plan-segmentation-and-reconstruction/artifacts/processed-data/train/1/image024_gt_2.png"
#     MODEL_FILE = "./checkpoints/cross_entropy_best.pt"
#     # ---------------------------

#     run_inference(IMAGE_FILE, MASK_FILE, MODEL_FILE)


import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from scipy.spatial import cKDTree
from src.components.model_mod_3 import get_model
from src.utils import load_pickle

COLOR_MAPPING_PATH = './artifacts/processed-data/color_to_class.pkl'

def process_ground_truth(mask_path, color_to_class_mapping, color_to_id_mapping, num_classes):
    mask = Image.open(mask_path).convert("RGB")
    mask_np = np.array(mask)
    
    unique_colors = list(color_to_class_mapping.keys())
    numpyed_unique_color = np.array(unique_colors)
    
    kd_search_tree = cKDTree(numpyed_unique_color)
    _, points = kd_search_tree.query(mask_np)
    fixed_mask = numpyed_unique_color[points]
    
    mask_height, mask_width = fixed_mask.shape[:-1]
    refined_mask = np.zeros((mask_height, mask_width), dtype=np.int64)
    
    for color, id in color_to_id_mapping.items():
        refined_mask[np.all(fixed_mask == color, axis=-1)] = int(id)

    # --- CHANGE HERE: COLLAPSE ONLY IF TRAINING BINARY ---
    if num_classes == 2:
        # Example: assume Wall is ID 3, collapse everything else to 0
        # You can also use the target_class_name logic from previous turn
        refined_mask = np.where(refined_mask == 3, 1, 0) 
    
    mask_tf = transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.NEAREST),
    ])
    
    final_gt = np.array(mask_tf(Image.fromarray(refined_mask.astype(np.uint8))))
    return final_gt

def run_inference(image_path, mask_path, model_path, num_classes=8):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    color_to_class_mapping = load_pickle(COLOR_MAPPING_PATH)
    color_to_id_mapping = {color: id for id, color in enumerate(color_to_class_mapping.keys())}

    model = get_model(image_channel=3, number_of_class=num_classes)
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    else:
        model.load_state_dict(checkpoint)
    model.to(device).eval()

    img_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    original_img = Image.open(image_path).convert("RGB")
    input_tensor = img_tf(original_img).unsqueeze(0).to(device)

    # Pass num_classes to decide if we collapse labels or show all
    gt_mask_processed = process_ground_truth(mask_path, color_to_class_mapping, color_to_id_mapping, num_classes)

    with torch.inference_mode():
        logits = model(input_tensor)
        prediction = torch.argmax(torch.softmax(logits, dim=1), dim=1)
        prediction = prediction.squeeze(0).cpu().numpy()

    # --- VISUALIZATION CHANGES ---
    plt.figure(figsize=(18, 6))

    plt.subplot(1, 3, 1)
    plt.imshow(original_img.resize((224, 224)))
    plt.title("Original Image")
    plt.axis("off")

    # Use 'tab10' or 'jet' for multiclass to see different colors for each ID
    # Use 'gray' only if num_classes == 2
    cmap_to_use = 'gray' if num_classes == 2 else 'tab10'

    plt.subplot(1, 3, 2)
    plt.imshow(gt_mask_processed, cmap=cmap_to_use, vmin=0, vmax=num_classes-1)
    plt.title(f"Ground Truth ({num_classes} classes)")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(prediction, cmap=cmap_to_use, vmin=0, vmax=num_classes-1)
    plt.title(f"Prediction ({num_classes} classes)")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    IMAGE_FILE = r"E:/floor-plan-segmentation-and-reconstruction/artifacts/processed-data/train/1/image024.png"
    MASK_FILE = r"E:/floor-plan-segmentation-and-reconstruction/artifacts/processed-data/train/1/image024_gt_2.png"
    MODEL_FILE = "./checkpoints/cross_entropy_best.pt"

    run_inference(IMAGE_FILE, MASK_FILE, MODEL_FILE, num_classes=2)