import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import pickle
import os

# --- Import your legacy model loader ---
from src.components.legacy_models.model_mod_3 import get_model

# ==============================================================================
# 1. MODEL ARCHITECTURE (Must match training script exactly)
# ==============================================================================

class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        super(ConvLSTMCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.bias = bias
        self.conv = nn.Conv2d(
            in_channels=self.input_dim + self.hidden_dim,
            out_channels=4 * self.hidden_dim,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias
        )

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        device = self.conv.weight.device
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=device))

class RecurrentRefiner(nn.Module):
    def __init__(self, num_classes, hidden_dim=64, iterations=3):
        super(RecurrentRefiner, self).__init__()
        self.iterations = iterations
        self.num_classes = num_classes
        input_channels = 3 + num_classes
        self.conv_lstm = ConvLSTMCell(input_channels, hidden_dim, 3, True)
        self.out_conv = nn.Conv2d(hidden_dim, num_classes, kernel_size=1)

    def forward(self, image, coarse_logits):
        batch_size, _, h, w = image.shape
        hidden_state = self.conv_lstm.init_hidden(batch_size, (h, w))
        current_logits = coarse_logits
        outputs = []

        for _ in range(self.iterations):
            current_probs = torch.softmax(current_logits, dim=1)
            lstm_input = torch.cat([image, current_probs], dim=1)
            h, c = self.conv_lstm(lstm_input, hidden_state)
            hidden_state = (h, c)
            delta = self.out_conv(h)
            current_logits = current_logits + delta
            outputs.append(current_logits)
        return outputs

# ==============================================================================
# 2. UTILITIES
# ==============================================================================

def load_color_map(pickle_path):
    with open(pickle_path, "rb") as f:
        color_to_class = pickle.load(f)
    # Create ID -> RGB mapping
    id_to_color = {idx: color for idx, color in enumerate(color_to_class.keys())}
    return id_to_color, len(color_to_class)

def decode_segmap(mask_idx, id_to_color):
    """Converts a class-ID mask back to an RGB image."""
    h, w = mask_idx.shape
    rgb_image = np.zeros((h, w, 3), dtype=np.uint8)
    for idx, color in id_to_color.items():
        rgb_image[mask_idx == idx] = color
    return rgb_image

def preprocess_image(image_path, device):
    img = Image.open(image_path).convert("RGB")
    original_img = img.resize((224, 224)) # Keep for visualization
    
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    tensor = tf(img).unsqueeze(0).to(device)
    return original_img, tensor

# ==============================================================================
# 3. INFERENCE FUNCTION
# ==============================================================================

def run_inference(image_path, base_model_path, refiner_model_path, pickle_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")

    # 1. Load Mappings
    id_to_color, num_classes = load_color_map(pickle_path)

    # 2. Load Models
    print("[INFO] Loading Base Model...")
    base_model = get_model(image_channel=3, number_of_class=num_classes)
    checkpoint = torch.load(base_model_path, map_location=device)
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
    base_model.load_state_dict(state_dict)
    base_model.to(device).eval()

    print("[INFO] Loading Refiner Model...")
    refiner = RecurrentRefiner(num_classes=num_classes, iterations=3)
    refiner.load_state_dict(torch.load(refiner_model_path, map_location=device))
    refiner.to(device).eval()

    # 3. Process Image
    original_pil, input_tensor = preprocess_image(image_path, device)

    # 4. Predict
    with torch.no_grad():
        # Step A: Get Coarse Mask from Base Model
        coarse_logits = base_model(input_tensor)
        
        # Step B: Get Refined Mask from LSTM (We take the last iteration output)
        refined_outputs = refiner(input_tensor, coarse_logits)
        final_refined_logits = refined_outputs[-1] 

        # Step C: Convert to Class IDs
        coarse_mask_id = torch.argmax(coarse_logits, dim=1).squeeze().cpu().numpy()
        refined_mask_id = torch.argmax(final_refined_logits, dim=1).squeeze().cpu().numpy()

    # 5. Decode to RGB for Visualization
    coarse_rgb = decode_segmap(coarse_mask_id, id_to_color)
    refined_rgb = decode_segmap(refined_mask_id, id_to_color)

    # 6. Visualize
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    axes[0].imshow(original_pil)
    axes[0].set_title("Input Image")
    axes[0].axis("off")

    axes[1].imshow(coarse_rgb)
    axes[1].set_title("Base Model (Coarse)")
    axes[1].axis("off")

    axes[2].imshow(refined_rgb)
    axes[2].set_title("LSTM Refined Mask")
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()

# ==============================================================================
# 4. RUNNER
# ==============================================================================

if __name__ == "__main__":
    # --- UPDATE THESE PATHS ---
    CONFIG = {
        "image": r"E:\floor-plan-segmentation-and-reconstruction\artifacts\augmented\test\4\Ib_RI0501_sommaire.png",
        
        "base_model": r"E:\floor-plan-segmentation-and-reconstruction\training_results\trial-2\Best.pt",
        
        "refiner_model": r"./training_results/refiner/refiner_best.pt",
        
        "pickle": r"./artifacts/processed-data/color_to_class.pkl"
    }

    # Run
    if os.path.exists(CONFIG["image"]):
        run_inference(
            CONFIG["image"], 
            CONFIG["base_model"], 
            CONFIG["refiner_model"], 
            CONFIG["pickle"]
        )
    else:
        print(f"[ERROR] Image not found at: {CONFIG['image']}")