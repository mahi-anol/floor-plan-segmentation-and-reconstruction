import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
from tqdm import tqdm

# --------------------------------------------------------
# IMPORTS FROM YOUR PIPELINE
# --------------------------------------------------------
# Ensure data_pipeline.py is in the same folder or python path
from src.pipelines.data_pipeline import get_train_test_loader, DatasetConfig
from src.components.legacy_models.model_mod_3 import get_model

# ==============================================================================
# 1. CONVLSTM CELL (The Memory Unit)
# ==============================================================================
class ConvLSTMCell(nn.Module):
    """
    A Convolutional LSTM Cell. 
    Unlike a standard LSTM, this preserves spatial dimensions (H, W).
    """
    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        super(ConvLSTMCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.bias = bias

        # The core convolution that handles Input + Hidden State
        self.conv = nn.Conv2d(
            in_channels=self.input_dim + self.hidden_dim,
            out_channels=4 * self.hidden_dim, # 4 gates (Input, Forget, Output, Cell)
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias
        )

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        
        # Concatenate Input (Image+Mask) with Previous Hidden State
        combined = torch.cat([input_tensor, h_cur], dim=1)
        
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)

        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        # Update Cell State and Hidden State
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        device = self.conv.weight.device
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=device))


# ==============================================================================
# 2. THE RECURRENT REFINER MODEL
# ==============================================================================
class RecurrentRefiner(nn.Module):
    """
    Architecture:
    Input: [Original Image RGB] + [Current Mask Probabilities]
    Output: Residual update to the Mask Logits
    """
    def __init__(self, num_classes, hidden_dim=64, iterations=3):
        super(RecurrentRefiner, self).__init__()
        self.iterations = iterations
        self.num_classes = num_classes
        
        # Input = 3 (RGB Image) + num_classes (Probability Map)
        input_channels = 3 + num_classes
        
        self.conv_lstm = ConvLSTMCell(
            input_dim=input_channels, 
            hidden_dim=hidden_dim, 
            kernel_size=3, 
            bias=True
        )
        
        # Maps the LSTM hidden state back to Class Logits
        self.out_conv = nn.Conv2d(hidden_dim, num_classes, kernel_size=1)

    def forward(self, image, coarse_logits):
        batch_size, _, h, w = image.shape
        hidden_state = self.conv_lstm.init_hidden(batch_size, (h, w))
        
        current_logits = coarse_logits
        outputs = []

        for _ in range(self.iterations):
            # 1. Convert logits to probabilities (Softmax)
            # This helps the network understand confidence levels
            current_probs = torch.softmax(current_logits, dim=1)
            
            # 2. Create Input: Stack Image and Probabilities
            lstm_input = torch.cat([image, current_probs], dim=1)
            
            # 3. LSTM Step
            h, c = self.conv_lstm(lstm_input, hidden_state)
            hidden_state = (h, c)
            
            # 4. Predict Residual (Delta)
            delta = self.out_conv(h)
            
            # 5. Update Logits
            current_logits = current_logits + delta
            outputs.append(current_logits)

        return outputs


# ==============================================================================
# 3. TRAINING LOOP
# ==============================================================================

def train_refiner(config):
    # Setup Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")

    # 1. Load Data (Using your specific pipeline)
    print("[INFO] Loading Dataset...")
    train_loader, test_loader = get_train_test_loader(batch_size=config["batch_size"])
    
    # Extract num_classes from the dataset instance inside the loader
    # (Accessing the underlying dataset object)
    num_classes = len(train_loader.dataset.color_to_id)
    print(f"[INFO] Detected {num_classes} classes.")

    # 2. Load FROZEN Base Model
    print("[INFO] Loading and Freezing Base Model...")
    base_model = get_model(image_channel=3, number_of_class=num_classes)
    checkpoint = torch.load(config["base_model_ckpt"], map_location=device)
    
    # Handle checkpoint dictionary vs state_dict
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        base_model.load_state_dict(checkpoint["model"])
    else:
        base_model.load_state_dict(checkpoint)
        
    base_model.to(device)
    base_model.eval()
    
    # IMPORTANT: Freeze base model so we don't ruin it
    for param in base_model.parameters():
        param.requires_grad = False

    # 3. Initialize Refiner
    print("[INFO] Initializing Recurrent Refiner...")
    refiner = RecurrentRefiner(num_classes=num_classes, iterations=config["lstm_iterations"])
    refiner.to(device)
    
    optimizer = optim.Adam(refiner.parameters(), lr=config["lr"])
    criterion = nn.CrossEntropyLoss()

    # 4. Training Loop
    print("[INFO] Starting Training...")
    
    for epoch in range(config["epochs"]):
        refiner.train()
        epoch_loss = 0
        
        # TQDM for progress bar
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']}")
        
        for batch_idx, (images, gt_masks) in enumerate(loop):
            images = images.to(device)
            gt_masks = gt_masks.to(device) # Shape: [B, H, W] with class IDs
            
            # A. Get Initial Coarse Mask (No Grad)
            with torch.no_grad():
                coarse_logits = base_model(images)
            
            # B. Refiner Forward Pass
            # Returns list of logits: [step1_logits, step2_logits, step3_logits]
            refined_outputs = refiner(images, coarse_logits)
            
            # C. Deep Supervision Loss
            # We calculate loss at EVERY step to force iterative improvement
            loss = 0
            for step_logits in refined_outputs:
                loss += criterion(step_logits, gt_masks)
            
            # Average loss over timesteps
            loss = loss / len(refined_outputs)

            # D. Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            loop.set_postfix(loss=loss.item())
        
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1} Completed. Avg Loss: {avg_loss:.4f}")
        
        # Save Checkpoint
        save_path = os.path.join(config["save_dir"], "refiner_best.pt")
        torch.save(refiner.state_dict(), save_path)

    print(f"[SUCCESS] Training finished. Model saved to {config['save_dir']}")


# ==============================================================================
# 4. CONFIGURATION
# ==============================================================================

if __name__ == "__main__":
    # Ensure save directory exists
    os.makedirs("./training_results/refiner", exist_ok=True)

    CONFIG = {
        # Path to the model you already trained
        "base_model_ckpt": r"E:\floor-plan-segmentation-and-reconstruction\training_results\trial-2\Best.pt",
        
        "save_dir": r"./training_results/refiner",
        
        # Training Hyperparameters
        "batch_size": 4,  # Keep low (4 or 8) because ConvLSTM uses a lot of VRAM
        "epochs": 20,
        "lr": 1e-4,
        "lstm_iterations": 3 # How many times the refiner polishes the mask
    }

    train_refiner(CONFIG)