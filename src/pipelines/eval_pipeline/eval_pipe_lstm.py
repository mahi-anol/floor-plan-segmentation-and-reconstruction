import torch
import torch.nn as nn
import pandas as pd
import os
import numpy as np
from dataclasses import dataclass
from src.pipelines.data_pipeline import get_train_test_loader
from src.components.legacy_models.model_mod_3 import get_model
from src.components.eval_metrices import get_eval_metrices_outcome

# ==============================================================================
# 1. ARCHITECTURE REPLICATION (Must match your inference script)
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
        i, f, o, g = torch.sigmoid(cc_i), torch.sigmoid(cc_f), torch.sigmoid(cc_o), torch.tanh(cc_g)
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
        self.conv_lstm = ConvLSTMCell(3 + num_classes, hidden_dim, 3, True)
        self.out_conv = nn.Conv2d(hidden_dim, num_classes, kernel_size=1)

    def forward(self, image, coarse_logits):
        batch_size, _, h, w = image.shape
        hidden_state = self.conv_lstm.init_hidden(batch_size, (h, w))
        current_logits = coarse_logits
        for _ in range(self.iterations):
            current_probs = torch.softmax(current_logits, dim=1)
            lstm_input = torch.cat([image, current_probs], dim=1)
            h, c = self.conv_lstm(lstm_input, hidden_state)
            hidden_state = (h, c)
            current_logits = current_logits + self.out_conv(h)
        return current_logits

# ==============================================================================
# 2. CONFIGURATION
# ==============================================================================

@dataclass
class Configs:
    # Update these paths to your best weights
    base_model_path = r"E:\floor-plan-segmentation-and-reconstruction\training_results\trial-2\Best.pt"
    refiner_model_path = r"./training_results/refiner/refiner_best.pt"
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_classes = 2  # Ensure this matches your pickle/dataset
    batch_size = 16

# ==============================================================================
# 3. EVALUATION ENGINE
# ==============================================================================

def load_models():
    # Load Base
    base_model = get_model(image_channel=3, number_of_class=Configs.num_classes)
    checkpoint = torch.load(Configs.base_model_path, map_location=Configs.device)
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
    base_model.load_state_dict(state_dict)
    base_model.to(Configs.device).eval()

    # Load Refiner
    refiner = RecurrentRefiner(num_classes=Configs.num_classes)
    refiner.load_state_dict(torch.load(Configs.refiner_model_path, map_location=Configs.device))
    refiner.to(Configs.device).eval()
    
    return base_model, refiner

def run_evaluation():
    print(f"[INFO] Initializing Evaluation on {Configs.device}...")
    _, test_loader = get_train_test_loader(batch_size=Configs.batch_size)
    base_model, refiner = load_models()

    # We will track metrics for both to see the improvement
    metrics_base = {}
    metrics_refined = {}

    with torch.no_grad():
        for i, (images, gt) in enumerate(test_loader):
            images, gt = images.to(Configs.device), gt.to(Configs.device)

            # --- Inference ---
            coarse_logits = base_model(images)
            refined_logits = refiner(images, coarse_logits)

            # --- Process Base Predictions ---
            pred_base = torch.argmax(coarse_logits, dim=1).cpu().numpy()
            # --- Process Refined Predictions ---
            pred_refined = torch.argmax(refined_logits, dim=1).cpu().numpy()
            
            gt_np = gt.cpu().numpy()

            # --- Calculate Metrics ---
            res_base = get_eval_metrices_outcome(gt_np, pred_base, num_class=Configs.num_classes)
            res_refined = get_eval_metrices_outcome(gt_np, pred_refined, num_class=Configs.num_classes)

            # Accumulate
            for k, v in res_base.items():
                metrics_base[k] = metrics_base.get(k, 0) + v
            for k, v in res_refined.items():
                metrics_refined[k] = metrics_refined.get(k, 0) + v
            
            if i % 5 == 0:
                print(f"Batch {i}/{len(test_loader)} processed...")

    # Average the metrics
    n_batches = len(test_loader)
    for k in metrics_base: metrics_base[k] /= n_batches
    for k in metrics_refined: metrics_refined[k] /= n_batches

    return metrics_base, metrics_refined

# ==============================================================================
# 4. MAIN EXECUTION
# ==============================================================================

if __name__ == "__main__":
    m_base, m_refined = run_evaluation()

    # Combine into a single report for comparison
    df_base = pd.DataFrame([m_base])
    df_base['model_type'] = 'Base_Only'
    
    df_refined = pd.DataFrame([m_refined])
    df_refined['model_type'] = 'Base_plus_LSTM'

    final_report = pd.concat([df_base, df_refined], ignore_index=True)
    
    # Save results
    output_dir = "./results/comparison_report"
    os.makedirs(output_dir, exist_ok=True)
    final_report.to_csv(f"{output_dir}/evaluation_comparison.csv", index=False)
    
    print("\n" + "="*30)
    print("EVALUATION COMPLETE")
    print("="*30)
    print(final_report)