import torch
import torch.nn as nn
import pandas as pd
import os
import time
import psutil
from dataclasses import dataclass
from src.pipelines.data_pipeline import get_train_test_loader
# from src.components.dev_models.Novel_v1.model import get_model
from src.components.legacy_models.model_mod_3 import get_model

# ==============================================================================
# 1. ARCHITECTURE REPLICATION
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
    seg_model_path = r"E:\floor-plan-segmentation-and-reconstruction\training_results\trial-2\Best.pt"
    refiner_model_path = r"./training_results/refiner/refiner_best.pt"
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_classes = 2
    batch_size = 1  # Forced to 1 for single image inference

# ==============================================================================
# 3. HELPER FUNCTIONS
# ==============================================================================

def get_memory_usage(device):
    """Returns memory usage in MB."""
    if device == 'cuda':
        return torch.cuda.max_memory_allocated() / (1024 ** 2)
    else:
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 ** 2)

def reset_memory_stats(device):
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()

# ==============================================================================
# 4. BENCHMARK ENGINE
# ==============================================================================

def load_models():
    # Load Segmentation Model
    print(f"[INFO] Loading Segmentation Model...")
    seg_model = get_model(image_channel=3, number_of_class=Configs.num_classes,deploy=True)
    # checkpoint = torch.load(Configs.seg_model_path, map_location=Configs.device)
    # state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
    # seg_model.load_state_dict(state_dict)
    seg_model.to(Configs.device).eval()

    # Load Refiner
    print(f"[INFO] Loading Recurrent Refiner...")
    refiner = RecurrentRefiner(num_classes=Configs.num_classes)
    refiner.load_state_dict(torch.load(Configs.refiner_model_path, map_location=Configs.device))
    refiner.to(Configs.device).eval()
    
    return seg_model, refiner

def benchmark_single_image():
    print(f"[INFO] Starting Single Image Benchmark on {Configs.device}...")
    
    # 1. Get Data
    # We grab just the first batch (which is size 1)
    _, test_loader = get_train_test_loader(batch_size=Configs.batch_size)
    images, _ = next(iter(test_loader))
    images = images.to(Configs.device)

    print(f"[INFO] Input Shape: {images.shape}")
    
    seg_model, refiner = load_models()

    # 2. Warmup
    # Run a few dummy passes to initialize CUDA kernels so they don't affect timing
    print("[INFO] Warming up GPU/CPU...")
    with torch.no_grad():
        for _ in range(5):
            _ = seg_model(images)

    # ---------------------------------------------------------
    # 3. Benchmark: Segmentation Model Only
    # ---------------------------------------------------------
    reset_memory_stats(Configs.device)
    if Configs.device == 'cuda': torch.cuda.synchronize()
    
    start_time = time.time()
    with torch.no_grad():
        coarse_logits = seg_model(images)
    
    if Configs.device == 'cuda': torch.cuda.synchronize()
    end_time = time.time()
    
    seg_time_ms = (end_time - start_time) * 1000
    seg_ram_mb = get_memory_usage(Configs.device)

    # ---------------------------------------------------------
    # 4. Benchmark: Full Pipeline (Seg + LSTM Refiner)
    # ---------------------------------------------------------
    reset_memory_stats(Configs.device)
    if Configs.device == 'cuda': torch.cuda.synchronize()
    
    start_time = time.time()
    with torch.no_grad():
        # NOTE: We re-run seg_model here because in a real pipeline
        # you have to run the base model first to get logits for the refiner.
        c_logits = seg_model(images) 
        refined_logits = refiner(images, c_logits)
    
    if Configs.device == 'cuda': torch.cuda.synchronize()
    end_time = time.time()
    
    ref_time_ms = (end_time - start_time) * 1000
    ref_ram_mb = get_memory_usage(Configs.device)

    return {
        "Segmentation_Only": {"Inference_Time_ms": seg_time_ms, "Peak_RAM_MB": seg_ram_mb},
        "Segmentation_plus_LSTM": {"Inference_Time_ms": ref_time_ms, "Peak_RAM_MB": ref_ram_mb}
    }

# ==============================================================================
# 5. MAIN EXECUTION
# ==============================================================================

if __name__ == "__main__":
    results = benchmark_single_image()

    # Create DataFrame
    df = pd.DataFrame(results).T.reset_index()
    df.rename(columns={'index': 'Model_Type'}, inplace=True)
    
    # Save results
    output_dir = "./results/performance_report"
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(f"{output_dir}/single_image_benchmark.csv", index=False)
    
    print("\n" + "="*40)
    print("SINGLE IMAGE INFERENCE BENCHMARK")
    print("="*40)
    print(df.to_string(index=False))
    print("="*40)