import random
import numpy as np
import torch
from src.pipelines.data_pipeline_v2 import get_train_test_loader
from src.components.dev_models.Novel_V2.model import get_model
from tqdm import tqdm
import logging
import os
from src.utils import saving_model_with_state_and_logs, ThreeMusketeerLoss, ArConsistencyLoss
# Import the new spatial loss
from src.components.dev_models.Novel_V2.spatial_loss import SpatialEdgeLoss
import torch.optim as optim
import torch.nn as nn

# deterministic behavior
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

if not os.path.exists('./artifacts/augmented'):
    from src.components import data_augmentation
    data_augmentation.run_augmentation()

# Dataset
train_dataset_loader, test_dataset_loader = get_train_test_loader(batch_size=16)

# Model
model = get_model(image_channel=3, number_of_class=2)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

# --- LOSS FUNCTIONS ---
# 1. Main Segmentation Loss
seg_loss_fn = ArConsistencyLoss() 
seg_loss_fn.to(device)

# 2. New Edge/Spatial Loss
edge_loss_fn = SpatialEdgeLoss()
edge_loss_fn.to(device)

def train_step(model, data_loader, seg_loss_fn, edge_loss_fn, optimizer, device):
    model.train()
    train_loss, train_accuracy = 0.0, 0.0
    bar = tqdm(data_loader, desc='Training Epoch going on', leave=False)
    
    for batch, (x, Y) in enumerate(bar):
        x, Y = x.to(device), Y.to(device)
        
        # Forward pass returns TWO outputs now
        seg_logits, edge_logits = model(x)
        
        # 1. Calculate Segmentation Loss
        loss_segmentation = seg_loss_fn(seg_logits, Y)
        
        # 2. Calculate Edge Loss (Sobel is applied internally to Y)
        loss_edge = edge_loss_fn(edge_logits, Y)
        
        # 3. Combine Losses (Weighted Sum)
        # You can tune 'lambda_edge' (e.g., 0.5) to balance the tasks
        lambda_edge = 0.5
        total_loss = loss_segmentation + (lambda_edge * loss_edge)
        
        train_loss += total_loss.item()
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        # Metrics (based on Segmentation Head)
        y_pred = torch.argmax(torch.softmax(seg_logits, dim=1), dim=1)
        batch_accuracy = (y_pred == Y).sum().item() / y_pred.numel()
        train_accuracy += batch_accuracy
        
        bar.set_postfix(
            loss=f'{total_loss.item():.4f}', 
            seg_l=f'{loss_segmentation.item():.2f}',
            edge_l=f'{loss_edge.item():.2f}',
            acc=f'{batch_accuracy*100:.1f}%'
        )
        
    train_loss = train_loss / len(data_loader)
    train_accuracy = (train_accuracy / len(data_loader)) * 100
    return train_accuracy, train_loss

def test_step(model, data_loader, seg_loss_fn, edge_loss_fn, device):
    model.eval()
    test_loss, test_accuracy = 0.0, 0.0
    with torch.inference_mode():
        bar = tqdm(data_loader, desc='Testing Epoch going on', leave=False)
        for batch, (x, Y) in enumerate(bar):
            x, Y = x.to(device), Y.to(device)
            
            seg_logits, edge_logits = model(x)
            
            loss_segmentation = seg_loss_fn(seg_logits, Y)
            loss_edge = edge_loss_fn(edge_logits, Y)
            
            total_loss = loss_segmentation + (0.5 * loss_edge)
            test_loss += total_loss.item()
            
            y_pred = torch.argmax(torch.softmax(seg_logits, dim=1), dim=1)
            batch_accuracy = (y_pred == Y).sum().item() / y_pred.numel()
            test_accuracy += batch_accuracy
            
            bar.set_postfix(loss=f'{total_loss.item():.4f}', acc=f'{batch_accuracy*100:.1f}%')
            
        test_accuracy = (test_accuracy / len(data_loader)) * 100
        test_loss = test_loss / len(data_loader)
    return test_accuracy, test_loss

def train(model, train_dataloader, test_dataloader, optimizer, seg_loss_fn, edge_loss_fn, epochs, device, checkpoint_saving_gap, resume_from_previous_state, exp_no):
    # ... (Loading Logic remains the same) ...
    logging.info(f"Starting training using : {device}")

    best_test_accuracy = -float('inf') 
    
    results = {
        "train_loss": [],
        "test_loss": [],
        "train_accuracy": [],
        "test_accuracy": [],
    }
    
    for epoch in range(epochs):
        # Updated to pass both loss functions
        train_accuracy, train_loss = train_step(model, train_dataloader, seg_loss_fn, edge_loss_fn, optimizer, device)
        test_accuracy, test_loss = test_step(model, test_dataloader, seg_loss_fn, edge_loss_fn, device)
        
        logging.info(f'Epoch: {epoch+1} | Train loss: {train_loss:.4f} | Train acc: {train_accuracy:.4f} | Test loss: {test_loss:.4f} | Test acc: {test_accuracy:.4f}')
        
        results['train_accuracy'].append(train_accuracy)
        results['train_loss'].append(train_loss)
        results["test_accuracy"].append(test_accuracy)
        results['test_loss'].append(test_loss)

        if (epoch + 1) % checkpoint_saving_gap == 0:
            saving_model_with_state_and_logs(model, optimizer, exp_no, results, f"Epoch-{epoch+1}_trained_model.pt")
            logging.info(f"Saved epoch checkpoint at epoch {epoch+1}")
            
        if test_accuracy > best_test_accuracy:
            best_test_accuracy = test_accuracy
            logging.info(f"Saving best model with Test Accuracy: {best_test_accuracy:.4f}")
            current_results_for_best = {k: v[:] for k, v in results.items()} 
            saving_model_with_state_and_logs(model, optimizer, exp_no, current_results_for_best, "Best.pt")

    logging.info("Saving last trained model")
    saving_model_with_state_and_logs(model, optimizer, exp_no, results, "Last.pt")
    return results

# Update the main execution call
train(
    model=model,
    train_dataloader=train_dataset_loader,
    test_dataloader=test_dataset_loader,
    optimizer=optimizer,
    seg_loss_fn=seg_loss_fn,    # Pass seg loss
    edge_loss_fn=edge_loss_fn,  # Pass edge loss
    epochs=100,
    device=device,
    checkpoint_saving_gap=1,
    resume_from_previous_state=False,
    exp_no=3
)