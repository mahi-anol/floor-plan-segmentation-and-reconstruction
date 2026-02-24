import random
import numpy as np
import torch
from src.pipelines.data_pipeline.data_pipeline_cubicasa import get_train_test_loader
from src.components.dev_models.unet_plus_plus.model import get_model
from torch.optim.lr_scheduler import ReduceLROnPlateau,StepLR
from tqdm import tqdm
import torch
import logging
import os
from src.utils import saving_model_with_state_and_logs,ThreeMusketeerLoss,PolyTverskyLoss,ArConsistencyLoss
import torch.optim as optim
import torch.nn as nn

# deterministic behavior
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

### For changing model config or transferlearn/finetune config ....Need to go to src/model/__init__.py

if not os.path.exists('./artifacts/augmented'):
    from src.components import data_augmentation
    data_augmentation.run_augmentation()

# Dataset
train_dataset_loader,test_dataset_loader=get_train_test_loader(batch_size=8)
# Model
model=get_model(image_channel=3,number_of_class=2)

device='cuda' if torch.cuda.is_available() else 'cpu'

model.to(device)
optimizer=optim.Adam(model.parameters(),lr=1e-5,weight_decay=1e-4)
# loss_fn=nn.CrossEntropyLoss()
# loss_fn=MulticlassDiceCELoss()
loss_fn=ThreeMusketeerLoss(loss_weights=[0.5,0.5,1.0])
# loss_fn = PolyTverskyLoss(alpha=0.7, beta=0.3)
# loss_fn=ArConsistencyLoss()
loss_fn.to(device)


# scheduler = StepLR(
#     optimizer,
#     step_size=5,  # decrease every 5 epochs
#     gamma=10     # multiply LR by 10 or 0.1
# )

def train_step(model,data_loader,loss_fn,optimizer,device):
    model.train()
    train_loss,train_accuracy=0.0,0.0
    bar=tqdm(data_loader,desc='Training Epoch going on',leave=False)
    for batch,(x,Y) in enumerate(bar):
        x,Y=x.to(device),Y.to(device)
        # print("type of Y is ",type(x))
        logits=model(x)
        batch_loss=loss_fn(logits,Y)
        train_loss+=batch_loss.item()
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
        y_pred=torch.argmax(torch.softmax(logits,dim=1),dim=1)
        batch_accuracy=(y_pred==Y).sum().item()/y_pred.numel()
        train_accuracy+=batch_accuracy
        bar.set_postfix(batch_loss=f'{batch_loss}',batch_accuracy=f'{batch_accuracy*100}%')
    train_loss=train_loss/len(data_loader)
    train_accuracy=(train_accuracy/len(data_loader))*100
    return train_accuracy,train_loss

def test_step(model,data_loader,loss_fn,device):
    model.eval()
    test_loss, test_accuracy=0.0,0.0
    with torch.inference_mode():
        bar=tqdm(data_loader,desc='Testing Epoch going on',leave=False)
        for batch,(x,Y) in enumerate(bar):
            x,Y=x.to(device),Y.to(device)
            logits=model(x)
            batch_loss=loss_fn(logits,Y)
            test_loss+=batch_loss.item()
            y_pred=torch.argmax(torch.softmax(logits,dim=1),dim=1)
            batch_accuracy=(y_pred==Y).sum().item()/y_pred.numel()
            test_accuracy+=batch_accuracy
            bar.set_postfix(batch_loss=f'{batch_loss}',batch_accuracy=f'{batch_accuracy*100}%')
        test_accuracy=(test_accuracy/len(data_loader))*100
        test_loss=test_loss/len(data_loader)
    return test_accuracy,test_loss

def train(model,train_dataloader,test_dataloader,optimizer,loss_fn,epochs,device,checkpoint_saving_gap,resume_from_previous_state,exp):

    ### Loading Prev states
    logging.info(f"Starting training using : {device}")
    if resume_from_previous_state:
        try:
            checkpoint = torch.load("./checkpoints/cross_entropy_best.pt", map_location=device)
        except Exception as e:
            logging.error(f"Encounted following error while loading the previous checkpoint: {e}")
        
        if "model" in checkpoint:
            model.load_state_dict(checkpoint["model"])
            logging.info("Successfully loaded the weights for best epoch")
        else:
            logging.info("Error while loading weights")
            
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
            logging.info("Succesfully loaded the optimizer state for best epoch")
        else:
            logging.info("Error while loading optimzier state.")
    ###

    best_test_accuracy = -float('inf') 
    best_test_loss=float('inf')
    ### storing logs
    results={
        "train_loss":[],
        "test_loss":[],
        "train_accuracy":[],
        "test_accuracy":[],
    }
    for epoch in range(epochs):
        current_lr = optimizer.param_groups[0]['lr']
        train_accuracy,train_loss=train_step(model,train_dataloader,loss_fn,optimizer,device)
        test_accuracy,test_loss=test_step(model,test_dataloader,loss_fn,device)
        logging.info(f'Epoch: {epoch+1} | Train loss: {train_loss:.4f} | Train acc: {train_accuracy:.4f} | Test loss: {test_loss:.4f} | Test acc: {test_accuracy:.4f}')
        results['train_accuracy'].append(train_accuracy)
        results['train_loss'].append(train_loss)
        results["test_accuracy"].append(test_accuracy)
        results['test_loss'].append(test_loss)
         # Save checkpoint every nth epoch
        if (epoch + 1) % checkpoint_saving_gap == 0:
            saving_model_with_state_and_logs(model, optimizer,exp, results, f"Epoch-{epoch+1}_trained_model.pt")
            logging.info(f"Saved epoch checkpoint at epoch {epoch+1}")
        # Save best model based on test accuracy
        if test_accuracy > best_test_accuracy:
            logging.info("weights from current epoch outperformed previous weights. ")
            best_test_accuracy = test_accuracy
            logging.info(f"Saving best model with Test Accuracy: {best_test_accuracy:.4f} at epoch {epoch+1} @ ./checkpoint")
            # When saving 'best.pt', ensure 'results' reflects the metrics *up to that point*.
            # A shallow copy is usually sufficient if saving_model_with_state_and_logs doesn't modify it.
            # Using slice [:] creates a shallow copy of the lists within results.
            current_results_for_best = {k: v[:] for k, v in results.items()} 
            saving_model_with_state_and_logs(model, optimizer,exp, current_results_for_best, "Best.pt")

        # scheduler.step()
        # updated_lr = optimizer.param_groups[0]['lr']
        # if updated_lr!=current_lr:
        #     logging.info(f"Learning rate has been updated from {current_lr} to {updated_lr}")

    # After the training loop finishes, save the last model
    logging.info("Saving last trained model @ ./models")
    saving_model_with_state_and_logs(model, optimizer,exp, results, "Last.pt")



    return results

# if __name__=="__main__":
train(model=model
    ,train_dataloader=train_dataset_loader
    ,test_dataloader=test_dataset_loader
    ,optimizer=optimizer
    ,loss_fn=loss_fn
    ,epochs=20
    ,device=device
    ,checkpoint_saving_gap=1
    ,resume_from_previous_state=False,
    exp="unetpp"
    )
