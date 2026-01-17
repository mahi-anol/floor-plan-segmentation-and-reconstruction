import random
import numpy as np
import torch
from src.pipelines.data_pipeline import get_train_test_loader
from src.components.model_mod_3 import get_model
from tqdm import tqdm
import torch
import logging
import os
from src.utils import saving_model_with_state_and_logs,ThreeMusketeerLoss
import torch.optim as optim
import torch.nn as nn
import optuna
from torch.optim.lr_scheduler import ReduceLROnPlateau 

# deterministic behavior
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

device='cuda' if torch.cuda.is_available() else 'cpu'

def load_trainer_objects(hyperParameters,image_channel=3,number_of_class=2,resume_from_prev_state=False):
    if not os.path.exists('./artifacts/augmented'):
        from src.components import data_augmentation
        data_augmentation.run_augmentation()

    # Dataset
    train_dataset_loader,test_dataset_loader=get_train_test_loader(batch_size=hyperParameters['batch_size'])
    # Model
    model=get_model(image_channel=image_channel,number_of_class=number_of_class)

    model.to(device)

    optimizer=getattr(optim,hyperParameters['optimizer_name'])(model.parameters(),lr=hyperParameters['lr'],weight_decay=hyperParameters['weight_decay'])
    # loss_fn=nn.CrossEntropyLoss()
    # loss_fn=MulticlassDiceCELoss()
    loss_fn=ThreeMusketeerLoss()

    scheduler=ReduceLROnPlateau(
        optimizer=optimizer,
        mode='max',
        factor=1e-1,
        patience=4,
        threshold=0.05,
        threshold_mode='abs',
        min_lr=1e-6,
        cooldown=2
    )

    ### Loading Prev states
    if resume_from_prev_state:
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

    return {
        'train_test_loader':[train_dataset_loader,test_dataset_loader],
        'model':model,
        'optimizer':optimizer,
        'loss_fn':loss_fn,
        'scheduler':scheduler
    }


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

def train(hyperParameters,trial_number=-1,image_channels=3,number_of_class=2,epochs=200,checkpoint_saving_gap=1,resume_from_prev_state=False):

    train_objects=load_trainer_objects(hyperParameters,
                                       image_channel=image_channels,
                                       number_of_class=number_of_class,
                                       resume_from_prev_state=resume_from_prev_state) 

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
        train_accuracy,train_loss=train_step(train_objects['model'],train_objects['train_test_loader'][0],train_objects['loss_fn'],train_objects['optimizer'],device)
        test_accuracy,test_loss=test_step(train_objects['model'],train_objects['train_test_loader'][1],train_objects['loss_fn'],device)
        train_objects['scheduler'].step(test_accuracy)

        logging.info(f'Epoch: {epoch+1} | Train loss: {train_loss:.4f} | Train acc: {train_accuracy:.4f} | Test loss: {test_loss:.4f} | Test acc: {test_accuracy:.4f}')
        
        results['train_accuracy'].append(train_accuracy)
        results['train_loss'].append(train_loss)
        results["test_accuracy"].append(test_accuracy)
        results['test_loss'].append(test_loss)
         # Save checkpoint every nth epoch
        if (epoch + 1) % checkpoint_saving_gap == 0:
            saving_model_with_state_and_logs(train_objects['model'], train_objects['optimizer'],trial_number,results,f"Epoch-{epoch+1}_trained_model.pt")
            logging.info(f"Saved epoch checkpoint at epoch {epoch+1}")
        # Save best model based on test accuracy
        if test_accuracy > best_test_accuracy:
            logging.info("weights from current epoch outperformed previous weights. ")
            best_test_accuracy = test_accuracy
            logging.info(f"Saving best model with Test Accuracy: {best_test_accuracy:.4f} at epoch {epoch+1} @ ./checkpoints")
            # When saving 'best.pt', ensure 'results' reflects the metrics *up to that point*.
            # A shallow copy is usually sufficient if saving_model_with_state_and_logs doesn't modify it.
            # Using slice [:] creates a shallow copy of the lists within results.
            current_results_for_best = {k: v[:] for k, v in results.items()} 
            saving_model_with_state_and_logs(train_objects['model'], train_objects['optimizer'],trial_number,current_results_for_best,f"Best.pt")

    # After the training loop finishes, save the last model
    logging.info("Saving last trained model @ ./checkpoints")
    saving_model_with_state_and_logs(train_objects['model'], train_objects['optimizer'],trial_number, results,f"Last.pt")
    return results

def hyperParameterOptimizer(trial):
    optimizer_name=trial.suggest_categorical("optimizer_name",['Adam','AdamW'])
    # lr=trial.suggest_float("lr",1e-5,1e-1,log=True) #contineous space
    lr=trial.suggest_categorical("lr",[1e-2,1e-3,1e-4]) 
    batch_size=trial.suggest_categorical("batch_size",[8,32,128])
    # weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True) # contineuous space
    weight_decay=trial.suggest_categorical("weight_decay",[1e-4,1e-3,1e-5]) #catagorical space

    hyperParameters={'optimizer_name':optimizer_name,'lr':lr,'batch_size':batch_size,'weight_decay':weight_decay}


    epochs=50
    logging.info(f"Starting a training session for {epochs} Epochs")
    logging.info("Going with trial no : %s",trial.number)
    logging.info("Used Parameters: %s",trial.params)
    logging.info(f"Starting training using : {device}")
    results=train(
                    hyperParameters,
                    trial_number=trial.number,
                    image_channels=3,
                    number_of_class=2,
                    epochs=epochs,
                    checkpoint_saving_gap=1,
                    resume_from_prev_state=False
            )
    
    return results['test_accuracy']
    
if __name__=="__main__":
    study=optuna.create_study(direction="maximize",study_name="Floor plan model hyperParameter Tunning")
    study.optimize(func=hyperParameterOptimizer,n_trials=10,show_progress_bar=True)

    logging.info("Best Test Acc: %s",study.best_value)
    logging.info("Best Params: %s",study.best_params)
    logging.info("Best trial Number: %s",study.best_trial.number)
    logging.info("Best checkpoint is stored on : %s",f"./checkpoints/trial-{study.best_trial.number}")
    