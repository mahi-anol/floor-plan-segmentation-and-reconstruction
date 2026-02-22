import torch
from src.pipelines.data_pipeline import  get_train_test_loader
from src.components.legacy_models.model_mod_3 import get_model
from src.components.eval_metrices import get_eval_metrices_outcome
import pandas as pd
from dataclasses import dataclass
import os

@dataclass
class configs:
    checkpoint_locations=[
            r"E:\floor-plan-segmentation-and-reconstruction\training_results\trial-6\Best.pt",
            r"E:\floor-plan-segmentation-and-reconstruction\training_results\trial-2\Best.pt",
        ]
    device='cpu' if not torch.cuda.is_available() else 'cuda'
    No_of_class=2
    batch_size=16

def get_trained_models_with_weights(checkpoint_loc=None,No_of_class=configs.No_of_class):
    checkpoint=torch.load(checkpoint_loc,map_location=configs.device)
    model=get_model(image_channel=3,number_of_class=No_of_class)
    model.load_state_dict(state_dict=checkpoint['model'])
    return model


def get_models_evaluation(checkpoint_locations=configs.checkpoint_locations):

    _,test_data_loader=get_train_test_loader(batch_size=configs.batch_size)

    results=[]
    for checkpoint_loc in checkpoint_locations:
        model=get_trained_models_with_weights(checkpoint_loc)
        model.eval()
        result={}
        for input,gt in test_data_loader:
            with torch.no_grad():
                logits=model(input)
                logits=torch.argmax(input=logits,dim=1).cpu().numpy()
                gt=gt.cpu().numpy()
                curr_result=get_eval_metrices_outcome(gt,logits,num_class=configs.No_of_class)
                for k,v in curr_result.items():
                    result[k]=result.get(k,0)+v

        n_batch=len(test_data_loader)

        for k,v in result.items():
            result[k]/=n_batch

        results.append(result)
    return results


if __name__=="__main__":
    results=get_models_evaluation()

    for i,result in enumerate(results):
        result_df=pd.DataFrame(result)
        desired_folder=os.path.basename(os.path.dirname(configs.checkpoint_locations[i]))
        os.makedirs(f"./results/{desired_folder}",exist_ok=True)
        result_df.to_csv(f"./results/{desired_folder}/eval_metric.csv")