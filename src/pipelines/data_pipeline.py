import torch
import torch.nn as nn 
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
from PIL import Image
import os
import json
import numpy as np
from src.utils import load_pickle

@dataclass
class DatasetConfig:
    train_data_path="./artifacts/processed-data/train"
    test_data_path="./artifacts/processed-data/test"
    class_to_color_mapping_json='./artifacts/processed-data/class_to_color.pkl'

class CVC_FP_dataset(Dataset):
    def __init__(self,dataset_path=None):
        super().__init__()
        self.image_mask_pair_paths=self.get_image_mask_pair_paths(dataset_path)   
        self.class_to_color_mapping=load_pickle(DatasetConfig.class_to_color_mapping_json)    

    @staticmethod
    def get_image_mask_pair_paths(dataset_path=None):

        list_of_image_mask_pair=[]

        directories=os.listdir(dataset_path)

        for directory in directories:
            folder_of_pair = os.path.join(dataset_path,directory)
            files=os.listdir(folder_of_pair)

            # Assuming file[0]=image and file[1]=mask
            if 'gt' in files[0]:
                files[0],files[1]=files[1],files[0]
            
            files[0]=os.path.join(folder_of_pair,files[0]).replace('\\','/')
            files[1]=os.path.join(folder_of_pair,files[1]).replace('\\','/')
            
            list_of_image_mask_pair.append((files[0],files[1]))

        return list_of_image_mask_pair
    
    @staticmethod
    def load_image_and_mask(image_path,mask_path):
        image=Image.open(image_path).convert("L")
        mask=Image.open(mask_path).convert("RGB")
        mask_np=np.array(mask)
        unique_RGB=set()
        for x in mask_np:
            for y in x:
                unique_RGB.add(tuple(y.tolist()))
        return image,mask,unique_RGB
    
    def __len__(self):
        return len(self.image_mask_pair_paths)

    def __getitem__(self, index):
        image,mask,unique_RGB=self.load_image_and_mask(*self.image_mask_pair_paths[index])
        return image,mask,unique_RGB
    

if __name__=="__main__":
    dataset=CVC_FP_dataset(DatasetConfig.train_data_path)
    print(dataset[0][2])


