from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from dataclasses import dataclass
from PIL import Image
import os
import matplotlib.pyplot as plt
import json
import numpy as np
from src.utils import load_pickle
from scipy.spatial import cKDTree


@dataclass
class DatasetConfig:
    train_data_path="./artifacts/processed-data/train"
    test_data_path="./artifacts/processed-data/test"
    class_to_color_mapping_json='./artifacts/processed-data/color_to_class.pkl'

class CVC_FP_dataset(Dataset):
    img_tf = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
    ])
    mask_tf = transforms.Compose([
        transforms.Resize((224,224), interpolation=InterpolationMode.NEAREST),
        transforms.ToTensor(),
    ])
    def __init__(self,dataset_path=None):
        super().__init__()
        self.image_mask_pair_paths=self.get_image_mask_pair_paths(dataset_path)   
        self.color_to_class_mapping=load_pickle(DatasetConfig.class_to_color_mapping_json) 
        self.color_to_id_mapping={color:id for id,color in enumerate(self.color_to_class_mapping.keys(),0)} 

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
    
    @classmethod
    def load_image_and_mask(cls,image_path,mask_path,color_to_class_mapping,color_to_id_mapping):
        """
            args:
                image_path: path to the image
                mask_path: path to the mask
                class to color mapping: mappings of class and color.
            Returns: 
                original image in numpy, mask
        """
        image=Image.open(image_path).convert("RGB")
        mask=Image.open(mask_path).convert("RGB")
        mask_np=np.array(mask)
        unique_colors=color_to_class_mapping.keys()
        # print(unique_colors)
        numpyed_unique_color=np.array(list(unique_colors))
        
        kd_search_tree=cKDTree(numpyed_unique_color)

        _,points=kd_search_tree.query(mask_np)
        fixed_mask_from_closest_points=numpyed_unique_color[points]
        mask_height,mask_width=fixed_mask_from_closest_points.shape[:-1]
        refined_mask=np.zeros((mask_height,mask_width),dtype=np.int64)
        
        for color,id in color_to_id_mapping.items():
            refined_mask[np.all(fixed_mask_from_closest_points==color,axis=-1)]=np.int_(id)

        ### ADDED 
        # WALL_ORIGINAL_ID = 3 
        # binary_mask = np.where(refined_mask == WALL_ORIGINAL_ID, 1, 0)
        ### ADDED END
        image=cls.img_tf(image)
        mask=cls.mask_tf(Image.fromarray(refined_mask.astype(np.int32))).squeeze(0).long()
        # mask = cls.mask_tf(Image.fromarray(binary_mask.astype(np.uint8))).squeeze(0).long()
        return image,mask
        

    def __len__(self):
        return len(self.image_mask_pair_paths)

    def __getitem__(self, index):
        image,mask=self.load_image_and_mask(*self.image_mask_pair_paths[index],
                                            self.color_to_class_mapping,
                                            self.color_to_id_mapping)
        # image=self.img_tf(image)
        # mask=self.mask_tf(image)
        return image,mask
    
def get_train_test_loader(batch_size=16):
    train_dataset=CVC_FP_dataset(DatasetConfig.train_data_path)
    test_dataset=CVC_FP_dataset(DatasetConfig.test_data_path)
    train_data_loader=DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True,num_workers=0)
    test_data_loader=DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=False,num_workers=0)
    return train_data_loader,test_data_loader

    

if __name__=="__main__":
    train_dataset=CVC_FP_dataset(DatasetConfig.train_data_path)
    image,mask=train_dataset[0]
    # print(mask)

    image_np = image.permute(1, 2, 0).cpu().numpy()


    mask_np = mask.cpu().numpy()

    print(image_np.shape,mask_np.shape)

    plt.figure(figsize=(8,4))

    plt.subplot(1, 2, 1)
    plt.imshow(image_np)
    plt.title("Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(mask_np)
    plt.title("Mask")
    plt.axis("off")

    plt.show()

    


