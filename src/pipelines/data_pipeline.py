# from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms
# from torchvision.transforms import InterpolationMode
# from dataclasses import dataclass
# from PIL import Image
# import os
# import matplotlib.pyplot as plt
# import json
# import numpy as np
# from src.utils import load_pickle
# from scipy.spatial import cKDTree


# @dataclass
# class DatasetConfig:
#     train_data_path="./artifacts/processed-data/train"
#     test_data_path="./artifacts/processed-data/test"
#     class_to_color_mapping_json='./artifacts/processed-data/color_to_class.pkl'

# class CVC_FP_dataset(Dataset):
#     img_tf = transforms.Compose([
#         transforms.Resize((224,224)),
#         transforms.ToTensor(),
#     ])
#     mask_tf = transforms.Compose([
#         transforms.Resize((224,224), interpolation=InterpolationMode.NEAREST),
#         transforms.ToTensor(),
#     ])
#     def __init__(self,dataset_path=None):
#         super().__init__()
#         self.image_mask_pair_paths=self.get_image_mask_pair_paths(dataset_path)   
#         self.color_to_class_mapping=load_pickle(DatasetConfig.class_to_color_mapping_json) 
#         self.color_to_id_mapping={color:id for id,color in enumerate(self.color_to_class_mapping.keys(),0)} 

#     @staticmethod
#     def get_image_mask_pair_paths(dataset_path=None):

#         list_of_image_mask_pair=[]

#         directories=os.listdir(dataset_path)

#         for directory in directories:
#             folder_of_pair = os.path.join(dataset_path,directory)
#             files=os.listdir(folder_of_pair)

#             # Assuming file[0]=image and file[1]=mask
#             if 'gt' in files[0]:
#                 files[0],files[1]=files[1],files[0]
            
#             files[0]=os.path.join(folder_of_pair,files[0]).replace('\\','/')
#             files[1]=os.path.join(folder_of_pair,files[1]).replace('\\','/')
#             list_of_image_mask_pair.append((files[0],files[1]))

#         return list_of_image_mask_pair
    
#     @classmethod
#     def load_image_and_mask(cls,image_path,mask_path,color_to_class_mapping,color_to_id_mapping):
#         """
#             args:
#                 image_path: path to the image
#                 mask_path: path to the mask
#                 class to color mapping: mappings of class and color.
#             Returns: 
#                 original image in numpy, mask
#         """
#         image=Image.open(image_path).convert("RGB")
#         mask=Image.open(mask_path).convert("RGB")
#         mask_np=np.array(mask)
#         unique_colors=color_to_class_mapping.keys()
#         # print(unique_colors)
#         numpyed_unique_color=np.array(list(unique_colors))
        
#         kd_search_tree=cKDTree(numpyed_unique_color)

#         _,points=kd_search_tree.query(mask_np)
#         fixed_mask_from_closest_points=numpyed_unique_color[points]
#         mask_height,mask_width=fixed_mask_from_closest_points.shape[:-1]
#         refined_mask=np.zeros((mask_height,mask_width),dtype=np.int64)
        
#         for color,id in color_to_id_mapping.items():
#             refined_mask[np.all(fixed_mask_from_closest_points==color,axis=-1)]=np.int_(id)

#         ### ADDED 
#         # WALL_ORIGINAL_ID = 2 
#         # binary_mask = np.where(refined_mask == WALL_ORIGINAL_ID, 1, 0)
#         ### ADDED END
#         image=cls.img_tf(image)
#         mask=cls.mask_tf(Image.fromarray(refined_mask.astype(np.int32))).squeeze(0).long()
#         # mask = cls.mask_tf(Image.fromarray(binary_mask.astype(np.uint8))).squeeze(0).long()
#         return image,mask
        

#     def __len__(self):
#         return len(self.image_mask_pair_paths)

#     def __getitem__(self, index):
#         image,mask=self.load_image_and_mask(*self.image_mask_pair_paths[index],
#                                             self.color_to_class_mapping,
#                                             self.color_to_id_mapping)
#         # image=self.img_tf(image)
#         # mask=self.mask_tf(image)
#         return image,mask
    
# def get_train_test_loader(batch_size=16):
#     train_dataset=CVC_FP_dataset(DatasetConfig.train_data_path)
#     test_dataset=CVC_FP_dataset(DatasetConfig.test_data_path)
#     train_data_loader=DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True,num_workers=0)
#     test_data_loader=DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=False,num_workers=0)
#     return train_data_loader,test_data_loader

    

# if __name__=="__main__":
#     train_dataset=CVC_FP_dataset(DatasetConfig.train_data_path)
#     image,mask=train_dataset[0]
#     # print(mask)

#     image_np = image.permute(1, 2, 0).cpu().numpy()


#     mask_np = mask.cpu().numpy()

#     print(image_np.shape,mask_np.shape)

#     plt.figure(figsize=(8,4))

#     plt.subplot(1, 2, 1)
#     plt.imshow(image_np)
#     plt.title("Image")
#     plt.axis("off")

#     plt.subplot(1, 2, 2)
#     plt.imshow(mask_np)
#     plt.title("Mask")
#     plt.axis("off")

#     plt.show()

    

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from dataclasses import dataclass
from PIL import Image
import os
import matplotlib.pyplot as plt
import numpy as np
import pickle

@dataclass
class DatasetConfig:
    train_data_path: str = "./artifacts/processed-data/train"
    test_data_path: str = "./artifacts/processed-data/test"
    class_to_color_mapping_pkl: str = './artifacts/processed-data/color_to_class.pkl'

class CVC_FP_dataset(Dataset):
    # Image transform: Normalizes to [0, 1]
    img_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    # Mask transform: Resize using NEAREST to avoid creating "blended" new colors
    mask_tf = transforms.Compose([
        transforms.Resize((224, 224), interpolation=InterpolationMode.NEAREST),
    ])

    def __init__(self, dataset_path=None):
        super().__init__()
        if dataset_path is None or not os.path.exists(dataset_path):
            raise ValueError(f"Invalid dataset path: {dataset_path}")
            
        # 1. Load the pickle mapping first to know what we are looking for
        with open(DatasetConfig.class_to_color_mapping_pkl, 'rb') as f:
            self.color_to_class_mapping = pickle.load(f)
        
        # Create an RGB tuple -> Integer ID lookup table
        self.color_to_id = {tuple(color): i for i, color in enumerate(self.color_to_class_mapping.keys())}
        
        # 2. Get paired paths with corrected logic
        self.image_mask_pair_paths = self.get_image_mask_pair_paths(dataset_path)

    @staticmethod
    def get_image_mask_pair_paths(dataset_path):
        list_of_image_mask_pair = []
        directories = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
        
        for directory in directories:
            folder_of_pair = os.path.join(dataset_path, directory)
            files = [f for f in os.listdir(folder_of_pair) if not f.startswith('.')]

            if len(files) < 2:
                continue

            # LOGIC FIX: 
            # In your DataTransformation, the mask is the file converted from SVG.
            # Usually, the raw image has its original name (e.g. 'room_1.jpg') 
            # and the mask is saved as 'room_1.png'.
            
            # We identify the mask by finding the file that was generated from the SVG.
            # In your transformation script: png_path = f"{processed_data_dir}/{index}/{file_name.split('.')[0]}.png"
            
            # Identify files
            svg_mask = None
            raw_image = None

            # Sort files by length - often the mask is the shorter/cleaner name if it was split
            # But a safer way is to check the file extension or specific naming patterns
            png_files = [f for f in files if f.lower().endswith('.png')]
            other_files = [f for f in files if not f.lower().endswith('.png')]

            if len(other_files) > 0:
                # If there's a non-png file (jpg/jpeg/bmp), that's definitely the raw image
                raw_image = other_files[0]
                svg_mask = png_files[0]
            else:
                # If both are PNGs, usually the one without 'fixed' or the one that 
                # matches the subfolder naming logic is the mask. 
                # Let's use the list sorting as a fallback but check for common patterns.
                svg_mask = files[0]
                raw_image = files[1]
                
                # Check if file names contain hints (like 'gt', 'mask', or 'fixed')
                for f in files:
                    if any(x in f.lower() for x in ['mask', 'gt', 'fixed']):
                        svg_mask = f
                        raw_image = [img for img in files if img != f][0]
                        break

            mask_path = os.path.join(folder_of_pair, svg_mask).replace('\\', '/')
            img_path = os.path.join(folder_of_pair, raw_image).replace('\\', '/')
            
            list_of_image_mask_pair.append((img_path, mask_path))

        return list_of_image_mask_pair

    def encode_segmap(self, mask_np):
        mask_height, mask_width = mask_np.shape[:2]
        refined_mask = np.zeros((mask_height, mask_width), dtype=np.int64)
        
        for color, class_id in self.color_to_id.items():
            match = np.all(mask_np == np.array(color), axis=-1)
            refined_mask[match] = class_id
            
        return refined_mask

    def __len__(self):
        return len(self.image_mask_pair_paths)

    def __getitem__(self, index):
        img_path, mask_path = self.image_mask_pair_paths[index]
        
        # Load Raw Image
        image = Image.open(img_path).convert("RGB")
        image = self.img_tf(image)
        
        # Load Mask
        mask_rgb = Image.open(mask_path).convert("RGB")
        mask_resized = self.mask_tf(mask_rgb) 
        mask_np = np.array(mask_resized)
        
        # Convert RGB to ID map
        refined_mask = self.encode_segmap(mask_np)
        
        return image, torch.from_numpy(refined_mask).long()
    

def get_train_test_loader(batch_size=16):
    train_dataset=CVC_FP_dataset(DatasetConfig.train_data_path)
    test_dataset=CVC_FP_dataset(DatasetConfig.test_data_path)
    train_data_loader=DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True,num_workers=0)
    test_data_loader=DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=False,num_workers=0)
    return train_data_loader,test_data_loader


if __name__ == "__main__":
    try:
        train_ds = CVC_FP_dataset(DatasetConfig.train_data_path)
        img, msk = train_ds[50]
        
        img_plot = img.permute(1, 2, 0).numpy()
        msk_plot = msk.numpy()

        plt.figure(figsize=(12, 6))
        
        # Subplot 1: The real photo
        plt.subplot(1, 2, 1)
        plt.title("Transformed Raw Image")
        plt.imshow(img_plot)
        plt.axis('off')

        # Subplot 2: The segmentation mask
        plt.subplot(1, 2, 2)
        plt.title("Class ID Mask")
        plt.imshow(msk_plot) 
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Error: {e}")