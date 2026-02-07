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
import cv2

@dataclass
class DatasetConfig:
    train_data_path: str = "./artifacts/augmented/train"
    test_data_path: str = "./artifacts/augmented/test"
    class_to_color_mapping_pkl: str = './artifacts/processed-data/color_to_class.pkl'
    use_otsu: bool = True 

class CVC_FP_dataset(Dataset):
    # Image transform: Initial resize and normalization to [0, 1]
    img_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    # Mask transform: Resize using NEAREST to preserve discrete class IDs
    mask_tf = transforms.Compose([
        transforms.Resize((224, 224), interpolation=InterpolationMode.NEAREST),
    ])

    def __init__(self, dataset_path=None):
        super().__init__()
        if dataset_path is None or not os.path.exists(dataset_path):
            raise ValueError(f"Invalid dataset path: {dataset_path}")
            
        with open(DatasetConfig.class_to_color_mapping_pkl, 'rb') as f:
            self.color_to_class_mapping = pickle.load(f)
        
        self.color_to_id = {tuple(color): i for i, color in enumerate(self.color_to_class_mapping.keys())}
        self.image_mask_pair_paths = self.get_image_mask_pair_paths(dataset_path)

    @staticmethod
    def get_image_mask_pair_paths(dataset_path):
        list_of_image_mask_pair = []
        directories = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
        
        for directory in directories:
            folder_of_pair = os.path.join(dataset_path, directory)
            files = [f for f in os.listdir(folder_of_pair) if not f.startswith('.')]

            if len(files) < 2: continue

            png_files = [f for f in files if f.lower().endswith('.png')]
            other_files = [f for f in files if not f.lower().endswith('.png')]

            if len(other_files) > 0:
                raw_image = other_files[0]
                svg_mask = png_files[0]
            else:
                svg_mask, raw_image = files[0], files[1]
                for f in files:
                    if any(x in f.lower() for x in ['mask', 'gt', 'fixed']):
                        svg_mask = f
                        raw_image = [img for img in files if img != f][0]
                        break

            mask_path = os.path.join(folder_of_pair, svg_mask).replace('\\', '/')
            img_path = os.path.join(folder_of_pair, raw_image).replace('\\', '/')
            list_of_image_mask_pair.append((img_path, mask_path))

        return list_of_image_mask_pair

    def apply_otsu_threshold(self, image_tensor):
        """
        Converts RGB tensor to Grayscale, applies Otsu's Binarization,
        and repeats it to 3 channels to match model input requirements.
        """
        # 1. Convert tensor to numpy (H, W, C) in range 0-255
        img_np = (image_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        
        # 2. Convert to grayscale for OpenCV
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        
        # 3. Apply Otsu's Thresholding
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 4. Convert back to tensor [1, H, W]
        otsu_1ch = torch.from_numpy(thresh).unsqueeze(0).float() / 255.0
        
        # 5. FIX: Repeat the channel 3 times to create [3, H, W]
        # This resolves: weight size [16, 3, 3, 3] expected 3 channels
        otsu_3ch = otsu_1ch.repeat(3, 1, 1)
        
        return otsu_3ch

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
        
        # Load Image
        image = Image.open(img_path).convert("RGB")
        image = self.img_tf(image)
        
        # Apply Otsu and broadcast to 3 channels if enabled
        if DatasetConfig.use_otsu:
            image = self.apply_otsu_threshold(image)
        
        # Load Mask
        mask_rgb = Image.open(mask_path).convert("RGB")
        mask_resized = self.mask_tf(mask_rgb) 
        mask_np = np.array(mask_resized)
        
        refined_mask = self.encode_segmap(mask_np)
        
        return image, torch.from_numpy(refined_mask).long()

def get_train_test_loader(batch_size=16):
    train_dataset = CVC_FP_dataset(DatasetConfig.train_data_path)
    test_dataset = CVC_FP_dataset(DatasetConfig.test_data_path)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

if __name__ == "__main__":
    try:
        train_ds = CVC_FP_dataset(DatasetConfig.train_data_path)
        img, msk = train_ds[0] 
        
        print(f"Final Image shape: {img.shape}") # Verified: [3, 224, 224]
        
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.title("Otsu (Broadcasted to 3ch)")
        # For plotting, permute back to [H, W, 3]
        plt.imshow(img.permute(1, 2, 0).numpy())
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.title("Ground Truth")
        plt.imshow(msk.numpy()) 
        plt.axis('off')
        
        plt.show()

    except Exception as e:
        print(f"Error: {e}")