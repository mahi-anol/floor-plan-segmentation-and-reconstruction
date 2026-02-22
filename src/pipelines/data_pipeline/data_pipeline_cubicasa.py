import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from dataclasses import dataclass
from PIL import Image
import os
import matplotlib.pyplot as plt
import numpy as np

@dataclass
class DatasetConfig:
    train_data_path: str = r"E:/floor-plan-segmentation-and-reconstruction/cubicasa_data/train"
    test_data_path: str = r"E:/floor-plan-segmentation-and-reconstruction/cubicasa_data/test"
    image_size: tuple = (224, 224)

class SimpleBinaryDataset(Dataset):
    def __init__(self, root_path):
        super().__init__()
        if not os.path.exists(root_path):
            raise ValueError(f"Directory not found: {root_path}")

        # Each subfolder is a sample
        self.sample_folders = [
            os.path.join(root_path, d) for d in os.listdir(root_path) 
            if os.path.isdir(os.path.join(root_path, d))
        ]

        # Standard Image Transform
        self.img_tf = transforms.Compose([
            transforms.Resize(DatasetConfig.image_size),
            transforms.ToTensor(), # Scales to [0, 1]
        ])
        
        # Mask Transform: Must use NEAREST to keep edges sharp
        self.mask_tf = transforms.Compose([
            transforms.Resize(DatasetConfig.image_size, interpolation=InterpolationMode.NEAREST),
        ])

    def __len__(self):
        return len(self.sample_folders)

    def __getitem__(self, index):
        folder_path = self.sample_folders[index]
        
        img_path = os.path.join(folder_path, "original_image.png")
        mask_path = os.path.join(folder_path, "mask_output.png")

        # 1. Process Image
        image = Image.open(img_path).convert("RGB")
        image = self.img_tf(image)
        
        # 2. Process Mask (L mode = Grayscale 0-255)
        mask = Image.open(mask_path).convert("L")
        mask = self.mask_tf(mask)
        
        # 3. Convert to Binary Class IDs
        # Any pixel > 128 becomes 1 (Wall), else 0 (Background)
        mask_tensor = torch.from_numpy(np.array(mask))
        mask_tensor = (mask_tensor > 128).long() 
        
        return image, mask_tensor

# --- Loaders ---

def get_train_test_loader(batch_size=16):
    train_ds = SimpleBinaryDataset(DatasetConfig.train_data_path)
    test_ds = SimpleBinaryDataset(DatasetConfig.test_data_path)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

if __name__ == "__main__":
    import numpy as np
    try:
        ds = SimpleBinaryDataset(DatasetConfig.train_data_path)
        img, msk = ds[34]
        
        print(f"Image shape: {img.shape}") # [3, 224, 224]
        print(f"Mask shape: {msk.shape}")   # [224, 224]
        print(f"Unique values in mask: {torch.unique(msk)}") # Should be [0, 1]

        # Quick Visualization
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(img.permute(1, 2, 0))
        ax[0].set_title("Image")
        ax[1].imshow(msk, cmap='gray')
        ax[1].set_title("Binary Mask (0/1)")
        plt.show()

    except Exception as e:
        print(f"Error: {e}")