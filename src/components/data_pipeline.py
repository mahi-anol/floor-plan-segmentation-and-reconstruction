import torch
from torch.utils.data import DataLoader,Dataset
from torchvision.datasets import CIFAR10
from dataclasses import dataclass
from torchvision import transforms
@dataclass
class DataConfig:
    data_dir=r"E:\Architecture-implementation-from-research-papers\Data\image-classification"

train_dataset=CIFAR10(root=DataConfig.data_dir,download=True,train=True,transform=transforms.ToTensor())
test_dataset=CIFAR10(root=DataConfig.data_dir,download=True,train=False,transform=transforms.ToTensor())

print("Train Dataset size: %s, Test Dataset size: %s"%(len(train_dataset),len(test_dataset)))

def get_train_test_loader():
    train_data_loader=DataLoader(dataset=train_dataset,batch_size=16,shuffle=True,num_workers=0)
    test_data_loader=DataLoader(dataset=test_dataset,batch_size=16,shuffle=False,num_workers=0)
    return train_data_loader,test_data_loader
