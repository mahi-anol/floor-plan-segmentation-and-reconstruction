import torch
import torch.nn as nn

class ACBlock(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(ACBlock,self).__init__()
        self.sqare=nn.Conv2d(in_channels=)