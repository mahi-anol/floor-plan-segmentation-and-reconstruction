import torch
import torch.nn as nn

import torch.nn.functional as F

def ChannelAvgPool(x):
    return x.mean(dim=1,keepdim=True) 
    
def ChannelMaxPool(x):
    return x.max(dim=1,keepdim=True)

class SAMBlock(nn.Module):
    def __init__(self,in_channels):
        super().__init__()
        self.conv1=nn.Conv2d(in_channels=in_channels,out_channels=in_channels//2,kernel_size=(1,1),bias=True)
        self.extract_net=nn.Sequential(
            nn.Conv2d(in_channels=in_channels,out_channels=1,kernel_size=(1,1),bias=True)
            ,nn.Conv2d(in_channels=1,out_channels=1,kernel_size=(3,3),padding=1,bias=True)
            ,nn.Conv2d(in_channels=1,out_channels=1,kernel_size=(1,1),padding=1,bias=True)
            ,nn.Conv2d(in_channels=1,out_channels=1,kernel_size=(3,3),padding=2,dilation=2,bias=True)
            ,nn.Conv2d(in_channels=1,out_channels=1,kernel_size=(1,1),bias=True)
            ,nn.Conv2d(in_channels=1,out_channels=1,kernel_size=(3,3),padding=3,dilation=3,bias=True)
        )

    def forward(self,x):
        x=self.conv1(x)
        x1=ChannelAvgPool(x)
        x2=ChannelMaxPool(x)
        out=torch.cat([x1,x2],dim=1)
        out=out+self.extract_net(out)
        out=F.sigmoid(out)
        out=x*out
        return out






class CAMBlock(nn.Module):
    def __init__(self,in_channels):
        # output ch = input ch/2
        super().__init__()
        self.conv1=nn.Conv2d(in_channels=in_channels,out_channels=in_channels//2,kernel_size=(1,1),bias=True)

        self.conv2=nn.Conv2d(in_channels=in_channels//2,out_channels=in_channels//32,kernel_size=(1,1),bias=True)
        self.conv3=nn.Conv2d(in_channels=in_channels//2,out_channels=in_channels//32,kernel_size=(1,1),bias=True)

        self.conv4=nn.Conv2d(in_channels=in_channels//32,out_channels=in_channels//2,kernel_size=(1,1),bias=True)
        self.conv5=nn.Conv2d(in_channels=in_channels//32,out_channels=in_channels//2,kernel_size=(1,1),bias=True)

    def forward(self,x):
        x=self.conv1(x)

        max_out=F.adaptive_max_pool2d(x,(1,1))
        avg_out=F.adaptive_avg_pool2d(x,(1,1))

        x1=self.conv2(max_out)
        x2=self.conv3(avg_out)

        x1=self.conv4(x1)
        x2=self.conv5(x2)

        out=x1+x2
        out=F.sigmoid(out)

        out=x*out

        return out
        

class Modified_Attention_Block(nn.Module):
    def __init__(self,in_channels):
        super().__init__()
        self.CAM=CAMBlock(in_channels=in_channels)
        self.SAM=SAMBlock(in_channels=in_channels)

    def forward(self,x):
        x1=self.CAM(x)
        x2=self.SAM(x)
        out=torch.cat([x1,x2],dim=1)
        

