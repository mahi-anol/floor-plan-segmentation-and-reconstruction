import torch
import torch.nn as nn

class ChannelAttention(nn.Module):
    def __init__(self,in_channels,out_channels,se_ratio=16):
        super(ChannelAttention,self).__init__()
        self.conv=nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=(1,1),bias=False)
        self.avg_pool=nn.AdaptiveAvgPool2d(1)
        self.max_pool=nn.AdaptiveMaxPool2d(1)

        self.fc11=nn.Conv2d(in_channels=out_channels,out_channels=int(out_channels//se_ratio),kernel_size=(1,1),bias=False)
        self.fc12=nn.Conv2d(in_channels=int(out_channels//se_ratio),out_channels=out_channels,kernel_size=(1,1),bias=False)


        self.fc21=nn.Conv2d(in_channels=out_channels,out_channels=int(out_channels//se_ratio),kernel_size=(1,1),bias=False)
        self.fc22=nn.Conv2d(in_channels=int(out_channels//se_ratio),out_channels=out_channels,kernel_size=(1,1),bias=False)
        self.relu=nn.ReLU(inplace=False)
        self.sigmoid=nn.Sigmoid()

    
    def forward(self,x):
        x=self.conv(x)
        avg_out=self.fc12(self.relu(self.fc11(self.avg_pool(x))))
        max_out=self.fc22(self.relu(self.fc11(self.max_pool(x))))

        out=avg_out+max_out
        return x*self.sigmoid(out)







class ACBlock(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(ACBlock,self).__init__()
        self.sqare=nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=(3,3),padding=(1,1),stride=(1,1),bias=False) # Original Implementation have bias True
        self.horizontal=nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=(1,3),padding=(0,1),stride=(1,1),bias=False)
        self.vertical=nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=(3,1),padding=(1,0),stride=(1,1),bias=False)
        self.bn=nn.BatchNorm2d(num_features=out_channels)
        self.relu=nn.ReLU()

    def forward(self,x):
        x1=self.sqare(x)
        x2=self.horizontal(x)
        x3=self.vertical(x)
        output=self.relu(self.bn(x1+x2+x3))
        return output
    

class MACUNet(nn.Module):
    def __init__(self,image_channel,class_num):
        super(MACUNet,self).__init__()

        channels=[16,32,64,128,256,512]

        self.conv1=nn.Sequential(
            ACBlock(image_channel,channels[0]),###(B,3,224,224)
            ACBlock(channels[0],channels[0])###(B,16,224,224)
        )   

        self.conv12=nn.Sequential(
            nn.MaxPool2d(kernel_size=(2,2),stride=(2,2)),###(B,16,112,112)
            ACBlock(channels[0],channels[1])###(B,32,112,112)
        )

        self.conv13=nn.Sequential(
            nn.MaxPool2d(kernel_size=(2,2),stride=(2,2)),###(B,32,56,56)
            ACBlock(channels[1],channels[2])###(B,64,56,56)

        )

        self.conv14=nn.Sequential(
            nn.MaxPool2d(kernel_size=(2,2),stride=(2,2)),###(B,64,28,28)
            ACBlock(channels[2],channels[3]) ###(B,128,28,28)
        )


        self.conv2=nn.Sequential(
            nn.MaxPool2d(nn.MaxPool2d(kernel_size=(2,2),stride=(2,2))),###(B,16,112,112)
            ACBlock(channels[0], channels[1]),###(B,32,112,112)
            ACBlock(channels[1], channels[1])###(B,32,112,112)
        )

        self.conv23 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),###(B,32,56,56)
            ACBlock(channels[1], channels[2])###(B,64,56,56)
        )
        self.conv24 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),###(B,64,28,28)
            ACBlock(channels[2], channels[3])###(B,128,28,28)
        )


        self.conv3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)), #32,56,56
            ACBlock(channels[1], channels[2]), # 64,56,56
            ACBlock(channels[2], channels[2]), # 64,56,56
            ACBlock(channels[2], channels[2]) # 64,56,56
        )
        self.conv34 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)), # 64,28,28
            ACBlock(channels[2], channels[3]) # 128,28,28
        )

        self.conv4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)), # 64,28,28
            ACBlock(channels[2], channels[3]), # 128,28,28
            ACBlock(channels[3], channels[3]), # 128,28,28
            ACBlock(channels[3], channels[3]) # 128,28,28
        )

        self.conv5 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)), #128,14,14
            ACBlock(channels[3], channels[4]), # 256,14,14
            ACBlock(channels[4], channels[4]), # 256,14,14
            ACBlock(channels[4], channels[4]) # 256,14,14
        )


        self.skblock4=ChannelAttention(channels[3]*5,channels[3]*2,16)
        self.skblock3 = ChannelAttention(channels[2]*5, channels[2]*2, 16)
        self.skblock2 = ChannelAttention(channels[1]*5, channels[1]*2, 16)
        self.skblock1 = ChannelAttention(channels[0]*5, channels[0]*2, 16)



