import torch
import torch.nn as nn
import torch.nn.functional as F

class ACBlock(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(ACBlock,self).__init__()
        self.sqare=nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=(3,3),padding=(1,1),stride=(1,1),bias=False) 
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
    
def ChannelAvgPool(x):
    return x.mean(dim=1,keepdim=True)
    
def ChannelMaxPool(x):
    return x.max(dim=1,keepdim=True)[0]

class SAMBlock(nn.Module):
    def __init__(self,in_channels):
        super().__init__()
        self.conv1=nn.Conv2d(in_channels=in_channels,out_channels=in_channels//2,kernel_size=(1,1),bias=True)
        self.extract_net1=nn.Sequential(
            nn.Conv2d(in_channels=2,out_channels=1,kernel_size=(1,1),bias=True)
            ,nn.Conv2d(in_channels=1,out_channels=1,kernel_size=(3,3),padding=1,bias=True)
        )
        self.extract_net2=nn.Sequential(
            nn.Conv2d(in_channels=2,out_channels=1,kernel_size=(1,1),bias=True)
            ,nn.Conv2d(in_channels=1,out_channels=1,kernel_size=(3,3),padding=2,dilation=2,bias=True)
        )

        self.extract_net3=nn.Sequential(
            nn.Conv2d(in_channels=2,out_channels=1,kernel_size=(1,1),bias=True)
            ,nn.Conv2d(in_channels=1,out_channels=1,kernel_size=(3,3),padding=3,dilation=3,bias=True)
        )

    def forward(self,x):
        # print(x.shape)
        reduce=self.conv1(x)
        x1=ChannelAvgPool(x)
        x2=ChannelMaxPool(x)
        # print(x1.shape)
        # print(x2.shape)
        out=torch.cat([x1,x2],dim=1)
        ext1=self.extract_net1(out)
        ext2=self.extract_net2(out)
        ext3=self.extract_net3(out)
        out=ext1+ext2+ext3
        out=F.sigmoid(out)
        out=reduce*out
        # print(out.shape)
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
        # print(out.shape)

        return out
        

class Modified_Attention_Block(nn.Module):
    def __init__(self,in_channels):
        super().__init__()
        self.in_channels=in_channels
        self.CAM=CAMBlock(in_channels=in_channels)
        self.SAM=SAMBlock(in_channels=in_channels)
        self.ACB=ACBlock(in_channels=in_channels,out_channels=(in_channels//5)*2)
    def forward(self,x):
        x1=self.CAM(x)
        x2=self.SAM(x)
        out=torch.cat([x1,x2],dim=1)
        out=self.ACB(out)
        # print("Attention Input: ",x.shape)
        # print("Attention Block Output: ",out.shape)
        return out


    

class MULTI_UNIT_FLOOR_SEGMENT_MODEL(nn.Module):
    def __init__(self,image_channel,class_num):
        super(MULTI_UNIT_FLOOR_SEGMENT_MODEL,self).__init__()
        self.class_num=class_num

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
            nn.MaxPool2d(kernel_size=(2,2),stride=(2,2)),###(B,16,112,112)
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

        ### OLD
        # self.skblock4=ChannelAttention(channels[3]*5,channels[3]*2,16)
        # self.skblock3 = ChannelAttention(channels[2]*5, channels[2]*2, 16)
        # self.skblock2 = ChannelAttention(channels[1]*5, channels[1]*2, 16)
        # self.skblock1 = ChannelAttention(channels[0]*5, channels[0]*2, 16)

        ### NEW 
        self.skblock4=Modified_Attention_Block(channels[3]*5)
        self.skblock3 =Modified_Attention_Block(channels[2]*5)
        self.skblock2 = Modified_Attention_Block(channels[1]*5)
        self.skblock1 = Modified_Attention_Block(channels[0]*5)
        


        self.deconv4 = nn.ConvTranspose2d(channels[4], channels[3], kernel_size=(2, 2), stride=(2, 2))
        self.deconv43 = nn.ConvTranspose2d(channels[3], channels[2], kernel_size=(2, 2), stride=(2, 2))
        self.deconv42 = nn.ConvTranspose2d(channels[2], channels[1], kernel_size=(2, 2), stride=(2, 2))
        self.deconv41 = nn.ConvTranspose2d(channels[1], channels[0], kernel_size=(2, 2), stride=(2, 2))

        self.conv6 = nn.Sequential(
            ACBlock(channels[4], channels[3]),
            ACBlock(channels[3], channels[3]),
        )


        self.deconv3 = nn.ConvTranspose2d(channels[3], channels[2], kernel_size=(2, 2), stride=(2, 2))
        self.deconv32 = nn.ConvTranspose2d(channels[2], channels[1], kernel_size=(2, 2), stride=(2, 2))
        self.deconv31 = nn.ConvTranspose2d(channels[1], channels[0], kernel_size=(2, 2), stride=(2, 2))

        self.conv7 = nn.Sequential(
            ACBlock(channels[3], channels[2]),
            ACBlock(channels[2], channels[2]),
        )

        self.deconv2 = nn.ConvTranspose2d(channels[2], channels[1], kernel_size=(2, 2), stride=(2, 2))
        self.deconv21 = nn.ConvTranspose2d(channels[1], channels[0], kernel_size=(2, 2), stride=(2, 2))
        
        self.conv8 = nn.Sequential(
            ACBlock(channels[2], channels[1]),
            ACBlock(channels[1], channels[1])
        )

        self.deconv1 = nn.ConvTranspose2d(channels[1], channels[0], kernel_size=(2, 2), stride=(2, 2))

        self.conv9 = nn.Sequential(
            ACBlock(channels[1], channels[0]),
            ACBlock(channels[0], channels[0])
        )

        self.conv10 = nn.Conv2d(channels[0], self.class_num, kernel_size=1, stride=1)

    def forward(self,x):
        conv1=self.conv1(x)
        conv12=self.conv12(conv1)
        conv13=self.conv13(conv12)
        conv14=self.conv14(conv13)

        conv2=self.conv2(conv1)
        conv23=self.conv23(conv2)
        conv24=self.conv24(conv23)

        conv3=self.conv3(conv2)
        conv34=self.conv34(conv3)
        
        conv4=self.conv4(conv3)

        conv5=self.conv5(conv4) # 256,14,14

        deconv4 = self.deconv4(conv5)
        deconv43 = self.deconv43(deconv4)
        deconv42 = self.deconv42(deconv43)
        deconv41 = self.deconv41(deconv42)


        conv6 = torch.cat((deconv4, conv4, conv34, conv24, conv14), 1)
        conv6 = self.skblock4(conv6)
        conv6 = self.conv6(conv6)



        deconv3 = self.deconv3(conv6)
        deconv32 = self.deconv32(deconv3)
        deconv31 = self.deconv31(deconv32)


        conv7 = torch.cat((deconv3, deconv43, conv3, conv23, conv13), 1)
        conv7 = self.skblock3(conv7)
        conv7 = self.conv7(conv7)


        deconv2 = self.deconv2(conv7)
        deconv21 = self.deconv21(deconv2)


        conv8 = torch.cat((deconv2, deconv42, deconv32, conv2, conv12), 1)
        conv8 = self.skblock2(conv8)
        conv8 = self.conv8(conv8)


        deconv1 = self.deconv1(conv8)
        conv9 = torch.cat((deconv1, deconv41, deconv31, deconv21, conv1), 1)
        conv9 = self.skblock1(conv9)
        conv9 = self.conv9(conv9)

        output = self.conv10(conv9)

        return output


def get_model(image_channel=3,number_of_class=10):
    net = MULTI_UNIT_FLOOR_SEGMENT_MODEL(image_channel, number_of_class)
    return net
    

if __name__ == '__main__':
    classe_num = 10
    in_batch, inchannel, in_h, in_w = 4, 3, 224, 224
    x = torch.randn(in_batch, inchannel, in_h, in_w)
    net = get_model(3,classe_num)
    out = net(x)
    print("Output Shape: ", out.shape)