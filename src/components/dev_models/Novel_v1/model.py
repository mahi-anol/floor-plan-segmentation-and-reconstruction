import torch
import torch.nn as nn
from src.components.dev_models.Novel_v1.inverted_acb import InvertedACB
from src.components.dev_models.Novel_v1.attention_block import CoordinateAttention 
from src.components.dev_models.Novel_v1.acb_block import ACBBlock

class MULTI_UNIT_FLOOR_SEGMENT_MODEL(nn.Module):
    def __init__(self, image_channel, class_num):
        super(MULTI_UNIT_FLOOR_SEGMENT_MODEL, self).__init__()
        self.class_num = class_num

        c = [16, 32, 64, 128, 256] 
        

        self.conv1 = nn.Sequential(
            ACBBlock(image_channel, c[0]), 
            InvertedACB(c[0], c[0], expand_ratio=2) 
        )
        self.conv12 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            InvertedACB(c[0], c[1], stride=1)
        )
        self.conv2 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            InvertedACB(c[0], c[1], stride=1),
            InvertedACB(c[1], c[1], stride=1)
        )

        self.conv13 = nn.Sequential(nn.MaxPool2d(2, 2), InvertedACB(c[1], c[2]))
        self.conv23 = nn.Sequential(nn.MaxPool2d(2, 2), InvertedACB(c[1], c[2]))
        self.conv3  = nn.Sequential(
            nn.MaxPool2d(2, 2), 
            InvertedACB(c[1], c[2]),
            InvertedACB(c[2], c[2])
        )

        self.conv14 = nn.Sequential(nn.MaxPool2d(2, 2), InvertedACB(c[2], c[3]))
        self.conv24 = nn.Sequential(nn.MaxPool2d(2, 2), InvertedACB(c[2], c[3]))
        self.conv34 = nn.Sequential(nn.MaxPool2d(2, 2), InvertedACB(c[2], c[3]))
        self.conv4  = nn.Sequential(
            nn.MaxPool2d(2, 2),
            InvertedACB(c[2], c[3]),
            InvertedACB(c[3], c[3])
        )

        # Stage 5 (Bottleneck)
        self.conv5 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            InvertedACB(c[3], c[4]),
            InvertedACB(c[4], c[4])
        )

        self.up4 = nn.ConvTranspose2d(c[4], c[3], 2, 2)
        self.up3 = nn.ConvTranspose2d(c[3], c[2], 2, 2)
        self.up2 = nn.ConvTranspose2d(c[2], c[1], 2, 2)
        self.up1 = nn.ConvTranspose2d(c[1], c[0], 2, 2)


        self.up4_3 = nn.ConvTranspose2d(c[3], c[2], 2, 2)
        self.up4_2 = nn.ConvTranspose2d(c[2], c[1], 2, 2)
        self.up4_1 = nn.ConvTranspose2d(c[1], c[0], 2, 2)
        
        self.up3_2 = nn.ConvTranspose2d(c[2], c[1], 2, 2)
        self.up3_1 = nn.ConvTranspose2d(c[1], c[0], 2, 2)
        
        self.up2_1 = nn.ConvTranspose2d(c[1], c[0], 2, 2)

        self.fuse_red_4 = nn.Conv2d(c[3]*5, c[3], 1, bias=False)
        self.fuse_red_3 = nn.Conv2d(c[2]*5, c[2], 1, bias=False)
        self.fuse_red_2 = nn.Conv2d(c[1]*5, c[1], 1, bias=False)
        self.fuse_red_1 = nn.Conv2d(c[0]*5, c[0], 1, bias=False)


        self.att4 = CoordinateAttention(c[3], c[3])
        self.att3 = CoordinateAttention(c[2], c[2])
        self.att2 = CoordinateAttention(c[1], c[1])
        self.att1 = CoordinateAttention(c[0], c[0])

        self.conv6 = nn.Sequential(InvertedACB(c[3], c[3]), InvertedACB(c[3], c[3]))
        self.conv7 = nn.Sequential(InvertedACB(c[2], c[2]), InvertedACB(c[2], c[2]))
        self.conv8 = nn.Sequential(InvertedACB(c[1], c[1]), InvertedACB(c[1], c[1]))
        self.conv9 = nn.Sequential(InvertedACB(c[0], c[0]), InvertedACB(c[0], c[0]))

        self.final = nn.Conv2d(c[0], self.class_num, 1)

    def forward(self, x):
        # Encoder
        c1 = self.conv1(x)
        
        c12 = self.conv12(c1)
        c2  = self.conv2(c1) 
        
        c13 = self.conv13(c12)
        c23 = self.conv23(c2)
        c3  = self.conv3(c2) 
        
        c14 = self.conv14(c13)
        c24 = self.conv24(c23)
        c34 = self.conv34(c3)
        c4  = self.conv4(c3) 
        
        c5  = self.conv5(c4) 

       
        d4 = self.up4(c5)
        
        cat4 = torch.cat((d4, c4, c34, c24, c14), dim=1) 
        cat4 = self.fuse_red_4(cat4) 
        cat4 = self.att4(cat4)       
        out4 = self.conv6(cat4)

     
        d3 = self.up3(out4)
        d4_3 = self.up4_3(d4)
        cat3 = torch.cat((d3, d4_3, c3, c23, c13), dim=1)
        cat3 = self.fuse_red_3(cat3)
        cat3 = self.att3(cat3)
        out3 = self.conv7(cat3)

        
        d2 = self.up2(out3)
        d4_2 = self.up4_2(d4_3)
        d3_2 = self.up3_2(d3)
        cat2 = torch.cat((d2, d4_2, d3_2, c2, c12), dim=1)
        cat2 = self.fuse_red_2(cat2)
        cat2 = self.att2(cat2)
        out2 = self.conv8(cat2)

        
        d1 = self.up1(out2)
        d4_1 = self.up4_1(d4_2)
        d3_1 = self.up3_1(d3_2)
        d2_1 = self.up2_1(d2)
        cat1 = torch.cat((d1, d4_1, d3_1, d2_1, c1), dim=1)
        cat1 = self.fuse_red_1(cat1)
        cat1 = self.att1(cat1)
        out1 = self.conv9(cat1)

        return self.final(out1)

def get_model(image_channel=3,number_of_class=10):
    net = MULTI_UNIT_FLOOR_SEGMENT_MODEL(image_channel, number_of_class)
    return net
    

if __name__ == '__main__':
    model = MULTI_UNIT_FLOOR_SEGMENT_MODEL(3, 2)
    # Count parameters
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Parameters: {pytorch_total_params}")
    
    x = torch.randn(2, 3, 224, 224)
    y = model(x)
    print("Output shape:", y.shape)