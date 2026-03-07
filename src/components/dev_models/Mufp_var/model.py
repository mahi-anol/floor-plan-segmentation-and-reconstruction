import torch
import torch.nn as nn
import torch.nn.functional as F
from src.components.dev_models.Novel_v1.acb_block import ACBBlock as ACBlock


def ChannelAvgPool(x):
    return x.mean(dim=1, keepdim=True)

def ChannelMaxPool(x):
    return x.max(dim=1, keepdim=True)[0]


class SAMBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1, bias=True)
        self.extract_net1 = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=1, bias=True),
            nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=True)
        )
        self.extract_net2 = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=1, bias=True),
            nn.Conv2d(1, 1, kernel_size=3, padding=2, dilation=2, bias=True)
        )
        self.extract_net3 = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=1, bias=True),
            nn.Conv2d(1, 1, kernel_size=3, padding=3, dilation=3, bias=True)
        )

    def forward(self, x):
        reduce = self.conv1(x)
        x1 = ChannelAvgPool(x)
        x2 = ChannelMaxPool(x)
        out = torch.cat([x1, x2], dim=1)
        ext1 = self.extract_net1(out)
        ext2 = self.extract_net2(out)
        ext3 = self.extract_net3(out)
        out = ext1 + ext2 + ext3
        out = torch.sigmoid(out)   # F.sigmoid is deprecated
        return reduce * out


class CAMBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 2,  kernel_size=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels // 2, in_channels // 32, kernel_size=1, bias=True)
        self.conv3 = nn.Conv2d(in_channels // 2, in_channels // 32, kernel_size=1, bias=True)
        self.conv4 = nn.Conv2d(in_channels // 32, in_channels // 2, kernel_size=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels // 32, in_channels // 2, kernel_size=1, bias=True)

    def forward(self, x):
        x = self.conv1(x)
        max_out = F.adaptive_max_pool2d(x, (1, 1))
        avg_out = F.adaptive_avg_pool2d(x, (1, 1))
        x1 = self.conv4(self.conv2(max_out))
        x2 = self.conv5(self.conv3(avg_out))
        out = torch.sigmoid(x1 + x2)  # F.sigmoid is deprecated
        return x * out


class Modified_Attention_Block(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.CAM = CAMBlock(in_channels=in_channels)
        self.SAM = SAMBlock(in_channels=in_channels)
        self.ACB = ACBlock(in_channels=in_channels, out_channels=(in_channels // 5) * 2)

    def forward(self, x):
        x1 = self.CAM(x)
        x2 = self.SAM(x)
        out = torch.cat([x1, x2], dim=1)
        return self.ACB(out)



def _upsample_to(src: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if src.shape[2:] != target.shape[2:]:
        src = F.interpolate(src, size=target.shape[2:], mode='bilinear', align_corners=False)
    return src


class MULTI_UNIT_FLOOR_SEGMENT_MODEL(nn.Module):
    def __init__(self, image_channel, class_num):
        super().__init__()
        self.class_num = class_num
        channels = [16, 32, 64, 128, 256]


        self.conv1 = nn.Sequential(
            ACBlock(image_channel,  channels[0]),
            ACBlock(channels[0],    channels[0]),
            ACBlock(channels[0],    channels[0]),
            ACBlock(channels[0],    channels[0]),
        )
        self.conv12 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            ACBlock(channels[0], channels[1]),
            ACBlock(channels[1], channels[1]),
            ACBlock(channels[1], channels[1]),
            ACBlock(channels[1], channels[1]),
        )
        self.conv13 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            ACBlock(channels[1], channels[2]),
            ACBlock(channels[2], channels[2]),
            ACBlock(channels[2], channels[2]),
            ACBlock(channels[2], channels[2]),
        )
        self.conv14 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            ACBlock(channels[2], channels[3]),
            ACBlock(channels[3], channels[3]),
            ACBlock(channels[3], channels[3]),
            ACBlock(channels[3], channels[3]),
        )

        self.conv2 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            ACBlock(channels[0], channels[1]),
            ACBlock(channels[1], channels[1]),
            ACBlock(channels[1], channels[1]),
            ACBlock(channels[1], channels[1]),
        )
        self.conv23 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            ACBlock(channels[1], channels[2]),
            ACBlock(channels[2], channels[2]),
            ACBlock(channels[2], channels[2]),
            ACBlock(channels[2], channels[2]),
        )
        self.conv24 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            ACBlock(channels[2], channels[3]),
            ACBlock(channels[3], channels[3]),
            ACBlock(channels[3], channels[3]),
            ACBlock(channels[3], channels[3]),
        )

        self.conv3 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            ACBlock(channels[1], channels[2]),
            ACBlock(channels[2], channels[2]),
            ACBlock(channels[2], channels[2]),
            ACBlock(channels[2], channels[2]),
        )
        self.conv34 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            ACBlock(channels[2], channels[3]),
            ACBlock(channels[3], channels[3]),
            ACBlock(channels[3], channels[3]),
            ACBlock(channels[3], channels[3]),
        )

        self.conv4 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            ACBlock(channels[2], channels[3]),
            ACBlock(channels[3], channels[3]),
            ACBlock(channels[3], channels[3]),
            ACBlock(channels[3], channels[3]),
        )

        self.conv5 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            ACBlock(channels[3], channels[4]),
            ACBlock(channels[4], channels[4]),
            ACBlock(channels[4], channels[4]),
            ACBlock(channels[4], channels[4]),
        )

        self.skblock4 = Modified_Attention_Block(channels[3] * 5)
        self.skblock3 = Modified_Attention_Block(channels[2] * 5)
        self.skblock2 = Modified_Attention_Block(channels[1] * 5)
        self.skblock1 = Modified_Attention_Block(channels[0] * 5)

        # ── Decoder upsampling (ConvTranspose2d kept; size fixed via _upsample_to) ──
        self.deconv4  = nn.ConvTranspose2d(channels[4], channels[3], kernel_size=2, stride=2)
        self.deconv43 = nn.ConvTranspose2d(channels[3], channels[2], kernel_size=2, stride=2)
        self.deconv42 = nn.ConvTranspose2d(channels[2], channels[1], kernel_size=2, stride=2)
        self.deconv41 = nn.ConvTranspose2d(channels[1], channels[0], kernel_size=2, stride=2)

        self.conv6 = nn.Sequential(
            ACBlock(channels[4], channels[3]),
            ACBlock(channels[3], channels[3]),
        )

        self.deconv3  = nn.ConvTranspose2d(channels[3], channels[2], kernel_size=2, stride=2)
        self.deconv32 = nn.ConvTranspose2d(channels[2], channels[1], kernel_size=2, stride=2)
        self.deconv31 = nn.ConvTranspose2d(channels[1], channels[0], kernel_size=2, stride=2)

        self.conv7 = nn.Sequential(
            ACBlock(channels[3], channels[2]),
            ACBlock(channels[2], channels[2]),
        )

        self.deconv2  = nn.ConvTranspose2d(channels[2], channels[1], kernel_size=2, stride=2)
        self.deconv21 = nn.ConvTranspose2d(channels[1], channels[0], kernel_size=2, stride=2)

        self.conv8 = nn.Sequential(
            ACBlock(channels[2], channels[1]),
            ACBlock(channels[1], channels[1]),
        )

        self.deconv1 = nn.ConvTranspose2d(channels[1], channels[0], kernel_size=2, stride=2)

        self.conv9 = nn.Sequential(
            ACBlock(channels[1], channels[0]),
            ACBlock(channels[0], channels[0]),
        )

        self.conv10 = nn.Conv2d(channels[0], self.class_num, kernel_size=1)

    def forward(self, x):
       
        conv1  = self.conv1(x)
        conv12 = self.conv12(conv1)
        conv13 = self.conv13(conv12)
        conv14 = self.conv14(conv13)

        conv2  = self.conv2(conv1)
        conv23 = self.conv23(conv2)
        conv24 = self.conv24(conv23)

        conv3  = self.conv3(conv2)
        conv34 = self.conv34(conv3)

        conv4  = self.conv4(conv3)
        conv5  = self.conv5(conv4)

       
        deconv4  = self.deconv4(conv5)
        deconv43 = self.deconv43(deconv4)
        deconv42 = self.deconv42(deconv43)
        deconv41 = self.deconv41(deconv42)

        # Align all branches to deconv4's spatial size before cat
        conv6 = torch.cat([
            deconv4,
            _upsample_to(conv4,  deconv4),
            _upsample_to(conv34, deconv4),
            _upsample_to(conv24, deconv4),
            _upsample_to(conv14, deconv4),
        ], dim=1)
        conv6 = self.skblock4(conv6)
        conv6 = self.conv6(conv6)

      
        deconv3  = self.deconv3(conv6)
        deconv32 = self.deconv32(deconv3)
        deconv31 = self.deconv31(deconv32)

        conv7 = torch.cat([
            deconv3,
            _upsample_to(deconv43, deconv3),
            _upsample_to(conv3,    deconv3),
            _upsample_to(conv23,   deconv3),
            _upsample_to(conv13,   deconv3),
        ], dim=1)
        conv7 = self.skblock3(conv7)
        conv7 = self.conv7(conv7)

       
        deconv2  = self.deconv2(conv7)
        deconv21 = self.deconv21(deconv2)

        conv8 = torch.cat([
            deconv2,
            _upsample_to(deconv42, deconv2),
            _upsample_to(deconv32, deconv2),
            _upsample_to(conv2,    deconv2),
            _upsample_to(conv12,   deconv2),
        ], dim=1)
        conv8 = self.skblock2(conv8)
        conv8 = self.conv8(conv8)

        
        deconv1 = self.deconv1(conv8)

        conv9 = torch.cat([
            deconv1,
            _upsample_to(deconv41, deconv1),
            _upsample_to(deconv31, deconv1),
            _upsample_to(deconv21, deconv1),
            _upsample_to(conv1,    deconv1),
        ], dim=1)
        conv9 = self.skblock1(conv9)
        conv9 = self.conv9(conv9)

 
        output = self.conv10(conv9)

        # Ensure output matches input spatial size exactly
        if output.shape[2:] != x.shape[2:]:
            output = F.interpolate(output, size=x.shape[2:], mode='bilinear', align_corners=False)

        return output


def get_model(image_channel=3, number_of_class=10):
    return MULTI_UNIT_FLOOR_SEGMENT_MODEL(image_channel, number_of_class)


if __name__ == '__main__':
    model = MULTI_UNIT_FLOOR_SEGMENT_MODEL(3, 2)
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Parameters: {pytorch_total_params}")

    # Test variable sizes
    for h, w in [(224, 224), (300, 200), (128, 96), (513, 481)]:
        x = torch.randn(1, 3, h, w)
        y = model(x)
        assert y.shape == (1, 2, h, w), f"Shape mismatch: {y.shape}"
        print(f"Input {x.shape} → Output {y.shape} ✓")