import torch.nn as nn
from torchvision import models
from torchsummary import summary

class EfficientB2Unet(nn.Module):
    def __init__(self,n_classes=4,pretrained=True):
        super().__init__()
        self.efficientnet=models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.IMAGENET1K_V1 if pretrained else None)

    def see_model_summary(self):
        print(summary(self.efficientnet,(3,260,260)))



if __name__ == '__main__':
    model = EfficientB2Unet().see_model_summary()