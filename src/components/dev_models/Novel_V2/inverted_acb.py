from src.components.dev_models.Novel_V2.acb_block import ACBBlock
import torch.nn as nn

class InvertedACB(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, expand_ratio=4):
        super(InvertedACB, self).__init__()
        self.stride = stride
        hidden_dim = int(in_channels * expand_ratio)
        self.use_res_connect = self.stride == 1 and in_channels == out_channels

        layers = []
        if expand_ratio != 1:
            # 1. Pointwise Expansion
            layers.append(nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False))
            layers.append(nn.BatchNorm2d(hidden_dim))
            layers.append(nn.ReLU6(inplace=True))


        layers.append(ACBBlock(hidden_dim, hidden_dim, kernel_size=3, stride=stride, 
                               padding=1, groups=hidden_dim, deploy=False))
        layers.append(nn.ReLU6(inplace=True))

        layers.append(nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)