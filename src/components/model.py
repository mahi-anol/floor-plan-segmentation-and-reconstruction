import torch.nn as nn
from dataclasses import dataclass
from torch.nn import functional as F
import torch
from src.config import baseline_model_config,kernel_configs
from src.utils import model_helpers,model_configs
from typing import OrderedDict
from torchinfo import summary

class MBConvBlock(nn.Module):
    def __init__(self,expansion_ratio=1,re_ratio=0.25,in_channels=1,out_channels=1,input_image_size=None,stride=None,kernel_size=None):
        super().__init__()
        self.expansion_ratio=expansion_ratio
        self.in_channels=in_channels
        self.out_channels=out_channels
        self.stride=stride
        # Expansion Layer
        if expansion_ratio!=1:
            self.expansion_layer=model_helpers.same_padded_conv2D(image_size=input_image_size)(in_channels=in_channels
                                                                                                ,out_channels=in_channels*expansion_ratio
                                                                                                ,kernel_size=1
                                                                                                ,stride=1 # for expansion stride is supposed to be 1.
                                                                                                ,dilation=1
                                                                                                ,groups=1
                                                                                                ,bias=False
                                                                                                ) ## expansion layer.  112x122x32-> 112x112*(32*expans
            self.bn0=nn.BatchNorm2d(num_features=in_channels*expansion_ratio)

        # Depth-wise Layer, thats where spatial dimention go down.
        self.depth_wise_layer=model_helpers.same_padded_conv2D(image_size=input_image_size)(in_channels=in_channels*expansion_ratio
                                                                                            ,out_channels=in_channels*expansion_ratio
                                                                                            ,kernel_size=kernel_size
                                                                                            ,stride=stride
                                                                                            ,dilation=(1,1)
                                                                                            ,groups=in_channels*expansion_ratio
                                                                                            ,bias=False
                                                                                            )# 112,112,(32*expansion Ratio) -> 112,112,(32* expansion Ratio)
        self.bn1=nn.BatchNorm2d(num_features=expansion_ratio*in_channels)

        output_image_size=model_helpers.get_output_image_size(input_image_size=input_image_size,stride=stride)


        # # Sqeeze and Excitation Layer (attention)
        reduced_channels=int(max(1,(in_channels*re_ratio)))
        # print(reduced_channels)
        self.reduce=model_helpers.same_padded_conv2D(image_size=output_image_size)(in_channels=(expansion_ratio*in_channels),out_channels=reduced_channels,kernel_size=(1,1),stride=(1,1),dilation=(1,1),groups=1,bias=True)
        self.expand=model_helpers.same_padded_conv2D(image_size=output_image_size)(in_channels=reduced_channels,out_channels=(expansion_ratio*in_channels),kernel_size=(1,1),stride=(1,1),dilation=(1,1),groups=1,bias=True) # (1,1)

        # pointwise Conv
        self.point_conv=model_helpers.same_padded_conv2D(image_size=output_image_size)(in_channels=expansion_ratio*in_channels,out_channels=out_channels,kernel_size=(1,1),stride=(1,1),dilation=(1,1),groups=1,bias=False)
        self.bn2=nn.BatchNorm2d(num_features=out_channels)

    # Forward method.
    def forward(self,inputs,drop_connect_rate=None):

        x=inputs
        # Expansion Layer
        if self.expansion_ratio>1:
            x=self.expansion_layer(x)
            x=self.bn0(x)
            x=F.silu(x,inplace=False)

        # print('Expansion output: ',x.shape)

        # DepthWise Layer
        x=self.depth_wise_layer(x)  #halfs the spatial resolution
        x=self.bn1(x)
        x=F.silu(x,inplace=False)

        # print('Depthwise output:',x.shape)
        # ### Sqeeze and Excitation layer
        before_pool=x
        x=F.adaptive_avg_pool2d(x,(1,1))
        x=self.reduce(x)
        x=F.silu(x,inplace=False)
        x=self.expand(x)
        x=F.sigmoid(x)*before_pool #(b,c,1,1)*b,c,(h,W)  = b,c,h,w
        # print('Sqeeze and Excitation Output:',x.shape)

        # projection/pointwise conv
        x=self.point_conv(x)
        x=self.bn2(x)
        
        # skip connection
        if self.in_channels==self.out_channels and self.stride==1:
            if drop_connect_rate:
                x=model_helpers.drop_connect(inputs=x,p=drop_connect_rate,training=self.training)
            x+=inputs
        return x
    

class stem_layer(nn.Module):
    def __init__(self,input_resolution=(224,224),in_channels=3,out_channels=32,kernel_size=(3,3),stride=(2,2),dialation=(1,1),groups=1,bias=False):
        super().__init__()
        self.conv3x3=model_helpers.same_padded_conv2D(image_size=input_resolution)(in_channels=in_channels
                                                                                    ,out_channels=out_channels
                                                                                    ,kernel_size=kernel_size
                                                                                    ,stride=stride
                                                                                    ,dilation=dialation
                                                                                    ,groups=groups
                                                                                    ,bias=bias
                                                                                    )
        self.bn=nn.BatchNorm2d(num_features=out_channels)

    def forward(self,x):
        x=self.conv3x3(x)
        x=self.bn(x)
        x=F.silu(x,inplace=False)
        return x  


class final_bootleneck_layer(nn.Module):
    def __init__(self,image_size,in_channels,out_channels,kernel_size,stride,final_classes=10):
        super().__init__()
        self.conv1x1=model_helpers.same_padded_conv2D(image_size=image_size)(in_channels=in_channels
                                                                            ,out_channels=out_channels
                                                                            ,kernel_size=kernel_size
                                                                            ,stride=stride
                                                                            ,dilation=1
                                                                            ,groups=1
                                                                            ,bias=False
                                                                            )
        self.bn=nn.BatchNorm2d(num_features=out_channels)
        
        self.flattend=nn.Linear(in_features=out_channels,out_features=final_classes,bias=True)

    def forward(self,x):
        x=self.conv1x1(x)
        x=self.bn(x)
        x=F.silu(x,inplace=False)
        x=F.adaptive_avg_pool2d(x,1)
        x=torch.flatten(x,1)
        x=self.flattend(x)
        return x

class EfficientNet(nn.Module):
    def __init__(self,input_resolution,output_channel_configs,stage_repeats,stride_configs,final_classes=10):
        super().__init__()

        self.stem_layer=stem_layer(out_channels=output_channel_configs[0])
        output_image_shape= model_helpers.get_output_image_size(input_resolution,stride_configs[0]) # calculates the output shape after stem layer.

        self.mb_conv_layers=nn.Sequential() # will contain MBconv layers of stage 2-8
        # stage 2-8 repeat configs

        repeat_configs=stage_repeats[1:-1]  # First and last block/layer same regard less of scaling.
        # print(repeat_configs)
        for stage,repeat in enumerate(repeat_configs,1):
            for i in range(repeat):
                self.mb_conv_layers.add_module(f'MB_CONV-{stage}-{i}',MBConvBlock(expansion_ratio= 1 if stage==1 else 6
                                                        ,re_ratio=0.25
                                                        ,in_channels=output_channel_configs[stage-1] if i==0 else output_channel_configs[stage]
                                                         ,out_channels=output_channel_configs[stage]
                                                        ,input_image_size=output_image_shape
                                                        ,stride=stride_configs[stage] if i==0 else 1 
                                                        ,kernel_size=kernel_configs[stage]
                                                        )
                                            )
                
                # print('stage: ',stage,'repeat iter: ',i,'stride config: ',stride_configs[stage] if i==0 else 1)
                output_image_shape=model_helpers.get_output_image_size(output_image_shape,stride_configs[stage] if i==0 else 1)


        self.final_BottleNeck=final_bootleneck_layer(image_size=output_image_shape
                                                     ,in_channels=output_channel_configs[-2]
                                                     ,out_channels=output_channel_configs[-1]
                                                     ,kernel_size=1
                                                     ,stride=stride_configs[-1]
                                                     ,final_classes=final_classes
                                                    )        

    def forward(self,inputs):
        x=inputs
        # Stem layer Forward pass
        x=self.stem_layer(inputs)

        # Mbconv Layers Forward Pass
        x=self.mb_conv_layers(x)

        # Final Layer forward pass
        x=self.final_BottleNeck(x)
        return x

def get_model(varient='efficient_b0',final_classes=10):
    input_resolution,output_channel_configs,stage_repeats,stride_configs=model_configs.get_varient_configs(varient)
    model=EfficientNet(input_resolution,output_channel_configs,stage_repeats,stride_configs,final_classes=final_classes)
    return model

if __name__=="__main__":

    inputs=torch.rand((1,3,224,224))
    print('Test input shape: ',inputs.shape)
    ### unit testing

    # Testing mbconv
    print('Test input shape: ',inputs.shape)
    output=MBConvBlock(expansion_ratio=1,re_ratio=0.25,in_channels=3,out_channels=32,input_image_size=tuple(inputs.shape[-2:]),stride=2,kernel_size=(3,3))(inputs)
    print('Mobconv output shape , stride=2: ',output.shape)

    # testing stem layer
    print('Test input shape: ',inputs.shape)
    output=stem_layer(inputs.shape[-2:])(inputs)
    print("stemp layer output shape: ",output.shape)


    # testing final bottleneck layer
    inputs=torch.rand((1,320,7,7))
    print('Test input shape: ',inputs.shape)
    output=final_bootleneck_layer(image_size=(7,7),in_channels=inputs.shape[1],out_channels=1280,kernel_size=(1,1),stride=(1,1),final_classes=10)(inputs)
    print('final bottle neck output shape: ',output.shape)
    

    # integration testing

    input_resolution,output_channel_configs,stage_repeats,stride_configs=model_configs.get_varient_configs('efficient_b0')
    print('input_resolution: ',input_resolution)
    print('output_channel_configs: ',output_channel_configs)
    print('stage_repeats: ',stage_repeats)
    print('stride_configs: ',stride_configs)
    model=EfficientNet(input_resolution,output_channel_configs,stage_repeats,stride_configs,final_classes=10)

    # print(model)
    

    inputs=torch.rand((2,3,224,224))
    print('Test input shape: ',inputs.shape)
    output=model(inputs)
    print('Model output shape: ',output.shape)

    ### show compelte model insights
    summary(model,input_size=(1,3,224,224))
    print(f"Stem Params: {sum(p.numel() for p in model.stem_layer.parameters())}")
    print(f"MBConv Params: {sum(p.numel() for p in model.mb_conv_layers.parameters())}")
    print(f"Final Layer Params: {sum(p.numel() for p in model.final_BottleNeck.parameters())}")