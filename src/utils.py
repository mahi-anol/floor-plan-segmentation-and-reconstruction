import pickle

def load_pickle(path:str):
    with open(file=path,mode='rb') as file:
        pickle_file=pickle.load(file)
    return pickle_file




from src.config import baseline_model_config,best_grid_searched_coefficient,tpu_friendly_efficient_resolutions
import math
from functools import partial
from torch import nn
from torch.nn import functional as F
import torch
import os
from src.logger import logging
from src.exception import CustomException
import sys

class model_configs:
    @classmethod
    def fixing_width(cls,beta,value,divisor=8):
        scaled=beta*value
        
        min_depth=value or divisor
        fixed_width=max(min_depth,int((scaled+divisor/2)//divisor * divisor))
        if fixed_width<0.9*scaled:
            fixed_width+=divisor
        # print(fixed_width)
        return int(fixed_width)

    @classmethod
    def fixing_depth(cls,alpha,value):
        scaled=alpha*value
        fixed_depth=math.ceil(scaled)
        return int(fixed_depth)

    @classmethod
    def get_varient_configs(cls,varient_type:str='efficient_b0'):

        valid_varient_types=['efficient_b0','efficient_b1','efficient_b2','efficient_b3','efficient_b4','efficient_b5','efficient_b6','efficient_b7']

        if varient_type not in valid_varient_types:
            raise ValueError(f"type: {varient_type} is not valid, choose from {valid_varient_types}")

        config_index=valid_varient_types.index(varient_type)

        input_resolution=tpu_friendly_efficient_resolutions[config_index]
        alpha=best_grid_searched_coefficient.alpha[config_index]
        beta=best_grid_searched_coefficient.beta[config_index]

        channel_configs=[cls.fixing_width(beta=beta,value=v2,divisor=8) for k,v in baseline_model_config.items() for k2,v2 in v.items() if k2=='c'] # width==beta
        depth_configs=[cls.fixing_depth(alpha=alpha,value=v2) for k,v in baseline_model_config.items() for k2,v2 in v.items() if k2=='l'] # depth =alpha
        layer_wise_resolution_configs=[v2 for k,v in baseline_model_config.items() for k2,v2 in v.items() if k2=='r']

        stride_configs=[]
        for i in range(1,len(layer_wise_resolution_configs)):
            stride_configs.append(layer_wise_resolution_configs[i-1][0]//layer_wise_resolution_configs[i][0])

        if __name__=="__main__":
            print(input_resolution)
            print(channel_configs)
            print(depth_configs)
            print(stride_configs)
        return input_resolution,channel_configs,depth_configs,stride_configs



class model_helpers:
    @staticmethod
    def get_output_image_size(input_image_size:tuple|int=(224,224),stride:tuple|int=1):
        # h and w for input image
        ih,iw=(input_image_size,input_image_size) if isinstance(input_image_size,int) else input_image_size
        sh,sw=stride if isinstance(stride,tuple) else (stride,stride)
        image_height=int(math.ceil(ih/sh))
        image_width=int(math.ceil(iw/sw))
        return image_height,image_width
    
    @staticmethod
    def same_padded_conv2D(image_size=None):
        return model_helpers.dynamically_same_padded_conv2D if image_size is None else partial(model_helpers.statically_same_padded_conv2D,image_size=image_size)
    

    class dynamically_same_padded_conv2D(nn.Conv2d):
        def __init__(self,in_channels,out_channels,kernel_size,stride=1,dilation=1,groups=1,bias=True):
            super().__init__(in_channels=in_channels
                             ,out_channels=out_channels
                             ,kernel_size=kernel_size
                             ,stride=stride
                             ,dilation=dilation
                             ,groups=groups
                             ,bias=bias
                            )
        def forward(self, inputs):
            ih,iw=inputs.size()[-2:] # b,c,h,w -> h,w
            kh,kw=self.weight.size()[-2:] #filter,channel,h,w
            sh,sw=self.stride
            oh,ow=model_helpers.get_output_image_size((ih,iw),(sh,sw))
            pad_h=max((oh-1)*sh+(kh-1)*self.dilation[0]+1-ih,0)
            pad_w=max((ow-1)*sw+(kw-1)*self.dilation[1]+1-iw,0)
            
            if pad_h>0 or pad_w>0:
                inputs=F.pad(inputs,[pad_w//2,pad_w-pad_w//2,pad_h//2,pad_h-pad_h//2])
            return F.conv2d(input=inputs
                            ,weight=self.weight
                            ,bias=self.bias
                            ,stride=self.stride
                            ,padding=self.padding
                            ,dilation=self.dilation
                            ,groups=self.groups
                            )
            


    class statically_same_padded_conv2D(nn.Conv2d):
        def __init__(self,in_channels,out_channels,kernel_size,stride=1,image_size=None,**kwargs):
            super().__init__(in_channels=in_channels
                             ,out_channels=out_channels
                             ,kernel_size=kernel_size
                             ,stride=stride
                             ,**kwargs
                            )

            assert image_size is not None

            ih,iw=model_helpers.get_output_image_size(image_size) # b,c,h,w -> h,w
            kh,kw=self.weight.size()[-2:] #filter,channel,h,w
            sh,sw=self.stride
            oh,ow=model_helpers.get_output_image_size((ih,iw),(sh,sw))
            pad_h=max((oh-1)*sh+(kh-1)*self.dilation[0]+1-ih,0)
            pad_w=max((ow-1)*sw+(kw-1)*self.dilation[1]+1-iw,0)
            if  pad_h>0 or pad_w>0:
                self.static_padding=nn.ZeroPad2d((pad_w//2,pad_w-pad_w//2
                                                  ,pad_h//2,pad_h-pad_h//2))
            else:
                self.static_padding=nn.Identity()
        
        def forward(self,inputs):
            inputs=self.static_padding(inputs)
            return  F.conv2d(input=inputs
                            ,weight=self.weight
                            ,bias=self.bias
                            ,stride=self.stride
                            ,padding=self.padding
                            ,dilation=self.dilation
                            ,groups=self.groups
                            )
            
    @staticmethod
    def same_padded_maxpool2D(image_size=None):
        return model_helpers.dynamically_same_padded_maxpool2D if image_size is None else partial(model_helpers.statically_same_padded_maxpool2D,image_size=image_size)
        
    
    class dynamically_same_padded_maxpool2D(nn.MaxPool2d):

        def __init__(self,kernel_size,stride,padding=0,dilation=1,return_indices=False,ceil_mode=False):
            super().__init__(kernel_size=kernel_size,stride=stride,padding=padding,dilation=dilation,return_indices=return_indices,ceil_mode=ceil_mode)


        def forward(self,inputs):
            ih,iw=inputs.size()[-2:]
            kh,kw=self.kernel_size
            sh,sw=self.stride
            oh,ow=model_helpers.get_output_image_size((ih,iw),(sh,sw))
            pad_h=max((oh-1)*sh+(kh-1)*self.dilation+1-ih,0)
            pad_w=max((ow-1)*sw+(kw-1)*self.dilation+1-iw,0)
            
            if pad_h>0 or pad_w>0:
                inputs=F.pad(inputs,[pad_w//2,pad_w-pad_w//2,pad_h//2,pad_h-pad_h//2])

            return F.max_pool2d(inputs
                                ,self.kernel_size
                                ,self.stride
                                ,self.padding
                                ,self.dilation
                                ,self.ceil_mode
                                ,self.return_indices
                                )

    class statically_same_padded_maxpool2D(nn.MaxPool2d):
        def __init__(self,kernel_size,stride,image_size=None,**kwargs):
            super().__init__(kernel_size,stride,**kwargs)
            assert image_size is not None
            ih,iw=(image_size,image_size) if isinstance(image_size,int) else image_size
            kh,kw=self.kernel_size
            sh,sw=self.stride
            oh,ow=model_helpers.get_output_image_size((ih,iw),(sh,sw))
            pad_h=max((oh-1)*sh+(kh-1)*self.dilation+1-ih,0)
            pad_w=max((ow-1)*sw+(kw-1)*self.dilation+1-iw,0)

            if  pad_h>0 or pad_w>0:
                self.static_padding=nn.ZeroPad2d((pad_w//2,pad_w-pad_w//2
                                                  ,pad_h//2,pad_h-pad_h//2))
            else:
                self.static_padding=nn.Identity()

        def forward(self,inputs):
            inputs=self.static_padding(inputs)
            return F.max_pool2d(inputs
                                ,self.kernel_size
                                ,self.stride
                                ,self.padding
                                ,self.dilation
                                ,self.ceil_mode
                                ,self.return_indices
                                )
    @staticmethod
    def drop_connect(inputs:torch.Tensor,p,training):
        """
        Drop connect.

        Args:
            inputs: tensor.shape=(B,C,H,W).
            p (float: 0.0~1.0): Probability of drop connection.
            training (bool): The running mode.

        Returns:
            output: Output after drop connection.
        """

        if not training:
            return inputs

        batch_size=inputs.shape[0]
        keep_prob=1-p

        random_tensor=keep_prob
        random_tensor+=torch.rand([batch_size,1,1,1],dtype=inputs.dtype,device=inputs.device)
        binary_tensor=torch.floor(random_tensor)

        output=inputs/keep_prob*binary_tensor

        return output
    


def saving_model_with_state_and_logs(model,optimizer,results,file="model.pt"):
    try:
        os.makedirs('checkpoints',exist_ok=True)
        path=os.path.join('checkpoints',file)
        contents={
            'model':model.state_dict(),
            'optimizer':optimizer.state_dict(),
            'history':results,
        }
        torch.save(contents,path)
    except Exception as e:
        logging.error("There was an unexpect error during saving the model artifacts")
        raise CustomException(e,sys)
    else:
        logging.info(f"Succesfully saved the model artifact at {path}")



if __name__=="__main__":

    ## Testing configs.
    configs=model_configs()
    configs.get_varient_configs('efficient_b0')

    ### Testing Custom Layers.
    # Custom Layer object
    model_helpers=model_helpers()

    # Input
    inputs=torch.randn((1,3,224,224))
    print('Input: ',inputs.shape)

    # Testing get_model_output_size
    output=model_helpers.get_output_image_size(input_image_size=(224,224),stride=(2,2))
    print('get_output_size with stride 2: ',output)

    # Test dynamic layer
    dynamic_conv2d=model_helpers.dynamically_same_padded_conv2D(in_channels=3,out_channels=12,kernel_size=(3,3),stride=2,dilation=1,groups=1,bias=False)
    output=dynamic_conv2d(inputs)
    print('dynamic conv2d: ',output.shape)

    # Test static layer
    static_conv2d=model_helpers.statically_same_padded_conv2D(in_channels=3,out_channels=12,kernel_size=(3,3),stride=(2,2),image_size=(224,224))
    output=static_conv2d(inputs)
    print('static conv2d:',output.shape)

    # Testing same padded layer
    same_padded_conv2d=model_helpers.same_padded_conv2D((224,224))
    conv2d=same_padded_conv2d(in_channels=3,out_channels=12,kernel_size=(3,3),stride=(2,2),dilation=1,groups=1,bias=False)
    print('testing same padded: ',output.shape)
    
    same_padded_conv2d=model_helpers.same_padded_conv2D()
    conv2d=same_padded_conv2d(in_channels=3,out_channels=12,kernel_size=(3,3),stride=(2,2),dilation=1,groups=1,bias=False)
    print('testing same padded: ',output.shape)

    # Testing Maxpool 2D.
    same_padded_max_pool=model_helpers.same_padded_maxpool2D(image_size=(224,224))
    max_pool_2D=same_padded_max_pool(kernel_size=(3,3),stride=(2,2))
    output=max_pool_2D(inputs)
    print('Max pool static: ',output.shape)

    # Testing Maxpool 2D.
    same_padded_max_pool=model_helpers.same_padded_maxpool2D()
    max_pool_2D=same_padded_max_pool(kernel_size=(3,3),stride=(2,2))
    output=max_pool_2D(inputs)
    print('max pool dynamic: ',output.shape)
