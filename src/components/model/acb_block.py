import torch
import torch.nn as nn
import torch.nn.init as init

class ACBBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 padding_mode='zeros',
                 deploy=False,
                 use_affine=True,
                 reduce_gamma=False,
                 gamma_init=None
                 ):
        
        super(ACBBlock,self).__init__()
        self.deploy=deploy
        if deploy:
            self.fused_conv=nn.Conv2d(in_channels=in_channels,
                                      out_channels=out_channels,
                                      kernel_size=(kernel_size,kernel_size),
                                      stride=stride,
                                      padding=padding,
                                      dilation=dilation,
                                      groups=groups,
                                      bias=True,
                                      padding_mode=padding_mode
                                      )
        else:
            self.sqare_conv=nn.Conv2d(in_channels=in_channels,
                                      out_channels=out_channels,
                                      kernel_size=(kernel_size,kernel_size),
                                      stride=stride,
                                      padding=padding,
                                      dilation=dilation,
                                      groups=groups,
                                      bias=False,
                                      padding_mode=padding_mode
                                      )
            self.sqare_bn=nn.BatchNorm2d(num_features=out_channels,affine=use_affine)

            if padding-kernel_size//2>=0:
                self.crop=0
                hor_padding=[padding-kernel_size//2,padding]
                var_padding=[padding,padding-kernel_size//2]
            
            else:
                self.crop=kernel_size//2-padding
                hor_padding=[0,padding]
                ver_padding=[padding,0]

            self.ver_conv=nn.Conv2d(in_channels=in_channels,
                                    out_channels=out_channels,
                                    kernel_size=(kernel_size,1),
                                    stride=stride,
                                    padding=var_padding,
                                    dilation=dilation,
                                    groups=groups,
                                    bias=False,
                                    padding_mode=padding_mode
                                    )

            self.hor_conv=nn.Conv2d(in_channels=in_channels,
                                    out_channels=out_channels,
                                    kernel_size=(1,kernel_size),
                                    stride=stride,
                                    padding=hor_padding,
                                    dilation=dilation,
                                    groups=groups,
                                    bias=False,
                                    padding_mode=padding_mode
                                    )
            
            self.ver_bn=nn.BatchNorm2d(num_features=out_channels,affine=use_affine)
            self.hor_bn=nn.BatchNorm2d(num_features=out_channels,affine=use_affine)

            if reduce_gamma:
                self.init_gamma(1.0/3)
            
            if gamma_init is not None:
                assert not reduce_gamma
                self.init_gamma(gamma_init)



    def init_gamma(self,value):
        init.constant(self.sqare_bn.weight,value)
        init.constant(self.hor_bn.weight,value)
        init.constant(self.ver_bn.weight,value)
    
    def _add_to_sqare_kernel(self,sqare_kernel,asym_kernel):
        asym_h=asym_kernel.size(2)
        asym_w=asym_kernel.size(3)

        sqare_h=sqare_kernel.size(2)
        sqare_w=sqare_kernel.size(3)

        shift_h=sqare_h//2-asym_h//2
        shift_w=sqare_w//2-asym_w//2

        #out_channel,in_channel,h,w
        sqare_kernel[:,:,shift_h:shift_h+asym_h]

