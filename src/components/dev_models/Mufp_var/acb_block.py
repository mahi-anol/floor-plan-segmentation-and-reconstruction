import torch
import torch.nn as nn
import torch.nn.init as init


class ACBBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 dilation=1,
                 groups=1,
                 padding_mode='zeros',
                 deploy=False,
                 use_affine=True,
                 reduce_gamma=False,
                 gamma_init=None
                 ):

        super(ACBBlock, self).__init__()
        self.deploy = deploy
        if deploy:
            self.fused_conv = nn.Conv2d(in_channels=in_channels,
                                        out_channels=out_channels,
                                        kernel_size=(kernel_size, kernel_size),
                                        stride=stride,
                                        padding=padding,
                                        dilation=dilation,
                                        groups=groups,
                                        bias=True,
                                        padding_mode=padding_mode
                                        )
        else:
            self.square_conv = nn.Conv2d(in_channels=in_channels,
                                         out_channels=out_channels,
                                         kernel_size=(kernel_size, kernel_size),
                                         stride=stride,
                                         padding=padding,
                                         dilation=dilation,
                                         groups=groups,
                                         bias=False,
                                         padding_mode=padding_mode
                                         )

            self.square_bn = nn.BatchNorm2d(num_features=out_channels, affine=use_affine)

            if padding - kernel_size // 2 >= 0:
                self.crop = 0
                hor_padding = [padding - kernel_size // 2, padding]
                ver_padding = [padding, padding - kernel_size // 2]
            else:
                self.crop = kernel_size // 2 - padding
                hor_padding = [0, padding]
                ver_padding = [padding, 0]

            self.ver_conv = nn.Conv2d(in_channels=in_channels,
                                      out_channels=out_channels,
                                      kernel_size=(kernel_size, 1),
                                      stride=stride,
                                      padding=ver_padding,
                                      dilation=dilation,
                                      groups=groups,
                                      bias=False,
                                      padding_mode=padding_mode
                                      )

            self.hor_conv = nn.Conv2d(in_channels=in_channels,
                                      out_channels=out_channels,
                                      kernel_size=(1, kernel_size),
                                      stride=stride,
                                      padding=hor_padding,
                                      dilation=dilation,
                                      groups=groups,
                                      bias=False,
                                      padding_mode=padding_mode
                                      )

            self.ver_bn = nn.BatchNorm2d(num_features=out_channels, affine=use_affine)
            self.hor_bn = nn.BatchNorm2d(num_features=out_channels, affine=use_affine)

            if reduce_gamma:
                self.init_gamma(1.0 / 3)

            if gamma_init is not None:
                assert not reduce_gamma
                self.init_gamma(gamma_init)


    def _fuse_bn_tensor(self, conv: nn.Conv2d, bn: nn.BatchNorm2d):
        std = (bn.running_var + bn.eps).sqrt()
        t = (bn.weight / std).reshape(-1, 1, 1, 1)
        return conv.weight * t, bn.bias - bn.running_mean * bn.weight / std

    def _add_to_square_kernel(self, square_kernel, asym_kernel):
        asym_h = asym_kernel.size(2)
        asym_w = asym_kernel.size(3)
        square_h = square_kernel.size(2)
        square_w = square_kernel.size(3)
        shift_h = square_h // 2 - asym_h // 2
        shift_w = square_w // 2 - asym_w // 2
        square_kernel[:, :, shift_h:shift_h + asym_h, shift_w:shift_w + asym_w] += asym_kernel

    def get_equivalent_kernel_bias(self):
        hor_k, hor_b = self._fuse_bn_tensor(self.hor_conv, self.hor_bn)
        ver_k, ver_b = self._fuse_bn_tensor(self.ver_conv, self.ver_bn)
        square_k, square_b = self._fuse_bn_tensor(self.square_conv, self.square_bn)
        self._add_to_square_kernel(square_k, hor_k)
        self._add_to_square_kernel(square_k, ver_k)
        return square_k, hor_b + ver_b + square_b

    def switch_to_deploy(self):
        deploy_k, deploy_b = self.get_equivalent_kernel_bias()
        self.deploy = True
        self.fused_conv = nn.Conv2d(
            in_channels=self.square_conv.in_channels,
            out_channels=self.square_conv.out_channels,
            kernel_size=self.square_conv.kernel_size,
            stride=self.square_conv.stride,
            padding=self.square_conv.padding,
            dilation=self.square_conv.dilation,
            groups=self.square_conv.groups,
            bias=True,
            padding_mode=self.square_conv.padding_mode
        )
        self.__delattr__('square_conv')
        self.__delattr__('square_bn')
        self.__delattr__('hor_conv')
        self.__delattr__('hor_bn')
        self.__delattr__('ver_conv')
        self.__delattr__('ver_bn')
        self.fused_conv.weight.data = deploy_k
        self.fused_conv.bias.data = deploy_b

    def init_gamma(self, value):
        init.constant_(self.square_bn.weight, value)
        init.constant_(self.hor_bn.weight, value)
        init.constant_(self.ver_bn.weight, value)

    def single_init(self):
        init.constant_(self.square_bn.weight, 1.0)
        init.constant_(self.ver_bn.weight, 0.0)
        init.constant_(self.hor_bn.weight, 0.0)
        print('init gamma of square as 1, ver and hor as 0')


    def forward(self, input):
        if self.deploy:
            return self.fused_conv(input)

        square_outputs = self.square_bn(self.square_conv(input))

        if self.crop > 0:
            ver_input = input[:, :, :, self.crop:-self.crop]
            hor_input = input[:, :, self.crop:-self.crop, :]
        else:
            ver_input = input
            hor_input = input

        vertical_outputs = self.ver_bn(self.ver_conv(ver_input))
        horizontal_outputs = self.hor_bn(self.hor_conv(hor_input))

        return square_outputs + vertical_outputs + horizontal_outputs



if __name__ == '__main__':
    C = 2
    O = 8
    groups = 4

    # Different spatial sizes to prove variable-size support
    variable_sizes = [(62, 62), (128, 96), (256, 180), (64, 200), (37, 53)]

    test_kernel_padding = [(3, 1), (3, 0), (5, 1), (5, 2), (5, 3), (5, 4), (5, 6)]

    print("=" * 60)
    print("Variable-size image test  (batch_size=1)")
    print("=" * 60)

    for H, W in variable_sizes:
        x = torch.randn(1, C, H, W)   # batch_size always 1
        print(f"\nInput: {x.shape}")

        for k, p in test_kernel_padding:
            acb = ACBBlock(C, O, kernel_size=k, padding=p, stride=1, deploy=False)

            # eval() makes BN use running stats → safe with batch_size=1
            acb.eval()

            # Initialise BN running stats to something sensible
            for module in acb.modules():
                if isinstance(module, nn.BatchNorm2d):
                    nn.init.uniform_(module.running_mean, 0, 0.1)
                    nn.init.uniform_(module.running_var, 0, 0.2)
                    nn.init.uniform_(module.weight, 0, 0.3)
                    nn.init.uniform_(module.bias, 0, 0.4)

            out = acb(x)

            acb.switch_to_deploy()
            deploy_out = acb(x)

            max_diff = ((deploy_out - out) ** 2).sum().item()
            print(f"  kernel={k} pad={p} | out={tuple(out.shape)} | "
                  f"fused_diff={max_diff:.2e}")