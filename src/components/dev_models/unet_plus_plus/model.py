import torch
import torch.nn as nn
# from utils import align_outputs


class UNET_PLUS_PLUS(nn.Module):
    def __init__(self,image_channel=3,n_class=2,c_config=[64,128,256,512,1024],deep_supervision=False):
        super(UNET_PLUS_PLUS,self).__init__()

        self.deep_supervision=deep_supervision
        self.m_pool=nn.MaxPool2d(kernel_size=(2,2),stride=(2,2))

        self.x0_0=nn.Sequential(
            nn.Conv2d(in_channels=image_channel,out_channels=c_config[0],kernel_size=(3,3),stride=(1,1),padding=(1,1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=c_config[0],out_channels=c_config[0],kernel_size=(3,3),stride=(1,1),padding=(1,1)),
            nn.ReLU(),
        )
        self.x1_0=nn.Sequential(
            nn.Conv2d(in_channels=c_config[0],out_channels=c_config[1],kernel_size=(3,3),stride=(1,1),padding=(1,1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=c_config[1],out_channels=c_config[1],kernel_size=(3,3),stride=(1,1),padding=(1,1)),
            nn.ReLU(),
        )
        self.x1_0_transpose=nn.ConvTranspose2d(in_channels=c_config[1],out_channels=c_config[0],kernel_size=(2,2),stride=(2,2))

        self.x2_0=nn.Sequential(
            nn.Conv2d(in_channels=c_config[1],out_channels=c_config[2],kernel_size=(3,3),stride=(1,1),padding=(1,1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=c_config[2],out_channels=c_config[2],kernel_size=(3,3),stride=(1,1),padding=(1,1)),
            nn.ReLU(),
        )
        self.x2_0_transpose=nn.ConvTranspose2d(in_channels=c_config[2],out_channels=c_config[1],kernel_size=(2,2),stride=(2,2))

        self.x3_0=nn.Sequential(
            nn.Conv2d(in_channels=c_config[2],out_channels=c_config[3],kernel_size=(3,3),stride=(1,1),padding=(1,1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=c_config[3],out_channels=c_config[3],kernel_size=(3,3),stride=(1,1),padding=(1,1)),
            nn.ReLU(),
        )
        self.x3_0_transpose=nn.ConvTranspose2d(in_channels=c_config[3],out_channels=c_config[2],kernel_size=(2,2),stride=(2,2))

        self.x4_0=nn.Sequential(
            nn.Conv2d(in_channels=c_config[3],out_channels=c_config[4],kernel_size=(3,3),stride=(1,1),padding=(1,1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=c_config[4],out_channels=c_config[4],kernel_size=(3,3),stride=(1,1),padding=(1,1)),
            nn.ReLU(),
        )
        self.x4_0_transpose=nn.ConvTranspose2d(in_channels=c_config[4],out_channels=c_config[3],kernel_size=(2,2),stride=(2,2))


        self.x0_1=nn.Sequential(
            nn.Conv2d(in_channels=c_config[0]*2,out_channels=c_config[0],kernel_size=(3,3),stride=(1,1),padding=(1,1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=c_config[0],out_channels=c_config[0],kernel_size=(3,3),stride=(1,1),padding=(1,1)),
            nn.ReLU(),
        )
        self.x1_1=nn.Sequential(
            nn.Conv2d(in_channels=c_config[1]*2,out_channels=c_config[1],kernel_size=(3,3),stride=(1,1),padding=(1,1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=c_config[1],out_channels=c_config[1],kernel_size=(3,3),stride=(1,1),padding=(1,1)),
            nn.ReLU(),
        )
        self.x1_1_transpose=nn.ConvTranspose2d(in_channels=c_config[1],out_channels=c_config[0],kernel_size=(2,2),stride=(2,2))

        self.x2_1=nn.Sequential(
            nn.Conv2d(in_channels=c_config[2]*2,out_channels=c_config[2],kernel_size=(3,3),stride=(1,1),padding=(1,1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=c_config[2],out_channels=c_config[2],kernel_size=(3,3),stride=(1,1),padding=(1,1)),
            nn.ReLU(),
        )
        self.x2_1_transpose=nn.ConvTranspose2d(in_channels=c_config[2],out_channels=c_config[1],kernel_size=(2,2),stride=(2,2))

        self.x3_1=nn.Sequential(
            nn.Conv2d(in_channels=c_config[3]*2,out_channels=c_config[3],kernel_size=(3,3),stride=(1,1),padding=(1,1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=c_config[3],out_channels=c_config[3],kernel_size=(3,3),stride=(1,1),padding=(1,1)),
            nn.ReLU()
        )
        self.x3_1_transpose=nn.ConvTranspose2d(in_channels=c_config[3],out_channels=c_config[2],kernel_size=(2,2),stride=(2,2))

        self.x0_2=nn.Sequential(
            nn.Conv2d(in_channels=c_config[0]*3,out_channels=c_config[0],kernel_size=(3,3),stride=(1,1),padding=(1,1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=c_config[0],out_channels=c_config[0],kernel_size=(3,3),stride=(1,1),padding=(1,1)),
            nn.ReLU(),
        )
        self.x1_2=nn.Sequential(
            nn.Conv2d(in_channels=c_config[1]*3,out_channels=c_config[1],kernel_size=(3,3),stride=(1,1),padding=(1,1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=c_config[1],out_channels=c_config[1],kernel_size=(3,3),stride=(1,1),padding=(1,1)),
            nn.ReLU(),
        )
        self.x1_2_transpose=nn.ConvTranspose2d(in_channels=c_config[1],out_channels=c_config[0],kernel_size=(2,2),stride=(2,2))

        self.x2_2=nn.Sequential(
            nn.Conv2d(in_channels=c_config[2]*3,out_channels=c_config[2],kernel_size=(3,3),stride=(1,1),padding=(1,1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=c_config[2],out_channels=c_config[2],kernel_size=(3,3),stride=(1,1),padding=(1,1)),
            nn.ReLU()
        )
        self.x2_2_transpose=nn.ConvTranspose2d(in_channels=c_config[2],out_channels=c_config[1],kernel_size=(2,2),stride=(2,2))


        self.x0_3=nn.Sequential(
            nn.Conv2d(in_channels=c_config[0]*3,out_channels=c_config[0],kernel_size=(3,3),stride=(1,1),padding=(1,1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=c_config[0],out_channels=c_config[0],kernel_size=(3,3),stride=(1,1),padding=(1,1)),
            nn.ReLU(),
        )

        self.x1_3=nn.Sequential(
            nn.Conv2d(in_channels=c_config[1]*4,out_channels=c_config[1],kernel_size=(3,3),stride=(1,1),padding=(1,1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=c_config[1],out_channels=c_config[1],kernel_size=(3,3),stride=(1,1),padding=(1,1)),
            nn.ReLU(),
        )
        self.x1_3_transpose=nn.ConvTranspose2d(in_channels=c_config[1],out_channels=c_config[0],kernel_size=(2,2),stride=(2,2))

        self.x0_4=nn.Sequential(
            nn.Conv2d(in_channels=c_config[0]*5,out_channels=c_config[0],kernel_size=(3,3),stride=(1,1),padding=(1,1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=c_config[0],out_channels=c_config[0],kernel_size=(3,3),stride=(1,1),padding=(1,1)),
            nn.ReLU(),
        )


        if deep_supervision:
            self.output_layer1=nn.Conv2d(in_channels=c_config[0],out_channels=n_class,kernel_size=(1,1))
            self.output_layer2=nn.Conv2d(in_channels=c_config[0],out_channels=n_class,kernel_size=(1,1))
            self.output_layer3=nn.Conv2d(in_channels=c_config[0],out_channels=n_class,kernel_size=(1,1))
            self.output_layer4=nn.Conv2d(in_channels=c_config[0],out_channels=n_class,kernel_size=(1,1))
        else:
            self.output_layer=nn.Conv2d(in_channels=c_config[0],out_channels=n_class,kernel_size=(1,1))



    def forward(self,x):

        x0_0=self.x0_0(x)
        x1_0=self.x1_0(self.m_pool(x0_0))
        x1_0_transpose=self.x1_0_transpose(x1_0)
        x0_1=self.x0_1(torch.cat([x0_0,x1_0_transpose],dim=1))


        x2_0=self.x2_0(self.m_pool(x1_0))
        x2_0_transpose=self.x2_0_transpose(x2_0)
        x1_1=self.x1_1(torch.cat([x1_0,x2_0_transpose],dim=1))
        x1_1_transpose=self.x1_1_transpose(x1_1)
        x0_2=self.x0_2(torch.cat([x0_0,x0_1,x1_1_transpose],dim=1))

        x3_0=self.x3_0(self.m_pool(x2_0))
        x3_0_transpose=self.x3_0_transpose(x3_0)
        x2_1=self.x2_1(torch.cat([x2_0,x3_0_transpose],dim=1))
        x2_1_transpose=self.x2_1_transpose(x2_1)
        x1_2=self.x1_2(torch.cat([x1_0,x1_1,x2_1_transpose],dim=1))
        x1_2_transpose=self.x1_2_transpose(x1_2)
        x0_3=self.x0_3(torch.cat([x0_0,x0_2,x1_2_transpose],dim=1))

        x4_0=self.x4_0(self.m_pool(x3_0))
        x4_0_transpose=self.x4_0_transpose(x4_0)
        x3_1=self.x3_1(torch.cat([x3_0,x4_0_transpose],dim=1))
        x3_1_transpose=self.x3_1_transpose(x3_1)
        x2_2=self.x2_2(torch.cat([x2_0,x2_1,x3_1_transpose],dim=1))
        x2_2_transpose=self.x2_2_transpose(x2_2)
        x1_3=self.x1_3(torch.cat([x1_0,x1_1,x1_2,x2_2_transpose],dim=1))
        x1_3_transpose=self.x1_3_transpose(x1_3)
        x0_4=self.x0_4(torch.cat([x0_0,x0_1,x0_2,x0_3,x1_3_transpose],dim=1))


        if self.deep_supervision:
            out1=self.output_layer1(x0_1)
            out2=self.output_layer2(x0_2)
            out3=self.output_layer3(x0_3)
            out4=self.output_layer4(x0_4)
            return [out1,out2,out3,out4]
        else:
            out=self.output_layer(x0_4)
            return out
        
def get_model(image_channel=3, number_of_class=2):
    net = UNET_PLUS_PLUS(image_channel=image_channel,n_class=number_of_class)
    return net
if __name__=="__main__":
    IMG_CHANNEL=3
    N_CLASS=2
    CLASS_CONFIG=[64,128,256,512,1024]

    model=UNET_PLUS_PLUS(image_channel=IMG_CHANNEL,n_class=N_CLASS,c_config=CLASS_CONFIG,deep_supervision=False)

    TEST_INPUT=torch.randint(low=0,high=255,size=(2,3,224,224)).type(torch.float)

    output=model(TEST_INPUT)

    print(output.shape)