#Copied from https://github.com/mtliba/360-images-VGG-based-Autoencoder-Pytorch/blob/master/model.py

import torch
from torch import nn
from torch.nn.modules.upsampling import Upsample
# from torch.nn.functional import interpolate
from torch.nn import MaxPool2d, ConvTranspose2d
from torch.nn.modules.conv import Conv2d
from torch.nn.modules.activation import Sigmoid, ReLU

# create max pooling layer 
# class Downsample(nn.Module):
#     # specify the kernel_size for downsampling 
#     def __init__(self, kernel_size, stride=2):
#         super(Downsample, self).__init__()
#         self.pool = MaxPool2d
#         self.kernel_size = kernel_size
#         self.stride = stride

#     def forward(self, x):
#         x = self.pool(kernel_size=self.kernel_size, stride=self.stride)(x)
#         return x

# create unpooling layer 
# class Upsample(nn.Module):
#     # specify the scale_factor for upsampling 
#     def __init__(self, scale_factor, mode):
#         super(Upsample, self).__init__()
#         self.interp = interpolate
#         self.scale_factor = scale_factor
#         self.mode = mode

#     def forward(self, x):
#         x = self.interp(x, scale_factor=self.scale_factor, mode=self.mode)
#         return x

class Encoder(nn.Module):
    def  __init__(self):
        super(Encoder, self).__init__()
        # Create encoder based on VGG16 architecture 
        # Change just 4,5 th maxpooling layer to 4 scale instead of 2 
        # select only convolutional layers first 5 conv blocks ,cahnge maxpooling=> enlarge receptive field
        # each neuron on bottelneck will see (580,580) all viewports  ,
        # input (576,288) , features numbers on bottelneck (9*4)*512, exclude last maxpooling
        encoder_list = [
            Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(),
            Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(),
            MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(),
            Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(),          
            MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(),
            Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(),           
            Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(),
            MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(),
            Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(),          
            Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(),
            MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(),
            Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(),              
            Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(),
            # MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)                     
        ]
        self.features = torch.nn.Sequential(*encoder_list)
        print("encoder initialized")
        # print("architecture len :",str(len(self.Autoencoder))) 

    def forward(self, input):
        x = self.features(input)
        return x
class Decoder(nn.Module):
    def  __init__(self):
        super(Decoder, self).__init__()

        # This worked
        # decoder_list= [
        #     # Upsample(scale_factor= 2, mode='nearest'),

        #     Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        #     ReLU(),
        #     Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        #     ReLU(),
        #     Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        #     ReLU(),
        #     Upsample(scale_factor= 2, mode='nearest'),

        #     Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        #     ReLU(),
        #     Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        #     ReLU(),
        #     Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        #     ReLU(),
        #     Upsample(scale_factor= 2, mode='nearest'),

        #     Conv2d(512, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        #     ReLU(),
        #     Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        #     ReLU(),
        #     Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        #     ReLU(),
        #     Upsample(scale_factor=2, mode='nearest'),

        #     Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        #     ReLU(),
        #     Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        #     ReLU(),
        #     Upsample(scale_factor=2, mode='nearest'),

        #     Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        #     ReLU(),
        #     Conv2d(64, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        #     # ReLU(),
        #     # Conv2d(64, 1, kernel_size=(1, 1), stride=(1, 1), padding=(1, 1)),
        #     # Sigmoid(), #Arka's not so bad  idea. Put it back.
        # ]


        # all deconv doesnt seems to not work.
        # Now trying only replacing maxpool ones
        # decoder_list= [
        #     # Upsample(scale_factor= 2, mode='nearest'),

        #     Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        #     ReLU(),
        #     Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        #     ReLU(),
        #     Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        #     ReLU(),
        #     ConvTranspose2d(512, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=1),

        #     Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        #     ReLU(),
        #     Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        #     ReLU(),
        #     Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        #     ReLU(),
        #     ConvTranspose2d(512, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=1),

        #     Conv2d(512, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        #     ReLU(),
        #     Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        #     ReLU(),
        #     Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        #     ReLU(),
        #     ConvTranspose2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=1),

        #     Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        #     ReLU(),
        #     Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        #     ReLU(),
        #     ConvTranspose2d(128, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=1),

        #     Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        #     ReLU(),
        #     Conv2d(64, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        #     # ReLU(),
        #     # Conv2d(64, 1, kernel_size=(1, 1), stride=(1, 1), padding=(1, 1)),
        #     # Sigmoid(), #Arka's not so bad  idea. Put it back. Kidding this should be removed!
        # ]

        # PIXELUNSHUFFLE
        decoder_list= [
            # Upsample(scale_factor= 2, mode='nearest'),

            Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(),
            Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(),
            Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(),
            nn.PixelShuffle(2),

            Conv2d(128, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(),
            Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(),
            Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(),
            nn.PixelShuffle(2),

            Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(),
            Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(),
            Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(),
            nn.PixelShuffle(2),

            Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(),
            Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(),
            nn.PixelShuffle(2),

            Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(),
            Conv2d(64, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            # ReLU(),
            # Conv2d(64, 1, kernel_size=(1, 1), stride=(1, 1), padding=(1, 1)),
            # Sigmoid(), #Arka's not so bad  idea. Put it back.
        ]

        self.decoder = torch.nn.Sequential(*decoder_list)
        self._initialize_weights()
        print("decoder initialized")
        # print("architecture len :",str(len(self.Autoencoder))) 

    def forward(self, input):
        x = self.decoder(input)
        return x 
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            
class Autoencoder(nn.Module):
    """
    In this model, we aggregate encoder and decoder
    """
    def  __init__(self , pretrained_encoder=True):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        if pretrained_encoder:
            state_dict = torch.hub.load_state_dict_from_url('https://download.pytorch.org/models/vgg16-397923af.pth',progress=True)
            self.encoder.load_state_dict(state_dict, strict=False)
        print("Model initialized")
        # print("architecture len :", str(len(self.Autoencoder)))

    def forward(self, input):
        x = self.encoder(input)
        x = self.decoder(x)
        return x 