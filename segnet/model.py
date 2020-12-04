import random
import torch
import torchvision
import torch.nn.functional as F


class SegNet(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(SegNet, self).__init__()

        # SegNet Basic Architecture
        self.conv1 = self.__enc_conv_module(input_size, 64, 3, 1)
        self.conv2 = self.__enc_conv_module(64, 128, 3, 1)
        self.conv3 = self.__enc_conv_module(128, 256, 3, 1)
        self.conv4 = self.__enc_conv_module(256, 512, 3, 1)

        self.deconv4 = self.__dec_conv_module(512, 256, 3, 1)
        self.deconv3 = self.__dec_conv_module(256, 128, 3, 1)
        self.deconv2 = self.__dec_conv_module(128, 64, 3, 1)


    """
    Represents module in SegNet for convolution
    """
    def __enc_conv_module(self, in_channels, out_channels, kernel_size, pad):
        module = torch.nn.Sequential(torch.nn.Conv2d(in_channels=in_channels, 
                                                     out_channels=out_channels,
                                                     kernel_size=kernel_size,
                                                     padding=pad,
                                                     ),
                                     torch.nn.BatchNorm2d(out_channels),                 
                                )
        return module

    def __dec_conv_module(self, in_channels, out_channels, kernel_size, pad):
        module = torch.nn.Sequential(torch.nn.ConvTranspose2d(in_channels=in_channels,
                                                              out_channels=out_channels,
                                                              kernel_size=kernel_size,
                                                              padding=pad,
                                                              ),
                                     torch.nn.BatchNorm2d(out_channels),
                                    )
        return module



    """
    Forward pass for inference
    """
    def forward(self, x):
        pass

