import random
import torch
import torchvision
import torch.nn.functional as F
import numpy as np


"""
Class for converting masks to labels for training.
"""
class DataLabels:
    """
    Init with dictionary mapping from tuples to integers
    """
    def __init__(self, class_dict):
        self.color2label = class_dict

    def get_label(self, color):
        return self.color2label(tuple(color))

    """
    Label a batch of masks.
    """
    def label_batch(self, batch):
        curr_batch =  batch.numpy()
        curr_batch = curr_batch.reshape( (curr_batch.shape[0], 
                                          curr_batch.shape[2],
                                          curr_batch.shape[3],
                                          curr_batch.shape[1]) )

        labels = np.zeros( (batch.size()[0],
                            batch.size()[2],
                            batch.size()[3]) )

        for b in range(batch.size()[0]):
            for i in range(batch.size()[1]):
                for j in range(batch.size()[2]):
                    try:
                        labels[b][i][j] = self.get_label(curr_batch[b][i][j])
                    except:
                        labels[b][i][j] = 0

        return torch.from_numpy(labels).type(torch.int64)



"""
SegNet model as defined in paper. Consists of 5 encoder layers and 5 decoder layers.
The decoder layers receive max poolingn indices from the encoder. The output is an image
with the number of channels equal to the number of classes we are predicting.
"""
class SegNet(torch.nn.Module):

    def __init__(self, input_ch, output_ch):
        super(SegNet, self).__init__()

        # SegNet Architecture
        self.conv11 = self.__enc_conv_module(input_ch, 64, 3, 1)
        self.conv12 = self.__enc_conv_module(64, 64, 3, 1)
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        self.conv21 = self.__enc_conv_module(64, 128, 3, 1)
        self.conv22 = self.__enc_conv_module(128, 128, 3, 1)
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        self.conv31 = self.__enc_conv_module(128, 256, 3, 1)
        self.conv32 = self.__enc_conv_module(256, 256, 3, 1)
        self.conv33 = self.__enc_conv_module(256, 256, 3, 1)
        self.pool3 = torch.nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        self.conv41 = self.__enc_conv_module(256, 512, 3, 1)
        self.conv42 = self.__enc_conv_module(512, 512, 3, 1)
        self.conv43 = self.__enc_conv_module(512, 512, 3, 1)
        self.pool4 = torch.nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        self.conv51 = self.__enc_conv_module(512, 1024, 3, 1)
        self.conv52 = self.__enc_conv_module(1024, 1024, 3, 1)
        self.conv53 = self.__enc_conv_module(1024, 1024, 3, 1)
        self.pool5 = torch.nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        self.deconv51 = self.__dec_conv_module(1024, 1024, 3, 1)
        self.deconv52 = self.__dec_conv_module(1024, 1024, 3, 1)
        self.deconv53 = self.__dec_conv_module(1024, 512, 3, 1)

        self.deconv41 = self.__dec_conv_module(512, 512, 3, 1)
        self.deconv42 = self.__dec_conv_module(512, 512, 3, 1)
        self.deconv43 = self.__dec_conv_module(512, 256, 3, 1)

        self.deconv31 = self.__dec_conv_module(256, 256, 3, 1)
        self.deconv32 = self.__dec_conv_module(256, 256, 3, 1)
        self.deconv33 = self.__dec_conv_module(256, 128, 3, 1)

        self.deconv21 = self.__dec_conv_module(128, 128, 3, 1)
        self.deconv22 = self.__dec_conv_module(128, 64, 3, 1)

        self.deconv11 = self.__dec_conv_module(64, 64, 3, 1)
        self.deconv12 = self.__dec_conv_module(64, output_ch, 3, 1)


    """
    Represents modules in SegNet for convolution/deconvolution
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
    Forward pass for inference. The outputs are the logits.
    """
    def forward(self, x):
        # ENCODER
        ########################################################################
        dim_1 = x.size()
        z = F.relu(self.conv11(x))
        z = F.relu(self.conv12(z))
        z, idx_1 = self.pool1(z)

        dim_2 = z.size()
        z = F.relu(self.conv21(z))
        z = F.relu(self.conv22(z))
        z, idx_2 = self.pool2(z)

        dim_3 = z.size()
        z = F.relu(self.conv31(z))
        z = F.relu(self.conv32(z))
        z = F.relu(self.conv33(z))
        z, idx_3 = self.pool3(z)

        dim_4 = z.size()
        z = F.relu(self.conv41(z))
        z = F.relu(self.conv42(z))
        z = F.relu(self.conv43(z))
        z, idx_4 = self.pool4(z)

        dim_5 = z.size()
        z = F.relu(self.conv51(z))
        z = F.relu(self.conv52(z))
        z = F.relu(self.conv53(z))
        z, idx_5 = self.pool5(z)
        ########################################################################

        # DECODER
        ########################################################################
        z = F.max_unpool2d(z, idx_5, kernel_size=2, stride=2, output_size=dim_5)
        z = self.deconv51(z)
        z = self.deconv52(z)
        z = self.deconv53(z)

        z = F.max_unpool2d(z, idx_4, kernel_size=2, stride=2, output_size=dim_4)
        z = self.deconv41(z)
        z = self.deconv42(z)
        z = self.deconv43(z)

        z = F.max_unpool2d(z, idx_3, kernel_size=2, stride=2, output_size=dim_3)
        z = self.deconv31(z)
        z = self.deconv32(z)
        z = self.deconv33(z)

        z = F.max_unpool2d(z, idx_2, kernel_size=2, stride=2, output_size=dim_2)
        z = self.deconv21(z)
        z = self.deconv22(z)

        z = F.max_unpool2d(z, idx_1, kernel_size=2, stride=2, output_size=dim_1)
        z = self.deconv11(z)
        z = self.deconv12(z)

        out_softmax = F.softmax(z, dim=1)
        ########################################################################

        return z, out_softmax

