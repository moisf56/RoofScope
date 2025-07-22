import torch 
import torch.nn as nn
from UNET_blocks import DoubleConv, DownSample, UpSample
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

class UNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.down_conv_1 = DownSample(in_channels, 64)
        self.down_conv_2 = DownSample(64, 128)
        self.down_conv_3 = DownSample(128, 256)
        self.down_conv_4 = DownSample(256, 512)


        self.bottle_neck = DoubleConv(512, 1024)


        self.up_conv_1 = UpSample(1024, 512)
        self.up_conv_2 = UpSample(512, 256)
        self.up_conv_3 = UpSample(256, 128)
        self.up_conv_4 = UpSample(128, 64)


        self.out = nn.Conv2d(in_channels=64, out_channels = num_classes, kernel_size=1, )          #adjust out_channels for number of segmentation classs

    
    def forward(self, x):

        down_1, p1 = self.down_conv_1(x)
        down_2, p2 = self.down_conv_2(p1)
        down_3, p3 = self.down_conv_3(p2)
        down_4, p4 = self.down_conv_4(p3)


        bottle = self.bottle_neck(p4)

        up_1 = self.up_conv_1(bottle, down_4)
        up_2 = self.up_conv_2(up_1, down_3)
        up_3 = self.up_conv_3(up_2, down_2)
        up_4 = self.up_conv_4(up_3, down_1)


        out = self.out(up_4)
        return out
    


