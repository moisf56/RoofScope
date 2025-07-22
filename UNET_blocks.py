import numpy as np
import torch 
import torch.nn as nn
import torchvision.transforms.functional as TF

class DoubleConv(nn.Module):        #inherting from PyTorch
    def __init__(self, in_channels, out_channels):
        super(DoubleConv,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = 1) ,    #kernel size 3*3, padding = 1
            nn.BatchNorm2d(out_channels),            #normalize the output batch after the layer 
            nn.ReLU(inplace = True),            #Rectified Linear Unit Activation function f(x) = max(0,x)
            nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding = 1) , 
            #nn.BatchNorm2d(out_channels),          
            nn.ReLU(inplace = True),
        )


    def forward(self, x):
        return self.conv(x)             #apply the sequential conv. to input
    
    

class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2)           #Max ppoling to reduce spatial resolution and select prominent features



    def forward(self, x):
        down = self.conv(x)
        pool = self.pool(down)    

        return down, pool
    


class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size = 2, stride = 2) 
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x1, x2], 1)              #concatenate x1 and x2
        return self.conv(x)