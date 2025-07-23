from ARConv import ARConv
import torch
import torch.nn as nn
import torch.nn.functional as F

class ARConv_Block(nn.Module):
    def __init__(self, in_planes, flag=False):
        super(ARConv_Block, self).__init__()
        self.flag = flag
        self.conv1 = ARConv(in_planes, in_planes,3,1,1)
        self.relu = nn.ReLU()
        self.conv2 = ARConv(in_planes, in_planes,3,1,1)
    def forwad(self, x , epochs, hw_range):
        res = self.conv1(x, epochs, hw_range)
        res = self.relu(res)
        res = self.conv2(res, epochs, hw_range)
        x = x+res
        return x 
    
class ConvDown(nn.Module):
    def __init__(self, in_channels, dsconv=True, *args, **kwargs)->None:
        super().__init__(*args, **kwargs)
        if dsconv:
            self.conv = nn.Sequential(
               nn.Conv2d(in_channels, in_channels,2,2,0),
               nn.SELU(),
               nn.Conv2d(in_channels, in_channels,*2,3,1,1)                 
            )
    def forward(self,x):
        x = self.conv(x)
        return x

class ConvUp(nn.Module):
    def __init__(self, in_channels, dsconv=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv1 = nn.ConvTranspose2d(in_channels, in_channels // 2, 2, 2, 0)
        if dsconv:
            self.conv2 = nn.Sequential(
                nn.Conv2d(in_channels // 2, in_channels // 2, 3, 1, 1, groups=in_channels // 2, bias=False),
                nn.Conv2d(in_channels // 2, in_channels // 2, 1, 1, 0)
            )
        else:
            self.conv2 = nn.Conv2d(in_channels // 2, in_channels // 2, 3, 1, 1)
 
    def forward(self, x, y):
        x = F.leaky_relu(self.conv1(x))
        x = x + y
        x = F.leaky_relu(self.conv2(x))
        return x