import torch
from torch import nn

'''
vgg
'''

def vgg_block(num_convs, in_channels, out_channels):
    layers = []
    for _ in range(num_convs):
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        in_channels = out_channels
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*layers)

vgg_conv = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))
def vgg(conv_ : tuple = vgg_conv, in_channels = 3):
    conv_blks = []
    in_channels = 3
    for (num_convs, out_channels) in conv_:
        conv_blks.append(vgg_block(num_convs, in_channels, out_channels))
        in_channels = out_channels
    return(nn.Sequential(*conv_blks,  nn.Flatten()))

