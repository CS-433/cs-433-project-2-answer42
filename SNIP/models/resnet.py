import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from .init_utils import weights_init

_AFFINE = True
USE_BIAS = False


def create_convolution_normalization(
        in_channels, out_channels, kernel_size, stride, padding, bias):
    conv = nn.Conv2d(
        in_channels, out_channels, kernel_size=kernel_size, 
        stride=stride, padding=padding, bias=bias)
    bn = nn.BatchNorm2d(out_channels, affine=_AFFINE)
    return conv, bn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1, self.bn1 = create_convolution_normalization(
            in_channels, out_channels, kernel_size=3, 
            stride=stride, padding=1, bias=USE_BIAS
        )
        self.conv2, self.bn2 = create_convolution_normalization(
            out_channels, out_channels, kernel_size=3, 
            stride=1, padding=1, bias=USE_BIAS 
        )

        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        self.bn3 = None
        if stride != 1 or in_channels != out_channels:
            self.downsample, self.bn3 = create_convolution_normalization(
                in_channels, out_channels, kernel_size=1, stride=stride,
                padding=0, bias=USE_BIAS
            )

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            residual = self.bn3(self.downsample(x))
        out += residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, in_channels, num_classes, depth):
        super(ResNet, self).__init__()
        
        assert (depth - 2) % 6 == 0, 'depth must be 6n+2'
        n = (depth - 2) // 6

        channels_per_layer = [32, 64, 128]

        self.conv, self.bn = create_convolution_normalization(
            in_channels, channels_per_layer[0], kernel_size=3,
            stride=1, padding=1, bias=USE_BIAS
        )
        self.layer1 = self._make_layer(
            channels_per_layer[0], channels_per_layer[0], stride=1, num_blocks=n)
        self.layer2 = self._make_layer(
            channels_per_layer[0], channels_per_layer[1], stride=2, num_blocks=n)
        self.layer3 = self._make_layer(
            channels_per_layer[1], channels_per_layer[2], stride=2, num_blocks=n)
        self.avgpool = nn.AvgPool2d(8)
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(channels_per_layer[2], num_classes)

        self.apply(weights_init)

    def _make_layer(self, in_channels, out_channels, stride, num_blocks):
        layers = []
        for i in range(num_blocks):
            layers.append(ResidualBlock(in_channels, out_channels, stride))
            stride = 1
            in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.bn(self.conv(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        #print(out.size()) # torch.Size([64, 128, 1, 1])
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def resnet32(dataset):
    NETS = {
        'cifar10': lambda: ResNet(3, 10, 32),
        'cifar100': lambda: ResNet(3, 100, 32),
        'tinyimagenet': lambda: ResNet(3, 10, 32)
    }
    return NETS[dataset]()