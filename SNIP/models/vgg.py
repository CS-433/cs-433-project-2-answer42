import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .init_utils import weights_init

defaultcfg = {
    11: [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    13: [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    16: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512],
    19: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512],
}

class VGG(nn.Module):
    def __init__(self, num_channels, num_classes, depth, batch_norm=True, affine=False):
        if depth not in defaultcfg:
            raise ValueError('Unknown VGG config for depth', depth)
        self.affine = affine
        config = defaultcfg[depth]
        self.feature = make_layers(config, num_channels, batch_norm)
        self.fc = nn.Linear(config[-1], num_classes)
        self.apply(weights_init)

    def make_layers(self, config, num_channels, batch_norm):
        layers = []
        in_channels = num_channels
        for v in config:
            if v == 'M':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                layers.append(nn.Conv2d(in_channels, v, kernel_size=3, padding=1, bias=False))
                if batch_norm:
                    layers.append(nn.BatchNorm2d(v, affine=self.affine))
                layers.append(nn.ReLU(inplace=True))
                in_channels = v 
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.feature(x)
        out = F.avg_pool2d(out, out.size(3))
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def vgg19(dataset):
    depth = 19
    batch_norm = True
    affine = False
    NETS = {
        'cifar10': lambda: VGG(3, 10, depth, batch_norm, affine)
        'cifar100': lambda: VGG(3, 100, depth, batch_norm, affine),
        'tinyimagenet': lambda: VGG(3, 200, depth, batch_norm, affine)
    }
    return NETS[dataset]()