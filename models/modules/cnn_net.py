import torch
import torch.nn as nn
import torch.nn.functional as F
from .base.initialization import initialize

def get_cnn_net(net_conf):

    channels = net_conf['channels']

    net = []
    for i in range(len(channels) - 1):
        if i != len(channels) - 1:
            net += [nn.Conv2d(channels[i],
                              channels[i+1], kernel_size=3, stride=1,
                              padding=1, bias=False),
                    nn.BatchNorm2d(channels[i]),
                    nn.ReLU(True)]
            # if pool:
            #     net += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            net += [nn.Conv2d(channels[i-1],
                              channels[i], kernel_size=3, stride=1,
                              padding=1, bias=False),
                    nn.BatchNorm2d(channels[i])]

    net = nn.Sequential(*net)
    net.apply(initialize)

    return net
