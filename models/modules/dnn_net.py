import torch
import torch.nn as nn
import torch.nn.functional as F
from .base.initialization import initialize

def get_dnn_net(net_conf):

    channels = net_conf['dnn_net']['channels']

    net = []
    # net.append(nn.BatchNorm1d(channels[0]))
    for i in range(len(channels)-1):
        net.append(nn.Linear(channels[i], channels[i+1]))
        net.append(nn.BatchNorm1d(channels[i+1]))
        if i != len(channels)-2:
            net.append(nn.ReLU(True))

    net = nn.Sequential(*net)
    net.apply(initialize)

    return net

