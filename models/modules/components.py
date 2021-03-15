import torch
import torch.nn as nn
from .unet.model import Unet
from .fpn.model import FPN
from .transformer.Models import Transformer
import torch.nn.functional as F
import pdb

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out')
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, 0, 0.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

def check_freeze(net, net_conf):
    freeze = True if 'freeze' in net_conf and net_conf['freeze'] is True else False
    if freeze:
        for param in net.parameters():
            param.requires_grad = False

def get_nlayer_cnn(input_channels, channels, pool=False):
    net = []
    for i in range(len(channels)):
        if i == 0:
            net += [nn.Conv2d(input_channels,
                              channels[i], kernel_size=3, stride=1,
                              padding=1, bias=False),
                    nn.BatchNorm2d(channels[i]),
                    nn.ReLU(True)]
            if pool:
                net += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif i != len(channels)-1:
            net += [nn.Conv2d(channels[i-1],
                              channels[i], kernel_size=3, stride=1,
                              padding=1, bias=False),
                    nn.BatchNorm2d(channels[i]), nn.ReLU(True)]
        else:
            net += [nn.Conv2d(channels[i-1],
                              channels[i], kernel_size=1, stride=1,
                              padding=0, bias=False),
                    nn.BatchNorm2d(channels[i])]


    net = nn.Sequential(*net)
    net.apply(weights_init)

    return net

def get_enc_net(net_conf):
    enc_conf = net_conf['enc_net'] 
    if enc_conf['net_name'] == 'fpn_resnet_50':
        net = FPN('resnet50',
                  in_channels=enc_conf['num_in_channels'],
                  classes=enc_conf['num_out_channels'],
                  encoder_depth=enc_conf['depth'],
                  upsampling=enc_conf['upsampling'])
    elif enc_conf['net_name'] == 'fpn_vgg_16':
        net = FPN('vgg16_bn',
                  in_channels=enc_conf['num_in_channels'],
                  classes=enc_conf['num_out_channels'],
                  encoder_depth=enc_conf['depth'],
                  upsampling=enc_conf['upsampling'])
    elif enc_conf['net_name'] == 'unet_resnet_50':
        net = Unet('resnet50',
                   in_channels=enc_conf['num_in_channels'],
                   classes=enc_conf['num_out_channels'],
                   encoder_depth=enc_conf['encoder_depth'],
                   decoder_depth=enc_conf['decoder_depth'],
                   decoder_channels=enc_conf['decoder_channels'],
                   use_y_avg_pool=enc_conf['use_y_avg_pool'],
                   use_x_avg_pool=enc_conf['use_x_avg_pool'],
                   decoder_attention_type=enc_conf['att_type'] if 'att_type' in enc_conf else 'scse')

    elif enc_conf['net_name'] == 'unet_darts':
        net = Unet('darts_imagenet',
                   in_channels=enc_conf['num_in_channels'],
                   classes=enc_conf['num_out_channels'],
                   encoder_depth=enc_conf['encoder_depth'],
                   decoder_depth=enc_conf['decoder_depth'],
                   decoder_channels=enc_conf['decoder_channels'],
                   use_y_avg_pool=enc_conf['use_y_avg_pool'],
                   use_x_avg_pool=enc_conf['use_x_avg_pool'],
                   decoder_attention_type=enc_conf['att_type'] if 'att_type' in enc_conf else 'scse')

    elif enc_conf['net_name'] == 'unet_resnet_101':
        net = Unet('resnet101',
                   in_channels=enc_conf['num_in_channels'],
                   classes=enc_conf['num_out_channels'],
                   encoder_depth=enc_conf['encoder_depth'],
                   decoder_depth=enc_conf['decoder_depth'],
                   decoder_channels=enc_conf['decoder_channels'],
                   use_y_avg_pool=enc_conf['use_y_avg_pool'],
                   use_x_avg_pool=enc_conf['use_x_avg_pool'],
                   decoder_attention_type=enc_conf['att_type'] if 'att_type' in enc_conf else 'scse')

    elif enc_conf['net_name'] == 'unet_vgg_16':
        net = Unet('vgg16_bn',
                   in_channels=enc_conf['num_in_channels'],
                   classes=enc_conf['num_out_channels'],
                   encoder_depth=enc_conf['depth'],
                   decoder_channels=enc_conf['decoder_channels'],
                   decoder_attention_type=enc_conf['att_type'] if 'att_type' in enc_conf else 'scse')

    elif enc_conf['net_name'] == 'fpn_resnet_50_3x':
        net = FPN('resnet50',
                  in_channels=enc_conf['num_in_channels'],
                  classes=enc_conf['num_out_channels'],
                  encoder_depth=enc_conf['depth'],
                  upsampling=enc_conf['upsampling'])
    elif enc_conf['net_name'] == 'resnet_50_4x':
        net = resnet_v1.resnet50(True)

    elif enc_conf['net_name'] == 'unet_efficient_net':
        net = Unet('efficientnet-b3',
                   in_channels=enc_conf['num_in_channels'],
                   classes=enc_conf['num_out_channels'],
                   encoder_depth=enc_conf['encoder_depth'],
                   decoder_depth=enc_conf['decoder_depth'],
                   decoder_channels=enc_conf['decoder_channels'],
                   decoder_attention_type='scse')

    else:
        raise ValueError

    check_freeze(net, enc_conf)
    return net

def get_cls_net(net_conf):

    channels = net_conf['cls_net']['channels']
    channels = [net_conf['enc_net']['num_out_channels']] + channels

    net = []
    net.append(nn.BatchNorm1d(channels[0]))
    for i in range(len(channels)-1):
        net.append(nn.Linear(channels[i], channels[i+1]))
        net.append(nn.BatchNorm1d(channels[i+1]))
        if i != len(channels)-2:
            net.append(nn.ReLU(True))

    net = nn.Sequential(*net)
    return net

