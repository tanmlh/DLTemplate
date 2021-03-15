import random
import numpy as np
import pdb
import operator
import os
import sys
import time
import tqdm
import pickle

import torch
import torchvision as tv
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F

from .base_solver import BaseSolver
from .modules import components
from .modules import loss_fun
from utils import visualizer
from utils import metric
# import utils.distributed as dist

def get_solver(conf):
    return DummySolver(conf)

def get_model(conf):
    return DummyModel(conf)

class DummySolver(BaseSolver):

    def init_tensors(self):
        self.tensors = {}
        self.tensors['image'] = torch.FloatTensor()
        self.tensors['label'] = torch.LongTensor()

    def set_tensors(self, batch, phase):

        image, label = batch['image'], batch['label']

        self.tensors['image'].resize_(image.size()).copy_(image)
        self.tensors['label'].resize_(label.size()).copy_(label)

    def process_batch(self, batch, phase='train'):

        self.set_tensors(batch, phase)

        if phase == 'train':
            self.net.train()
        else:
            self.net.eval()

        loss, state = self.net.forward(self.tensors, phase)

        if phase == 'train':
            self.net.zero_grad()
            self.my_backward(loss)
            self.optimizer.step()

        if self.use_dist and self.rank != 0:
            return {}

        for key, value in state.items():
            if key.split('|')[0] == 'scalar':
                state[key] = value.mean().cpu().item()

        label = self.tensors['label'].detach().cpu()

        """ Visualization stuff """
        if self.global_step % self.print_freq == 0:
            image = self.tensors['image'].detach().cpu()

        return state

class DummyModel(nn.Module):
    def __init__(self, net_conf):
        super(DummyModel, self).__init__()

        self.net_conf = net_conf
        self.num_classes = net_conf['num_classes']

        self.enc_net = components.get_enc_net(net_conf)

        self.mse_loss_fun = nn.MSELoss()
        self.l1_loss_fun = nn.SmoothL1Loss()
        self.cse_loss_fun = nn.CrossEntropyLoss()
        self.nll_loss_fun = nn.NLLLoss()

    def forward(self, tensors, phase):

        """ Feature extraction """

        state = {}

        image = tensors['image']
        label = tensors['label'] # (B, label_len)

        features = self.enc_net(image) # (B, C, H, 1)

        B, C, H, W = features.shape
        label_len = label.shape[1]

        """ Loss Calculation """
        loss_total = None
        if phase == 'train':

            loss_w = self.net_conf['loss_weight']

            loss_dummy = features.sum()

            """ total loss """
            loss_total = sum([x * loss_w[i] for i, x in enumerate([loss_dummy])])
            state['scalar|loss'] = loss_total.detach()

        return loss_total, state

