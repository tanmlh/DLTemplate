from __future__ import absolute_import

import os
import io
import sys
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from utils.transforms import ImageNetPolicy
import pdb
from PIL import Image
import pickle
import time

class DummyDataset(Dataset):

    def __init__(self, loader_conf, phase='train'):

        super(DummyDataset, self).__init__()

        self.num_used_data = loader_conf['num_used_data']
        self.resize_type = loader_conf['resize_type']
        self.file_path = loader_conf['file_path']
        self.label_len = loader_conf['label_len']
        self.img_H = loader_conf['img_H']
        self.img_W = loader_conf['img_W']
        self.hw_ratio = loader_conf['hw_ratio']
        self.random_ratio = loader_conf['random_ratio']
        self.root = loader_conf['dataset_dir']
        self.num_classes = loader_conf['num_classes']
        self.phase = phase
        self.images = []
        self.labels = []
        self.lengths = []
        self.tfm_label_type = loader_conf['tfm_label_type'] if 'tfm_label_type' in loader_conf else 'indivisual'
        self.use_imagenet_aug = loader_conf['use_imagenet_aug'] if 'use_imagenet_aug' in loader_conf else False

        # augmentation
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        transform = []
        if phase == 'train':
            transform.append(transforms.RandomHorizontalFlip())
            if self.use_imagenet_aug:
                transform.append(ImageNetPolicy(keep_prob=0.5))

        transform.append(transforms.ToTensor())
        transform.append(normalize)
        self.transform = transforms.Compose(transform)

        if self.num_used_data != -1:
            self.data_list = self.data_list[:self.num_used_data]

    def __len__(self):
        return 128

    def __getitem__(self, index):
        dummy_data = torch.zeros(3, self.img_H, self.img_W)
        dummy_label = torch.zeros(1)

        out = {}
        out['image'] = dummy_data
        out['label'] = dummy_label

        return out

