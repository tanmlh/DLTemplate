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
import mc


class ChipDataset(Dataset):

    def __init__(self, loader_conf, phase='train'):

        super(ChipDataset, self).__init__()

        self.num_used_data = loader_conf['num_used_data']
        self.resize_type = loader_conf['resize_type']
        self.file_path = loader_conf['file_path']
        self.use_ceph = loader_conf['use_ceph']
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

        # check data and label
        with open(self.file_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                strs = line.strip().split(',')

                labels = np.zeros(self.label_len)
                tmp_labels = np.array(list(map(int, strs[1].split())))[::-1]
                if tmp_labels.size > self.label_len-1:
                    continue
                labels[:tmp_labels.size] += tmp_labels

                self.images.append(os.path.join(self.root, strs[0]))
                self.labels.append(labels)
                self.lengths.append(tmp_labels.size)

        if self.num_used_data != -1:
            self.data_list = self.data_list[:self.num_used_data]

        if self.use_ceph:
            # import ceph
            # self.s3client = ceph.S3Client()
            from petrel_client.client import Client
            self.s3client = Client('/mnt/lustre/zhangfahong/.s3cfg')

        self.img_read = self.ceph_read_img if self.use_ceph else self.mc_read_img



    def ceph_read_img(self, path):
        img_value = self.s3client.Get(path)
        value_buf = memoryview(img_value)
        try:
            img_array = np.frombuffer(value_buf, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        except Exception as e:
            raise IOError("image error:{} in loading from ceph: {}".format(e, path))
        img = img[:, :, ::-1]
        return img

    def mc_read_img(self, path):
        server_list_config_file = "/mnt/lustre/share/memcached_client/server_list.conf"
        client_config_file = "/mnt/lustre/share/memcached_client/client.conf"
        mclient = mc.MemcachedClient.GetInstance(server_list_config_file, client_config_file)
        value = mc.pyvector()
        mclient.Get(path, value)
        value_str = mc.ConvertBuffer(value)
        try:
            buff = io.BytesIO(value_str)
            with Image.open(buff) as img:
                img = img.convert('RGB')
        except:
            raise IOError('File error: ', path)
        return np.array(img)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img = self.img_read(self.images[index])
        H, W, _ = img.shape
        if self.phase == 'test':
            ratio = self.hw_ratio
        else:
            ratio = ((np.random.rand() * 2.0 - 1.0) * self.random_ratio) + self.hw_ratio

        output_h = min(self.img_H, int(ratio * float(H) / W * self.img_W))
        if output_h < self.img_H:
            output = np.zeros((self.img_H, self.img_W, 3), dtype=np.uint8) + 127
            output[:output_h, :, :] = cv2.resize(img, (self.img_W, output_h))
            img = output
            real_h = output_h
        else:
            img = cv2.resize(img, (self.img_W, self.img_H))
            real_h = self.img_H

        img = transforms.ToPILImage()(img)
        if self.transform is not None:
            img = self.transform(img)

        pos_range = -torch.ones(self.label_len, 2)
        # pos_range[:, 1] = real_h / self.img_H
        pos_range[:, 1] = 2 * real_h / self.img_H - 1
        for i in range(self.lengths[index], self.label_len):
            # pos_range[i, 0] = real_h / self.img_H
            pos_range[i, 0] = 2 * real_h / self.img_H - 1
            pos_range[i, 1] = 100

        seg_label = []
        last = -1
        for chip in self.labels[index]:
            if chip != last and chip != 0:
                seg_label.append(chip)
            last = chip
        seg_label_len = len(seg_label)
        temp = np.zeros(self.label_len)
        temp[:len(seg_label)] = seg_label
        seg_label = temp

        tfm_label = self.labels[index] if self.tfm_label_type == 'individual' else seg_label
        tfm_len = self.lengths[index] if self.tfm_label_type == 'individual' else seg_label_len

        label_in = np.zeros(self.label_len)
        label_in[0] = self.num_classes-2
        label_in[1:tfm_len+1] = tfm_label[:tfm_len]

        label_out = np.zeros(self.label_len)
        label_out[:tfm_len] = tfm_label[:tfm_len]
        label_out[tfm_len] = self.num_classes-1

        outputs = {}
        outputs['image'] = img
        outputs['label'] = torch.LongTensor(self.labels[index])
        outputs['label_in'] = torch.LongTensor(label_in)
        outputs['label_out'] = torch.LongTensor(label_out)
        outputs['chip_len'] = self.lengths[index]
        outputs['seg_label'] = torch.LongTensor(seg_label)
        outputs['seg_label_len'] = seg_label_len
        outputs['pos_range'] = pos_range

        return outputs

