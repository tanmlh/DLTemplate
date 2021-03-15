import torch
import torch.nn as nn
import torch.nn.functional as F
from . import components
import pdb


class GANLoss(nn.Module):

    def __init__(self, net_conf):
        super(GANLoss, self).__init__()
        self.net_conf = net_conf
        self.dis_net = components.get_dis_net(net_conf)
        self.l1_loss_fun = nn.SmoothL1Loss()

    def forward(self, pred_pos, tensors, loss_type='gen'):

        label = tensors['label']
        B, label_len = label.shape
        label_mask = (label > 0).float()
        masked_pred_pos = pred_pos * label_mask.unsqueeze(2) # (B, label_len, 2)

        if loss_type == 'gen':
            D_fake = self.dis_net(pred_pos.view(B, -1))
            tensors['gan_real_label'].resize_(D_fake.size()).fill_(1)
            loss_GAN = self.l1_loss_fun(D_fake, tensors['gan_real_label'])

        else:

            real_pos = tensors['cen_pos']
            # masked_real_pos = real_pos[label_mask[:, 0], label_mask[:, 1], :]
            masked_real_pos = real_pos * label_mask.unsqueeze(2)
            D_fake = self.dis_net(masked_pred_pos.view(B, -1))
            D_real = self.dis_net(masked_real_pos.view(B, -1))

            tensors['gan_fake_label'].resize_(D_fake.size()).fill_(0)
            tensors['gan_real_label'].resize_(D_fake.size()).fill_(1)

            loss_D1 = self.l1_loss_fun(D_fake, tensors['gan_fake_label'])
            loss_D2 = self.l1_loss_fun(D_real, tensors['gan_real_label'])
            loss_GAN = (loss_D1 + loss_D2) / 2

        return loss_GAN * 0.01
