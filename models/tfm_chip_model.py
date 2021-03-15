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
import linklink as link

from .base_solver import BaseSolver
from .modules import components
from .modules import loss_fun
import utils.distributed as dist
from utils import visualizer
from utils import metric

def get_solver(conf):
    return TFMChipSolver(conf)

def get_model(conf):
    return TFMChipModel(conf)

class TFMChipSolver(BaseSolver):

    def init_tensors(self):
        self.tensors = {}
        self.tensors['image'] = torch.FloatTensor()
        self.tensors['label'] = torch.LongTensor()
        self.tensors['label_in'] = torch.LongTensor()
        self.tensors['label_out'] = torch.LongTensor()
        self.tensors['chip_len'] = torch.FloatTensor()
        self.tensors['seg_label'] = torch.LongTensor()
        self.tensors['seg_label_len'] = torch.FloatTensor()

    def set_tensors(self, batch, phase):

        image, label, chip_len = batch['image'], batch['label'], batch['chip_len']
        seg_label = batch['seg_label']
        seg_label_len = batch['seg_label_len']
        label_in = batch['label_in']
        label_out = batch['label_out']

        self.tensors['image'].resize_(image.size()).copy_(image)
        self.tensors['label'].resize_(label.size()).copy_(label)
        self.tensors['label_in'].resize_(label_in.size()).copy_(label_in)
        self.tensors['label_out'].resize_(label_out.size()).copy_(label_out)
        self.tensors['chip_len'].resize_(chip_len.size()).copy_(chip_len)
        self.tensors['seg_label'].resize_(seg_label.size()).copy_(seg_label)
        self.tensors['seg_label_len'].resize_(seg_label_len.size()).copy_(seg_label_len)

    def process_batch(self, batch, phase='train'):

        downsample = self.loader_conf['downsample'] if 'downsample' in self.loader_conf else None

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
        merged_label = self.tensors['seg_label'].detach().cpu()
        used_label = label if self.net_conf['label_type'] == 'individual' else merged_label

        """ tfm prediction """
        pred_prob_tfm = state.pop('else|pred_prob_tfm').cpu()
        decode_res = self.decode_label(pred_prob_tfm, used_label, phase, decode_type='tfm')
        acc_tfm = decode_res['acc']
        pred_str = decode_res['pred_str']
        label_str = decode_res['label_str']
        err_idx = decode_res['err_idx']

        state['scalar|acc_tfm'] = acc_tfm

        if phase != 'train':
            state['else|pred_prob'] = pred_prob_tfm
            state['else|label'] = used_label
            state['else|chip_len'] = self.tensors['chip_len'].cpu()
            state['else|acc_list'] = torch.tensor(decode_res['acc_list'])

        """ Visualization stuff """
        if self.global_step % self.print_freq == 0:

            ori_image = self.tensors['image'].detach().cpu()
            enc_att = state['else|enc_att'].cpu()
            env_att = state['else|env_att'].cpu() if 'else|env_att' in state else None
            env_att_str = [','.join(list(map('{:.4f}'.format, x.tolist()))) for x in env_att] if env_att is not None else None

            image = ori_image
            if downsample is not None:
                image = torch.nn.functional.avg_pool2d(ori_image, downsample, downsample)


            if 'record_fail' in self.solver_conf and self.solver_conf['record_fail']:
                if err_idx != []:
                    image = image[err_idx]
                    ori_image = ori_image[err_idx]
                    pred_str = [pred_str[i] for i in err_idx]
                    label_str = [label_str[i] for i in err_idx]
                    enc_att = enc_att[err_idx]

            state['image|pred_label'] = visualizer.get_image_text(ori_image, pred_str)
            state['image|real_label'] = visualizer.get_image_text(ori_image, label_str)
            state['image|enc_att'] = visualizer.get_image_enc_att(ori_image, enc_att, downsample)
            if env_att is not None:
                state['image|env_att'] = visualizer.get_image_text(ori_image, env_att_str)

            if self.net_conf['use_unc']:
                confidence = state.pop('else|confidence').cpu()
                temp = []
                for line in confidence:
                    conf_str = [str(x.item()) for x in line]
                    conf_str = ','.join(conf_str)
                    temp.append(conf_str)
                state['image|confidence'] = visualizer.get_image_text(ori_image, temp)

        return state

    def decode_label(self, pred_prob, label, phase='train', decode_type='ctc', pred_len=None):

        B, _, num_classes = pred_prob.shape

        max_prob, pred_label = pred_prob.max(2) # (B, label_len)
        prob_str = []

        for prob_seq in max_prob:
            str_seq = []
            for prob in prob_seq:
                str_seq.append('{:.3f}'.format(prob.item()))

            prob_str.append(str_seq)


        decode_res = metric.acc_compute(self.loader_conf, pred_label,
                                        label, decode_type=decode_type,
                                        phase=phase, pred_label_len=pred_len)

        out_str_list = []
        for x_list, y_list in zip(decode_res['pred_str'], prob_str):
            temp = x_list.split(',')[:-1]
            output_str = [x + ' - ' + y for x, y in zip(temp, y_list)]
            output_str = ','.join(output_str)
            out_str_list.append(output_str)

        decode_res['pred_str'] = out_str_list

        return decode_res

    def test(self, test_loader):
        self.load_to_gpu()
        tq = tqdm.tqdm(test_loader)
        epoch_pred_prob = []
        epoch_labels = []
        epoch_chip_len = []
        epoch_dec_out = []
        epoch_acc_list = []
        epoch_max_chip_feats = []
        epoch_max_chip_labels = []

        for ite, batch in enumerate(tq):
            tq.set_description('Datset: {}'.format(self.loader_conf['file_path']))

            cur_state = self.process_batch(batch, 'test')
            batch_pred_prob = cur_state['else|pred_prob'].cpu()
            batch_label = cur_state['else|label'].cpu()
            batch_chip_len = cur_state['else|chip_len'].cpu()
            batch_dec_out = cur_state['else|dec_out'].cpu()
            batch_acc_list = cur_state['else|acc_list'].tolist()

            B, max_len, num_classes = batch_pred_prob.shape

            temp = []
            temp2 = []
            for i in range(B):
                for j in range(max_len):
                    if j == max_len-1 or batch_pred_prob[i, j, :].max(dim=-1)[1] == num_classes - 1:
                        if j == 0:
                            temp.append(batch_pred_prob[i, 0:1, :])
                            temp2.append(batch_dec_out[i, 0:1, :])
                        else:
                            temp.append(batch_pred_prob[i, :j, :])
                            temp2.append(batch_dec_out[i, :j, :])
                        break

            epoch_pred_prob += temp
            epoch_dec_out += temp2
            epoch_labels += [x for x in batch_label]
            epoch_chip_len += [x for x in batch_chip_len]
            epoch_acc_list += batch_acc_list

            self.summary_write_state(cur_state, self.global_step, 'test')
            self.global_step += 1

        for i in range(len(epoch_dec_out)):
            max_chip_idx = epoch_pred_prob[i].max(dim=-1)[0].min(dim=-1)[1].item()
            max_chip_feats = epoch_dec_out[i][max_chip_idx:max_chip_idx+1]
            epoch_max_chip_feats.append(max_chip_feats)
            epoch_max_chip_labels.append(epoch_labels[i][max_chip_idx:max_chip_idx+1])


        base_dir = self.solver_conf['checkpoints_dir']
        prob_conf = self.solver_conf['prob_conf']
        foreign_method = prob_conf['foreign_method'] if 'foreign_method' in prob_conf else 'normal'
        weibull_model_path = os.path.join(base_dir, 'weibull_models.pkl')
        weibull_model_vis_dir = os.path.join(base_dir, 'weibull_models_vis')
        feats_type = prob_conf['weibull_feats_type'] if 'weibull_feats_type' in prob_conf else 'all_chips'

        if foreign_method == 'normal':
            out = metric.cal_tpr_from_seq(epoch_pred_prob, epoch_labels,
                                          foreign_prob_type=self.solver_conf['foreign_prob_type'],
                                          need_hist=True)

            temp = {'image|hist': visualizer.np2tensor(out['hist'])}
            self.summary_write_state(temp, self.global_step + 1, 'test')

        elif foreign_method == 'openmax':

            if feats_type == 'all_chips':
                out_state = metric.cal_tpr_by_openmax(epoch_dec_out, epoch_labels,
                                                      num_classes, epoch_acc_list,
                                                      weibull_model_path)
            elif feats_type == 'max_chip':
                out_state = metric.cal_tpr_by_openmax(epoch_max_chip_feats, epoch_labels,
                                                      num_classes, epoch_acc_list,
                                                      weibull_model_path)


            weibull_models = out_state['weibull_models']
            dis_per_class = out_state['dis_per_class']
            outlier_probs = out_state['outlier_probs']

            visualizer.vis_weibull_models(weibull_models, dis_per_class,
                                          save_dir=weibull_model_vis_dir+'_dis')
            visualizer.vis_pos_neg(outlier_probs, save_dir=weibull_model_vis_dir + '_prob')

        elif foreign_method == 'get_weibull_model':

            if feats_type == 'all_chips':
                res = metric.get_weibull_models(epoch_dec_out, epoch_labels, num_classes, epoch_acc_list)
            elif feats_type == 'max_chip':
                res = metric.get_weibull_models(epoch_max_chip_feats, epoch_max_chip_labels, num_classes, epoch_acc_list)

            weibull_models = res['weibull_models']
            mean_zs = res['mean_zs']
            dis_zs = res['dis_zs']

            with open(weibull_model_path, 'wb') as f:
                pickle.dump([weibull_models, mean_zs], f)

            visualizer.vis_weibull_models(weibull_models, [dis_zs], save_dir=weibull_model_vis_dir)


class TFMChipModel(nn.Module):
    def __init__(self, net_conf):
        super(TFMChipModel, self).__init__()

        self.net_conf = net_conf
        self.use_unc = net_conf['use_unc'] if 'use_unc' in net_conf else False
        self.label_type = net_conf['label_type'] if 'label_type' in net_conf else 'individual'
        self.num_classes = net_conf['num_classes']
        self.temperature = net_conf['temperature'] if 'temperature' in net_conf else 1

        self.enc_net = components.get_enc_net(net_conf)
        self.cls_net = components.get_cls_net(net_conf) # map (B, H, C) -> (B, H, num_classes)
        self.tfm_net = components.get_transformer(net_conf)

        if self.use_unc:
            self.unc_net = components.get_unc_net(net_conf)

        self.mse_loss_fun = nn.MSELoss()
        self.l1_loss_fun = nn.SmoothL1Loss()
        self.cse_loss_fun = nn.CrossEntropyLoss()
        self.nll_loss_fun = nn.NLLLoss()

    def forward(self, tensors, phase):

        """ Feature extraction """

        state = {}

        image = tensors['image']
        label = tensors['label'] # (B, label_len)
        seg_label = tensors['seg_label']
        label_in = tensors['label_in']
        label_out = tensors['label_out']
        label_mask = label_out.view(-1).nonzero()[:, 0]

        features = self.enc_net(image) # (B, C, H, 1)

        B, C, H, W = features.shape
        label_len = label.shape[1]

        seq_features = features.squeeze(3).permute([0, 2, 1]).contiguous()

        if phase == 'train':
            out = self.tfm_net(seq_features, label_in, label_out)
            pred_prob_tfm = out['pred_prob_tfm']
            enc_att = out['enc_att']
            dec_out = out['dec_out']
            env_att = out['env_att'] if 'env_att' in out else None

        else:
            # Inference stage, decode one by one
            # assert B == 1
            y = torch.ones(B, 1).fill_(self.num_classes-2).long().cuda()
            for i in range(label_len-1):
                out = self.tfm_net(seq_features, y, None, phase='test')

                pred_prob_tfm = out['pred_prob_tfm']
                enc_att = out['enc_att']
                dec_out = out['dec_out']
                env_att = out['env_att'] if 'env_att' in out else None

                cur_y = pred_prob_tfm[:, -1:, :].max(dim=-1)[1]
                y = torch.cat([y, cur_y], dim=1)
                temp = (y == self.num_classes-1).max(dim=1)[0]

                if temp.sum().item() == B:
                    if i == 0:
                        temp = torch.zeros(B, 1, self.num_classes).cuda()
                        temp[:, :, self.num_classes-3] = 1
                        pred_prob_tfm = torch.cat([temp, pred_prob_tfm], dim=1)
                    break

            # pred_prob_tfm = pred_prob_tfm[:, :-1, :]
        state['else|pred_prob_tfm'] = pred_prob_tfm.detach()
        state['else|enc_att'] = enc_att[0].detach() # (B, num_heads, label_len, H)
        state['else|dec_out'] = dec_out.detach()
        if env_att is not None:
            state['else|env_att'] = env_att.detach()


        if self.use_unc:
            confidence = self.unc_net(dec_out)
            state['else|confidence'] = confidence.detach()

        """ Loss Calculation """
        loss_total = None
        if phase == 'train':

            loss_w = self.net_conf['loss_weight']

            """ transformer classification loss """
            if self.use_unc:
                confidence = confidence.view(B, label_len, 1)
                one_hot = torch.zeros(pred_prob_tfm.size()).cuda()
                one_hot.scatter_(2, label.view(label.size(0), label.size(1), 1), 1.0)

                prob_tfm = pred_prob_tfm * confidence + one_hot * (1 - confidence)
            else:
                prob_tfm = pred_prob_tfm

            prob_tfm = torch.log(pred_prob_tfm)

            loss_cls_tfm = self.nll_loss_fun(prob_tfm.view(B*label_len, -1)[label_mask, :],
                                             label_out.view(-1)[label_mask])
            state['scalar|loss_cls_tfm'] = loss_cls_tfm.detach()


            """ uncertainty loss """
            loss_unc = 0
            if self.use_unc:
                loss_unc = -torch.log(confidence).mean()

                state['scalar|loss_unc'] = loss_unc.detach()

            """ total loss """
            loss_total = sum([x * loss_w[i] for i, x in enumerate([loss_unc,
                                                                   loss_cls_tfm])])
            state['scalar|loss'] = loss_total.detach()

        return loss_total, state

