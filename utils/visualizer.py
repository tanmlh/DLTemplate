import numpy as np
import torch
import torch.nn.functional as F
import cv2
import pdb
import matplotlib.pyplot as plt
import random
import scipy.stats as stats
import os

def np2tensor(np_img):
    H, W, C = np_img.shape
    img = torch.tensor(np_img).permute([2, 0, 1])
    return img

def de_normalize(image):
    B, C, H, W = image.shape

    mean = torch.tensor([0.485, 0.456, 0.406]).unsqueeze(1).unsqueeze(1).unsqueeze(0)
    std = torch.tensor([0.229, 0.224, 0.225]).unsqueeze(1).unsqueeze(1).unsqueeze(0)

    image = image * std + mean

    return image

def merge_list(img_list):
    """
    img_list: list of tensor with shape (C, H, W), where only W varies
    """
    img_W_list = [img.shape[2] for img in img_list]
    img_W = max(img_W_list)
    res_list = []

    for img in img_list:
        if img.shape[2] != img_W:
            pad_img = np.pad(img, ((0, 0), (0, 0), (0, img_W - img.shape[2])), 'constant', constant_values = 0)
            res_list.append(torch.from_numpy(pad_img))
        else:
            res_list.append(img)

    return torch.cat(res_list, dim=1)

def split_batch(image):
    """
    Input:
        image: (B, 3, H, W)

    Output:
        tiled_image: (3, B * H, W)
    """
    B, C, H, W = image.shape
    img_list = []
    for i in range(min(B, 16)):
        img_list.append(image[i])
        img_list.append(torch.zeros(C, H, 5))

    return torch.cat(img_list, dim=2)

def concat(image1, image2):
    """
    Input:
        image1: (3, H, W)
        image2: (3, H, W)
    """
    return torch.cat([image1, image2], dim=2)

def get_image_enc_att(image, enc_att, downsample=1):
    B, num_heads, label_len, h = enc_att.shape
    B, C, H, W = image.shape

    assert h * downsample == H
    assert C == 3

    w = int(W / downsample)

    att = enc_att.mean(dim=1)
    img = de_normalize(image)

    temp = att.view(B, label_len, 1, h, 1).repeat(1, 1, 3, 1, w).contiguous()
    temp = temp.view(B * label_len, 3, h, w)
    temp = F.interpolate(temp, size=(H, W)).view(B, label_len, 3, H, W)
    temp = temp * 0.2 / temp.mean()

    temp_img = img.view(B, 1, 3, H, W).repeat(1, label_len, 1, 1, 1)


    merged_img = (temp  + temp_img).clamp(0, 1)
    img_list = []
    for i in range(min(B, 16)):
        img_list.append(merged_img[i])

    temp = torch.cat(img_list, dim=2) # (label_len, 3, B * H, W)

    img_list = []
    for i in range(min(label_len, 16)):
        img_list.append(temp[i])

    temp = torch.cat(img_list, dim=2)

    return temp

def get_image_label_seg(image, label_seg, downsample=1):
    image = de_normalize(image)
    B, C, H, W = image.shape
    num_color = 10
    range_list = list(range(num_color))
    cmap = plt.cm.prism(range_list)[:, :3] # (label_len, 3)
    cmap = torch.from_numpy(cmap).float()
    color_mask = torch.zeros(B, C, H, W)

    for i in range(B):
        color_idx = 0
        for j in range(W):
            if label_seg[i, j // downsample].item() == 1:
                color_pad = cmap[color_idx].unsqueeze(1)
                color_pad = color_pad.repeat(1, H)
                color_mask[i, :, :, j] = color_pad
                color_idx = (color_idx + 1) % num_color

    vis_img = (image * 0.3 + color_mask * 0.7).clamp(0, 1)
    return split_batch(vis_img)

def get_image_pos_range(image, pos_range):
    # point type
    if pos_range.shape[-1] == 2:
        pos_left = pos_range[:, :, 0:1]
        pos_right = pos_range[:, :, 1:]

        # pdb.set_trace()
        cat_left = torch.cat([torch.zeros_like(pos_left), pos_left], dim=2)
        cat_right = torch.cat([torch.zeros_like(pos_right), pos_right], dim=2)

        cat_all = torch.cat([cat_left, cat_right], dim=1)

        return get_image_cen_pos(image, cat_all)

    # circle type
    if pos_range.shape[-1] == 4:
        image = de_normalize(image)
        B, C, H, W = image.shape

        radius = pos_range[:, :, 3]
        cen_pos = pos_range[:, :, :2]

        center_pos_y, center_pos_x = np.where(np.ones((H, W)) > 0) 
        center_pos_y = np.expand_dims(center_pos_y, 0)
        center_pos_x = np.expand_dims(center_pos_x, 0)
        center_pos_y = np.expand_dims(center_pos_y, 2) # (1, H*W, 1)
        center_pos_x = np.expand_dims(center_pos_x, 2)

        bb_center_x = np.expand_dims(cen_pos[:, :, 0], 1) # (B, 1, label_len)
        bb_center_y = np.expand_dims(cen_pos[:, :, 1], 1)

        bb_center_x = (bb_center_x + 1) / 2 * W
        bb_center_y = (bb_center_y + 1) / 2 * H

        ord_dis = (center_pos_x - bb_center_x) ** 2 + (center_pos_y - bb_center_y) ** 2 # (B, H*W, label_len)
        ord_dis = torch.tensor(ord_dis).float()
        ring = torch.abs((ord_dis ** 0.5) - radius.unsqueeze(1)) < 0.3
        ring = ((ord_dis ** 0.5) - radius.unsqueeze(1)) < 0.01
        ring = ring.float().unsqueeze(1)

        vis_image = torch.clamp(image + ring, 0, 1)

        return split_batch(vis_image)


def get_image_cen_pos(image, cen_pos, label=None, sigma=1):
    # cen_pos: (B, label_len, 2)
    # lens: (B,)
    
    image = de_normalize(image)
    B, C, H, W = image.shape
    label_len = cen_pos.shape[1]

    sum_image = image.sum(dim=[1, 2])
    # real_W = (sum_image>1e-3).argmax(dim=1).cpu().unsqueeze(1).unsqueeze(2)+1
    real_W = W

    if label is not None:
        temp = label.view(B * label_len)
        label_mask = temp.nonzero()[:, 0]
        cen_pos = cen_pos.view(B*label_len, 2)
        temp = torch.ones_like(cen_pos) * 2
        temp[label_mask, :] = cen_pos[label_mask, :]
        cen_pos = temp.view(B, label_len, 2)
        cen_pos = cen_pos.cpu().numpy()

    center_pos_y, center_pos_x = np.where(np.ones((H, W)) > 0)
    center_pos_y = np.expand_dims(center_pos_y, 0)
    center_pos_x = np.expand_dims(center_pos_x, 0)
    center_pos_y = np.expand_dims(center_pos_y, 2) # (1, H*W, 1)
    center_pos_x = np.expand_dims(center_pos_x, 2)

    # bb_center_x = np.expand_dims(cen_pos[:, :, 0], 1) # (B, 1, label_len)
    # bb_center_y = np.expand_dims(cen_pos[:, :, 1], 1)

    bb_center_x = np.expand_dims(cen_pos[:, :, 0], 1) # (B, 1, label_len)
    bb_center_y = np.expand_dims(cen_pos[:, :, 1], 1)


    # pdb.set_trace()
    # bb_center_x = (bb_center_x + 1) / 2 * real_W.cpu().numpy()
    bb_center_x = (bb_center_x + 1) / 2 * real_W
    bb_center_y = (bb_center_y + 1) / 2 * H

    ord_dis = (center_pos_x - bb_center_x) ** 2 + (center_pos_y - bb_center_y) ** 2
    ord_dis = np.exp(- ord_dis / (2 * sigma ** 2)) # (B, H*W, label_len)

    pos_map = torch.from_numpy(ord_dis).view(B*H*W, label_len).float()

    range_list = list(range(label_len))
    cmap = plt.cm.prism(range_list)[:, :3] # (label_len, 3)
    cmap = torch.from_numpy(cmap).float()
    temp = torch.matmul(pos_map, cmap)
    temp = temp.view(B, H, W, 3).permute([0, 3, 1, 2])
    vis_img = (image * 0.3 + temp * 0.7).clamp(0, 1)

    return split_batch(vis_img)


def get_image_prob_map(image):
    B, label_len, num_classes = image.shape
    minv = image.min()
    maxv = image.max()
    image = (image-minv) / (maxv - minv)
    image = image.unsqueeze(1)

    return split_batch(image)

def get_image_loc_map(image, loc_map, ratio=[0.6, 1]):
    image = de_normalize(image)
    B, H, W = loc_map.shape
    temp = loc_map.unsqueeze(1).repeat(1, 3, 1, 1)
    vis_img = torch.clamp((image * ratio[0] + temp * ratio[1]), 0, 1)

    return split_batch(vis_img)

def get_image_mul_map(image, ord_map, loc_map, label_len):
    image = de_normalize(image)

    if len(ord_map.shape) == 4:
        ord_map = ord_map.max(1)[1]

    B, H, W = ord_map.shape

    ord_map_one_hot = torch.zeros(B, label_len, H, W)
    ord_map_one_hot.scatter_(1, ord_map.unsqueeze(1).long(), torch.ones(B, label_len, H, W))

    range_list = list(range(label_len))
    cmap = plt.cm.prism(range_list)[:, :3] # (label_len, 3)
    # cmap[0] = np.array([0, 0, 0])

    temp = ord_map_one_hot.permute([0, 2, 3, 1]).contiguous() # (B, H, W, label_len)
    temp = temp.view(-1, label_len)
    temp = torch.matmul(temp, torch.from_numpy(cmap).float())
    temp = temp.view(B, H, W, 3)
    temp = temp * loc_map.unsqueeze(3)
    temp = temp.permute([0, 3, 1, 2])

    vis_img = (image * 0.3 + temp * 0.7).clamp(0, 1)

    return split_batch(vis_img)


def get_image_seg_map(image, seg_map, num_classes, trans_background=True):
    """
    Input:
        image: (B, 3, H, W)
        seg_map: (B, H, W)

    Output:
        vis_img: (3, B*H, W)
    """
    image = de_normalize(image)
    B, H, W = seg_map.shape

    seg_map_one_hot = torch.zeros(B, num_classes, H, W)
    seg_map_one_hot.scatter_(1, seg_map.unsqueeze(1).long(), torch.ones(B, num_classes, H, W))


    range_list = list(range(num_classes))
    # random.shuffle(range_list)
    cmap = plt.cm.prism(range_list)[:, :3] # (num_classes, 3)
    if trans_background:
        cmap[0] = np.array([0, 0, 0])

    temp = seg_map_one_hot.permute([0, 2, 3, 1]).contiguous() # (B, H, W, num_classes)
    temp = temp.view(-1, num_classes)
    temp = torch.matmul(temp, torch.from_numpy(cmap).float())
    temp = temp.view(B, H, W, 3)
    temp = temp.permute([0, 3, 1, 2])

    vis_img = (image * 0.7 + temp * 0.3)


    return split_batch(vis_img)

def get_image_ord_map(image, ord_map):
    B, label_len, H, W = ord_map.shape
    img_H = image.shape[2]

    ord_map = ord_map.max(1)[1]
    
    if H == 1:
        ord_map = ord_map.repeat([1, img_H, 1])

    return get_image_seg_map(image, ord_map, label_len, True)


def get_image_text(image, pred_str):
    """
    Input:
        image: (B, C, H, W)
        pred_str: [str] * B

    Output:
        visualized_img: (H*B, W, C)
    """
    # data = torch.clamp(data, 0, 1) * 255
    # data = (data + 0.5) * 255
    # data = data.numpy().astype(np.uint8)

    # outputs = outputs.transpose(0, 1)

    image = de_normalize(image)

    B, C, H, W = image.shape
    assert C == 1 or C == 3

    img_list = []
    label_list = []
    for i in range(min(B, 16)):

        cur_str = pred_str[i]
        # visual_data = np.transpose(data[i], [1, 2, 0])
        visual_data = image[i]
        if C == 1:
            visual_data = visual_data[0, :, :]

        text_img = np.zeros((H, W))
        pos = (16, 16)
        temp = cur_str.split(',')
        for c in temp:
            text_img = cv2.putText(text_img, c, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                   (255, 255, 255), 1, bottomLeftOrigin=False)
            pos = (pos[0], pos[1] + 16)
        text_img = torch.from_numpy(text_img).float() / 255.0

        if C == 3:
            text_img = text_img.unsqueeze(0).repeat(3, 1, 1)

        visual_data = torch.cat([visual_data, text_img], dim=2)
        img_list.append(visual_data)
        img_list.append(torch.zeros(C, 10, W*2))

    img_visual = torch.cat(img_list, dim=1)

    return img_visual

def vis_weibull_models(weibull_models, dis_per_class, save_dir):

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for i in range(len(weibull_models)):
        if weibull_models[i] != []:
            x = np.linspace(0, 1, 1000)
            plt.plot(x, stats.exponweib.pdf(x, *weibull_models[i]))

            for j in range(len(dis_per_class)):
                plt.hist(dis_per_class[j][i], 30, density=True, alpha=0.5)

            plt.savefig(os.path.join(save_dir, 'class_{}.png'.format(i)))
            plt.close('all')
def vis_weibull_models(outlier_probs, save_dir):

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    x = np.linspace(0, 1, 1000)

    for j in range(len(outlier_probs)):
        plt.hist(outlier_probs[j][i], 30, density=True, alpha=0.5)

    plt.savefig(os.path.join(save_dir, 'outlier_probs.png'))
    plt.close('all')
