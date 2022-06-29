#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2019-05-16 16:20:01

import torch
import argparse
import numpy as np
from scipy.io import loadmat
from collections import OrderedDict
from networks.VIRNet import VIRAttResUNet

from utils import util_image
from utils.util_opts import str2bool

parser = argparse.ArgumentParser()
# trainning settings
parser.add_argument('--flip', type=str2bool, default='True', help="Data augmentation")
args = parser.parse_args()

print('Load the testing data')
im_noisy = loadmat('./test_data/DND/1.mat')['InoisySRGB']

C = 3
# load the pretrained model
print('Loading the Model')
net = VIRAttResUNet(im_chn=3,
                    sigma_chn=3,
                    n_feat=[96, 160, 224, 288],
                    dep_S=8,
                    n_resblocks=3,
                    extra_mode='Input',
                    noise_cond=True,
                    noise_avg=False).cuda()
ckpt_path = './model_zoo/denoising_real.pth'
ckpt = OrderedDict({key[7:]:value for key, value in torch.load(ckpt_path)['model_state_dict'].items()})
net.load_state_dict(ckpt, strict=True)
net.eval()

if args.flip:
    im_denoise = np.zeros((h, w, c), dtype=np.float32)
    for flag in range(8):
        im_noisy_in = util_image.data_aug_np(im_noisy, flag)
        im_noisy_in = im_noisy_in.transpose((2,0,1))[np.newaxis,]
        im_noisy_in = torch.from_numpy(im_noisy_in).type(torch.float32).cuda()
        with torch.autograd.set_grad_enabled(False):
            mu = net(im_noisy_in)[0]
            im_denoise_flag = mu.cpu().numpy().squeeze()
        im_denoise += util_image.inverse_data_aug_np(im_denoise_flag.transpose((1,2,0)), flag)
    im_denoise /= 8
else:
    im_noisy_in = im_noisy.transpose((2,0,1))[np.newaxis,]
    im_noisy_in = torch.from_numpy(im_noisy_in).type(torch.float32).cuda()
    with torch.autograd.set_grad_enabled(False):
        mu = net(im_noisy_in)[0]
    im_denoise = mu.cpu().numpy().squeeze().transpose((1,2,0))

im_denoise = np.clip(im_denoise, 0.0, 1.0)
im_all = np.concatenate([im_noisy, im_denoise], 1)
util_image.imshow(im_all)

