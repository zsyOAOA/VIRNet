#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2019-09-01 19:35:06

from math import log

import torch
import torch.nn as nn
import torch.nn.functional as F

from .DnCNN import DnCNN
from .AttResUNet import AttResUNet
from .KNet import KernelNet as KNet

log_max = log(1e2)
log_min = log(1e-10)

class VIRAttResUNet(nn.Module):
    '''
    For Denoising task with UNet denoiser.
    '''
    def __init__(self, im_chn,
                 sigma_chn=3,
                 n_feat=[64, 128, 192],
                 dep_S=5,
                 n_resblocks=2,
                 noise_cond=True,
                 extra_mode='Input',
                 noise_avg=False):
        super(VIRAttResUNet, self).__init__()
        self.SNet = DnCNN(im_chn, sigma_chn, dep=dep_S, noise_avg=noise_avg)

        self.noise_cond = noise_cond
        extra_chn = sigma_chn if noise_cond else 0
        self.RNet = AttResUNet(im_chn,
                               extra_chn=extra_chn,
                               out_chn=im_chn,
                               n_feat=n_feat,
                               n_resblocks=n_resblocks,
                               extra_mode=extra_mode)

    def forward(self, x):
        sigma = torch.exp(torch.clamp(self.SNet(x), min=log_min, max=log_max))
        extra_maps = sigma.sqrt() if self.noise_cond else None
        mu = self.RNet(x, extra_maps)
        return mu, sigma

class VIRAttResUNetSR(nn.Module):
    '''
    For Denoising task with UNet denoiser.
    '''
    def __init__(self, im_chn,
                 sigma_chn=1,
                 kernel_chn=3,
                 n_feat=[64, 128, 192],
                 dep_S=5,
                 dep_K=8,
                 noise_cond=True,
                 kernel_cond=True,
                 n_resblocks=1,
                 extra_mode='Down',
                 noise_avg=True):
        super(VIRAttResUNetSR, self).__init__()
        self.noise_cond = noise_cond
        self.noise_avg = noise_avg
        self.kernel_cond = kernel_cond

        extra_chn = 0
        if self.kernel_cond: extra_chn += kernel_chn
        if self.noise_cond: extra_chn += sigma_chn
        self.SNet = DnCNN(im_chn, sigma_chn, dep=dep_S, noise_avg=noise_avg)
        self.KNet = KNet(im_chn, kernel_chn, num_blocks=dep_K)
        self.RNet = AttResUNet(im_chn,
                               extra_chn=extra_chn,
                               out_chn=im_chn,
                               n_feat=n_feat,
                               n_resblocks=n_resblocks,
                               extra_mode=extra_mode)

    def forward(self, x, sf):
        sigma = torch.exp(torch.clamp(self.SNet(x), min=log_min, max=log_max))  # N x [] x 1 x 1
        kinfo_est = self.KNet(x)    # N x [] x 1 x 1
        x_up = F.interpolate(x, scale_factor=sf, mode='nearest')
        h_up, w_up = x_up.shape[-2:]
        if not self.noise_cond and not self.kernel_cond:
            extra_maps = None
        else:
            extra_temp = []
            if self.kernel_cond: extra_temp.append(kinfo_est.repeat(1,1,h_up,w_up))
            if self.noise_cond:
                if self.noise_avg:
                    extra_temp.append(sigma.sqrt().repeat(1,1,h_up,w_up))
                else:
                    extra_temp.append(F.interpolate(sigma.sqrt(), scale_factor=sf, mode='nearest'))
            extra_maps = torch.cat(extra_temp, 1)     # n x [] x h x w
        mu = self.RNet(x_up, extra_maps)
        return mu, kinfo_est.squeeze(-1).squeeze(-1), sigma

