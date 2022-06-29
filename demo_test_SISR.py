#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2020-04-07 11:58:30

import argparse
import cv2
from collections import OrderedDict
import numpy as np
from pathlib import Path
from scipy.io import loadmat
from networks.VIRNet import VIRAttResUNetSR
from skimage import img_as_float32, img_as_ubyte

import torch

from utils import util_image
from utils import util_sisr

parser = argparse.ArgumentParser()
# trainning settings
parser.add_argument('--sf', default=4, type=int, metavar='PATH',
                                                         help="Downsampling Scale, (default: None)")
# for the case of noise free, we suggest set noise_level to be 0.1 for stale calculation.
parser.add_argument('--noise_level', default=2.55, type=float,
                                                 help="Noise level for Degradation, (default:2.55)")
args = parser.parse_args()

C = 3
# load the network
net = VIRAttResUNetSR(im_chn=3,
                      sigma_chn=1,
                      dep_K=8,
                      dep_S=5,
                      n_feat=[96,  160, 224],
                      n_resblocks=2,
                      noise_cond=True,
                      kernel_cond=True,
                      extra_mode='Input',
                      noise_avg=True).cuda()
state_path = str(Path('./model_zoo') / ('sisr_x'+str(args.sf)+'.pth'))
ckpt = OrderedDict({key[7:]:value for key, value in torch.load(state_path)['model_state_dict'].items()})
net.load_state_dict(ckpt, strict=True)
net.eval()

# read testing images
data = 'Set14'
im_path_list = sorted([x for x in Path('./test_data/Set14').glob('*.bmp')])

# loading the blur kernel
kernels_8 = loadmat('./test_data/kernels_SISR/kernels_8.mat')['kernels']

p = 21
sf = args.sf
for ind_kernel in range(7):
    if ind_kernel == 0:
        kernel = util_sisr.shifted_anisotropic_Gaussian(p, sf, (0.40*sf)**2, (0.40*sf)**2, 0, False)[0]
    elif ind_kernel == 1:
        kernel = util_sisr.shifted_anisotropic_Gaussian(p, sf, (0.60*sf)**2, (0.60*sf)**2, 0, False)[0]
    elif ind_kernel == 2:
        kernel = util_sisr.shifted_anisotropic_Gaussian(p, sf, (0.80*sf)**2, (0.80*sf)**2, 0, False)[0]
    elif ind_kernel == 3:
        kernel = util_sisr.shifted_anisotropic_Gaussian(p, sf, (0.4*sf)**2, (0.2*sf)**2, 0, False)[0]
    elif ind_kernel == 4:
        kernel = util_sisr.shifted_anisotropic_Gaussian(p, sf, (0.6*sf)**2, (0.3*sf)**2, 0.75*np.pi, False)[0]
    elif ind_kernel == 5:
        kernel = util_sisr.shifted_anisotropic_Gaussian(p, sf, (0.8*sf)**2, (0.4*sf)**2, 0.25*np.pi, False)[0]
    else:
        kernel = util_sisr.shifted_anisotropic_Gaussian(p, sf, (0.8*sf)**2, (0.4*sf)**2, 0.50*np.pi, False)[0]

    psnr_imgs_y = {}
    ssim_imgs_y = {}
    psnr_mean_y = ssim_mean_y = 0
    for im_path_iter in im_path_list:
        im_name = im_path_iter.stem
        im_gt = util_image.imread(im_path_iter, chn='rgb', dtype='uint8')
        im_gt = util_sisr.modcrop(im_gt, args.sf)
        if im_gt.ndim == 2:
            im_gt = np.stack([im_gt,]*3, axis=2)

        # degradation
        im_lr = util_sisr.degrade_virnet(img_as_float32(im_gt),
                                         kernel=kernel,
                                         sf=sf,
                                         nlevel=args.noise_level,
                                         qf=None,
                                         downsampler='Bicubic')  # h x w x c, float32
        inputs = torch.from_numpy(im_lr.transpose((2,0,1))[np.newaxis,]).type(torch.float32).cuda()
        with torch.set_grad_enabled(False):
            outputs, _, _ = net(inputs, args.sf)

        im_sr = img_as_ubyte(np.clip(outputs[0,].cpu().numpy().transpose((1,2,0)), 0.0, 1.0))
        im_sr_y = util_image.rgb2ycbcr(im_sr, only_y=True)
        im_gt_y = util_image.rgb2ycbcr(im_gt, only_y=True)
        psnr_iter_y = util_image.calculate_psnr(im_sr_y, im_gt_y, sf**2)
        psnr_mean_y += psnr_iter_y
        ssim_iter_y = util_image.calculate_ssim(im_sr_y, im_gt_y, sf**2)
        ssim_mean_y += ssim_iter_y

    ssim_mean_y /= len(im_path_list)
    psnr_mean_y /= len(im_path_list)
    log_str = 'Dataset: {:>8s}, Kernel: {:d}, PSNR: {:5.2f}, SSIM: {:6.4f}'
    print(log_str.format(data, ind_kernel+1, psnr_mean_y, ssim_mean_y))

