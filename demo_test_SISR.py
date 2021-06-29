#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2020-04-07 11:58:30

import argparse
import cv2
import numpy as np
from pathlib import Path
from scipy.io import loadmat
from networks.VIRNet import VIRNetSISR
from skimage import img_as_float32, img_as_ubyte
from utils import rgb2ycbcr, modcrop, degradeSingleNumpy, calculate_psnr, calculate_ssim

import torch

parser = argparse.ArgumentParser()
# trainning settings
parser.add_argument('--scale', default=2, type=int, metavar='PATH',
                                                         help="Downsampling Scale, (default: None)")
parser.add_argument('--noise_level', default=2.55, type=float,
                                                 help="Noise level for Degradation, (default:2.55)")
args = parser.parse_args()

C = 3
# load the network
net = VIRNetSISR(C, wf=64, dep_U=3, dep_S=5).cuda()
state_path = str(Path('./model_zoo') / ('model_sisr_x'+str(args.scale)+'.pt'))
net.load_state_dict(torch.load(state_path))

# read testing images
data = 'Set14'
im_path_list = sorted([x for x in Path('./test_data/Set14').glob('*.bmp')])

# loading the blur kernel
kernels_8 = loadmat('./test_data/kernels_SISR/kernels_8.mat')['kernels']

for ind_kernel in range(8):
    kernel = kernels_8[:, :, ind_kernel]
    psnr_imgs_y = {}
    ssim_imgs_y = {}
    psnr_mean_y = ssim_mean_y = 0
    for im_path_iter in im_path_list:
        im_name = im_path_iter.stem
        gt_im_iter = cv2.imread(str(im_path_iter), cv2.IMREAD_COLOR)[:, :, ::-1]
        gt_im_iter = modcrop(gt_im_iter, args.scale)
        H, W = gt_im_iter.shape[:2]

        degrade_im_iter = degradeSingleNumpy(gt_im_iter, kernel, args.scale, downsampler='direct')
        degrade_im_iter = img_as_float32(degrade_im_iter)
        H_small, W_small = degrade_im_iter.shape[:2]
        if args.noise_level > 0:
            degrade_im_iter += np.random.randn(H, W, C) * (args.noise_level / 255.0)

        inputs = torch.from_numpy(degrade_im_iter.transpose((2,0,1))[np.newaxis,]).type(torch.float32).cuda()
        kernel_temp = torch.from_numpy(kernel[np.newaxis, np.newaxis,]).type(torch.float32).cuda()
        with torch.set_grad_enabled(False):
            outputs, _ = net(inputs, kernel_temp, args.scale,  pad=True)
            outputs = outputs[:, :3, :H, :W].squeeze()

        desisr_im_iter = img_as_ubyte(np.clip(outputs.cpu().numpy().transpose((1,2,0)), 0.0, 1.0))
        desisr_im_iter_y = rgb2ycbcr(desisr_im_iter, only_y=True)
        gt_im_iter_y = rgb2ycbcr(gt_im_iter, only_y=True)
        psnr_iter_y = calculate_psnr(desisr_im_iter_y, gt_im_iter_y, args.scale**2)
        psnr_mean_y += psnr_iter_y
        ssim_iter_y = calculate_ssim(desisr_im_iter_y, gt_im_iter_y, args.scale**2)
        ssim_mean_y += ssim_iter_y

    ssim_mean_y /= len(im_path_list)
    psnr_mean_y /= len(im_path_list)
    log_str = 'Dataset: {:>8s}, Kernel: {:d}, PSNR: {:5.2f}, SSIM: {:6.4f}'
    print(log_str.format(data, ind_kernel+1, psnr_mean_y, ssim_mean_y))

