#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2020-04-07 11:58:30

import torch
import argparse
import numpy as np
from pathlib import Path
from networks.VIRNet import VIRNetU
from skimage import img_as_float32, img_as_ubyte
import cv2
from utils import calculate_psnr, calculate_ssim, generate_gauss_kernel_mix, peaks, sincos_kernel

# load the network
net = VIRNetU(3, wf=64, dep_U=3, dep_S=5).cuda()
# state_path = '/home/oa/Desktop/VIRNet-Exp/Denoising/Simulation/NIID/VIRNet/wf64D3S5_eps6_Leaky/VDNPlus_Simulation_eps6_wf64D3S8/model_state_100.pt'
# net.load_state_dict(torch.load(state_path))
net.load_state_dict(torch.load('./model_zoo/model_denoising_simulate.pt'))

base_path = Path('./test_data')
# read testing images
datasets = ['CBSD68', 'Kodak24', 'McMaster']
exts = ['png', 'png', 'tif']

sigma_max = 75/255.0
sigma_min = 10/255.0
var_maps = [peaks(256), sincos_kernel(), generate_gauss_kernel_mix(256, 256)]

for data_name, ext_name in zip(datasets, exts):
    data_path = base_path / data_name
    im_list = sorted(list(data_path.glob('*.'+ext_name)))
    for jj, sigma in enumerate(var_maps):
        sigma = sigma_min + (sigma-sigma.min())/(sigma.max()-sigma.min()) * (sigma_max-sigma_min)
        mean_psrn = 0
        mean_ssim = 0
        for im_path in im_list:
            im_name = im_path.stem
            im_gt = cv2.imread(str(im_path), flags=cv2.IMREAD_COLOR)[:, :, ::-1]
            im_gt_float = img_as_float32(im_gt)
            H, W, C = im_gt.shape
            sigma_iter = cv2.resize(sigma, (W, H)).astype(np.float32) # H x W
            noise = np.random.randn(H, W, C) * np.expand_dims(sigma_iter, 2) # H x W x C
            im_noisy = im_gt_float[:H, :W,] + noise.astype(np.float32)
            inputs = torch.from_numpy(im_noisy.transpose(2,0,1)[np.newaxis,]).cuda()
            with torch.set_grad_enabled(False):
                phi_Z = net(inputs, 'test')
                outputs = phi_Z[:, :3,].squeeze().cpu().numpy().transpose([1,2,0])
                denoised_im_iter = img_as_ubyte(np.clip(outputs, 0.0, 1.0))
            psnr_iter = calculate_psnr(denoised_im_iter, im_gt[:H, :W,])
            mean_psrn += psnr_iter
            ssim_iter = calculate_ssim(denoised_im_iter, im_gt[:H, :W,])
            mean_ssim += ssim_iter
            # print('Dataset: {:s}, Image: {:s}, case: {:d}, PSNR: {:5.2f}, SSIM: {:6.4f}'.format(data_name, im_name,
                                                                                          # jj+1, psnr_iter, ssim_iter))
        mean_psrn /= len(im_list)
        mean_ssim /= len(im_list)
        print('Dataset: {:s}, case: {:d}, PSNR: {:5.2f}, SSIM: {:6.4f}'.format(data_name, jj+1, mean_psrn, mean_ssim))

