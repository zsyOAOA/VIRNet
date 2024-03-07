#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2020-04-07 11:58:30

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parents[1]))

import os
import cv2
import torch
import pickle
import logging
import argparse
import numpy as np
from collections import OrderedDict
from skimage import img_as_float32, img_as_ubyte

from utils import util_net
from utils import util_image
from utils import util_common
from utils import util_denoising
from utils.util_opts import str2bool

from thop import profile
from networks.VIRNet import VIRAttResUNet

parser = argparse.ArgumentParser()
# trainning settings
parser.add_argument(
    '--ckpt_path',
    type=str,
    default='model_zoo/virnet_denoising_syn.pth',
    metavar='PATH',
    help="Path to save the checkpoint!",
    )
parser.add_argument(
    '--noise_type',
    type=str,
    default='niid',
    choices=['niid', 'iid'],
    help="Noise type: niid or iid",
    )
parser.add_argument('--save_dir', type=str, default='results', help="Saving results")
args = parser.parse_args()

# logging settings
log_name = 'virnet_denoising_{:s}.log'.format(args.noise_type.lower())
log_path = Path(args.save_dir) / log_name
util_common.mkdir(args.save_dir, delete=False, parents=True)
if log_path.exists():
    log_path.unlink()
logger = util_common.make_log(str(log_path), file_level=logging.INFO, stream_level=logging.INFO)

logger.info('==========================Configurations===========================')
for key in vars(args):
    value = getattr(args, key)
    logger.info('{:12s}: {:s}'.format(key, str(value)))
logger.info('===================================================================')

# load the network
net = VIRAttResUNet(
        im_chn=3,
        sigma_chn=1,
        n_feat=[96, 192, 288],
        dep_S=5,
        n_resblocks=3,
        noise_cond=True,
        extra_mode="Input",
        noise_avg=False,
        ).cuda()
ckpt = torch.load(args.ckpt_path)['model_state_dict']
try:
    net.load_state_dict(ckpt, strict=True)
except:
    net.load_state_dict(OrderedDict({key[7:]:value for key, value in ckpt.items()}), strict=True)

logger.info('----------------------------Model Analysis-------------------------------')
inputs1 = torch.randn(1, 3, 256, 256).cuda()
flops1, _ = profile(net, inputs=(inputs1, ))
inputs2 = torch.randn(1, 3, 512, 512).cuda()
flops2, _ = profile(net, inputs=(inputs2, ))
num_params = util_net.calculate_parameters(net)
logger.info('Number of parameters: {:.2f}M'.format(num_params / 1000**2))
logger.info('FLOPs for 256: {:.2f}G'.format(flops1 / 1000**3))
logger.info('FLOPs for 512: {:.2f}G'.format(flops2 / 1000**3))
logger.info('-------------------------------------------------------------------------')

logger.info('------------------------------Evaluation---------------------------------')
base_path = Path('./test_data')
# read testing images
datasets = ['CBSD68', 'McMaster']
exts = ['png', 'tif']

# Gaussain noise
rng = util_denoising.noise_generator()

# variance map settings
sigma_max = 75/255.0
sigma_min = 10/255.0
if args.noise_type.lower() == 'niid':
    var_maps = [util_denoising.peaks(256),
                util_denoising.sincos_kernel(),
                util_denoising.generate_gauss_kernel_mix(256, 256, rng)]
elif args.noise_type.lower() == 'iid':
    var_maps = [15, 25, 50]
else:
    sys.exit('Please input corrected noise levels')

for data_name, ext_name in zip(datasets, exts):
    data_path = base_path / data_name
    im_list = sorted([str(x) for x in data_path.glob('*.'+ext_name)])
    for jj, sigma_base in enumerate(var_maps):
        denoised_imgs = {}
        noisy_imgs = {}
        denoised_psnr = {}
        denoised_ssim = {}
        mean_psrn = 0
        mean_ssim = 0
        if args.noise_type.lower() == 'niid':
            sigma_base = sigma_min + (sigma_base-sigma_base.min())/(sigma_base.max()-sigma_base.min()) * (sigma_max-sigma_min)
        for im_path in im_list:
            im_name = Path(im_path).stem
            im_gt = util_image.imread(im_path, chn='rgb', dtype='uint8')
            h, w = im_gt.shape[:2]
            if args.noise_type.lower() == 'niid':
                sigma = cv2.resize(sigma_base, (w, h), interpolation=cv2.INTER_NEAREST_EXACT).astype(np.float32) # H x W
            else:
                sigma = np.ones([h, w], dtype=np.float32) * (sigma_base / 255.)
            noise = rng.standard_normal(size=im_gt.shape) * sigma[:, :, np.newaxis]
            im_noisy = img_as_float32(im_gt) + noise.astype(np.float32)
            inputs = torch.from_numpy(im_noisy.transpose(2,0,1)[np.newaxis,]).cuda()
            with torch.set_grad_enabled(False):
                mu, _ = net(inputs)
                if isinstance(mu, list):
                    mu = mu[0]
                outputs = mu.squeeze(0).cpu().numpy().transpose([1,2,0])
                im_denoised = img_as_ubyte(np.clip(outputs, 0.0, 1.0))
            psnr_iter = util_image.calculate_psnr(im_denoised, im_gt, border=0, ycbcr=False)
            mean_psrn += psnr_iter
            ssim_iter = util_image.calculate_ssim(im_denoised, im_gt, border=0, ycbcr=False)
            mean_ssim += ssim_iter
            denoised_imgs['im_'+im_name] = im_denoised
            noisy_imgs['im_'+im_name] = img_as_ubyte(np.clip(im_noisy, 0.1, 1.0))
            denoised_psnr['im_'+im_name] = psnr_iter
            denoised_ssim['im_'+im_name] = ssim_iter
        mean_psrn /= len(im_list)
        denoised_psnr['mean'] = mean_psrn
        mean_ssim /= len(im_list)
        denoised_ssim['mean'] = mean_ssim
        if args.noise_type.lower() == 'niid':
            logger.info('Dataset: {:8s}, case: {:d}, PSNR: {:5.2f}, SSIM: {:6.4f}'.format(data_name,
                                                                        jj+1, mean_psrn, mean_ssim))
        else:
            logger.info('Dataset: {:8s}, sigma: {:d}, PSNR: {:5.2f}, SSIM: {:6.4f}'.format(data_name,
                                                                  sigma_base, mean_psrn, mean_ssim))
        if args.noise_type.lower() == 'niid':
            pkl_name = '{:s}_case{:d}.pkl'.format(data_name, jj+1)
        else:
            pkl_name = '{:s}_sigma{:d}.pkl'.format(data_name, sigma_base)
        pkl_path = Path(args.save_dir) / pkl_name
        if pkl_path.exists():
            pkl_path.unlink()
        with open(str(pkl_path), 'wb') as fpkl:
            pickle.dump({'denoised_imgs': denoised_imgs,
                         'noisy_imgs': noisy_imgs,
                         'denoised_psnr': denoised_psnr,
                         'denoised_ssim': denoised_ssim}, fpkl)

