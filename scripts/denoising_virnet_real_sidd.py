#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2019-05-16 16:20:01

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch
import logging
import argparse
import numpy as np
import scipy.io as sio
from collections import OrderedDict
from skimage import img_as_float, img_as_ubyte

from thop import profile
from networks.VIRNet import VIRAttResUNet

from utils import util_net
from utils import util_image
from utils import util_common
from utils.util_opts import str2bool

parser = argparse.ArgumentParser()
parser.add_argument(
    '--test',
    type=str2bool,
    default='False',
    help="Testing or Validataion.",
    )
parser.add_argument(
    '--ckpt_path',
    default='model_zoo/virnet_denoising_real.pth',
    type=str,
    help="Path to the checkpoint!",
    )
parser.add_argument(
    '--flip',
    default='True',
    type=str2bool,
    help="Whether to flip during testing (default: True)",
    )
parser.add_argument(
    '--sidd_dir',
    default='',
    type=str,
    help="Folder to save the SIDD data set.",
    )
parser.add_argument(
    '--save_dir',
    type=str,
    default='results_dnd',
    help="Path to save the denoising results.")

args = parser.parse_args()

# logging settings
test_str = 'test' if args.test else 'val'
flip_str = 'flip' if args.flip else 'noflip'
log_name = f"sidd_{test_str}_{flip_str}.log"
util_common.mkdir(args.save_dir, delete=True, parents=True)
log_path = Path(args.save_dir) / log_name
if log_path.exists():
    log_path.unlink()
logger = util_common.make_log(str(log_path), file_level=logging.INFO, stream_level=logging.INFO)

logger.info('==========================Configurations===========================')
for key in vars(args):
    value = getattr(args, key)
    logger.info('{:12s}: {:s}'.format(key, str(value)))
logger.info('===================================================================')

# load the pretrained model
logger.info(f'Loading the Model from {args.ckpt_path}...')
net = VIRAttResUNet(im_chn=3,
                    sigma_chn=3,
                    n_feat=[96, 160, 224, 288],
                    dep_S=8,
                    n_resblocks=3,
                    noise_cond=True,
                    extra_mode='Input',
                    noise_avg=False).cuda()
ckpt = torch.load(args.ckpt_path)['model_state_dict']
net.load_state_dict(OrderedDict({key[7:]:value for key, value in ckpt.items()}), strict=True)
net.eval()

logger.info('----------------------------Model Analysis-------------------------------')
inputs1 = torch.randn(1, 3, 512, 512).cuda()
flops1, _ = profile(net, inputs=(inputs1, ))
num_params = util_net.calculate_parameters(net)
logger.info('Number of parameters: {:.2f}M'.format(num_params / 1000**2))
logger.info('FLOPs for 512: {:.2f}G'.format(flops1 / 1000**3))
logger.info('-------------------------------------------------------------------------')


logger.info('Loading the data')
if args.test:
    noisy_path = Path(args.sidd_dir) / 'BenchmarkNoisyBlocksSrgb.mat'
    data_noisy = sio.loadmat(noisy_path)['BenchmarkNoisyBlocksSrgb']
else:
    noisy_path = Path(args.sidd_dir) / 'ValidationNoisyBlocksSrgb.mat'
    data_noisy = sio.loadmat(noisy_path)['ValidationNoisyBlocksSrgb']
    gt_path = Path(args.sidd_dir) / 'ValidationGtBlocksSrgb.mat'
    data_gt = sio.loadmat(gt_path)['ValidationGtBlocksSrgb']
num_im, num_blk, h, w, c = data_noisy.shape

denoised_res = np.zeros_like(data_noisy)
logger.info('Begin Testing')
psnr_all = ssim_all = 0
num_test = 0
total_time = 0
for ii in range(num_im):
    for jj in range(num_blk):
        num_test += 1
        if num_test % 100 == 0:
            logger.info('The {:d} images'.format(num_test))
        if not args.test:
            im_gt = data_gt[ii, jj,]
        if args.flip:
            im_denoise = np.zeros((h, w, c), dtype=np.float32)
            for flag in range(8):
                im_noisy = img_as_float(util_image.data_aug_np(data_noisy[ii, jj,], flag))
                im_noisy = im_noisy.transpose((2,0,1))[np.newaxis,]
                im_noisy = torch.from_numpy(im_noisy).type(torch.float32).cuda()
                with torch.autograd.set_grad_enabled(False):
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                    mu = net(im_noisy)[0]
                    end.record()
                    torch.cuda.synchronize()
                    total_time += start.elapsed_time(end)/1000
                    im_denoise_flag = mu.cpu().numpy().squeeze()
                im_denoise += util_image.inverse_data_aug_np(im_denoise_flag.transpose((1,2,0)), flag)
            im_denoise /= 8
        else:
            im_noisy = img_as_float(data_noisy[ii, jj,])
            im_noisy = im_noisy.transpose((2,0,1))[np.newaxis,]
            im_noisy = torch.from_numpy(im_noisy).type(torch.float32).cuda()
            with torch.autograd.set_grad_enabled(False):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                mu = net(im_noisy)[0]
                end.record()
                torch.cuda.synchronize()
                total_time += start.elapsed_time(end)/1000
            im_denoise = mu.cpu().numpy().squeeze().transpose((1,2,0))
        im_denoise = img_as_ubyte(im_denoise.clip(0,1))
        denoised_res[ii, jj,] = im_denoise
        if not args.test:
            psnr_all += util_image.calculate_psnr(im_gt, im_denoise, border=0, ycbcr=False)
            ssim_all += util_image.calculate_ssim(im_gt, im_denoise, border=0, ycbcr=False)

megatime = total_time * 1024 * 1024 / (num_im*num_blk*256*256)
if not args.test:
    psnr_mean = psnr_all / num_test
    ssim_mean = ssim_all / num_test
    logger.info('PSNR={:.4f}, SSIM={:.4f}'.format(psnr_mean, ssim_mean))
if args.save_dir:
    mat_name = 'sidd_{:s}_{:s}.mat'.format(test_str, flip_str)
    sio.savemat(str(Path(args.save_dir) / mat_name), {'denoised_res': denoised_res,
                                                      'megatime':megatime})

