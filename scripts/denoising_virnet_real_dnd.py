#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2019-05-20 14:54:56

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parents[1]))

import torch
import argparse

from collections import OrderedDict
from dnd_submission_py.dnd_denoise import denoise_srgb
from dnd_submission_py.pytorch_wrapper import pytorch_denoiser
from dnd_submission_py.bundle_submissions import bundle_submissions_srgb

from networks.VIRNet import VIRAttResUNet

from utils import util_common
from utils.util_opts import str2bool

parser = argparse.ArgumentParser(prog='DND Test', description='optional parameters for test')
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
    '--dnd_dir',
    default='',
    type=str,
    help="Folder to save the DND data set.",
    )
parser.add_argument(
    '--save_dir',
    type=str,
    default='results_dnd',
    help="Path to save the denoising results.")
args = parser.parse_args()

# load the pretrained model
print(f'Loading the Model from {args.ckpt_path}...')
net = VIRAttResUNet(
        im_chn=3,
        sigma_chn=3,
        n_feat=[96, 160, 224, 288],
        dep_S=8,
        n_resblocks=3,
        noise_cond=True,
        extra_mode='Input',
        noise_avg=False,
        ).cuda()
ckpt = torch.load(args.ckpt_path)['model_state_dict']
net.load_state_dict(OrderedDict({key[7:]:value for key, value in ckpt.items()}), strict=True)
net.eval()

def denoiser_fun(im_noisy, nlf):
    '''
    Input:
        im_noisy:
    '''

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    im_denoise = net(im_noisy)[0]
    end.record()

    torch.cuda.synchronize()
    time_cost = start.elapsed_time(end)/1000
    return im_denoise, time_cost

if __name__ == '__main__':
    flag = 'flip' if args.flip else 'noflip'
    out_folder = str(Path(args.save_dir) / f'dnd_{flag}')
    util_common.mkdir(out_folder, delete=True, parents=True)

    denoiser = pytorch_denoiser(denoiser_fun, True, flip=args.flip)
    denoise_srgb(denoiser, args.dnd_dir, out_folder)
    bundle_submissions_srgb(out_folder)

