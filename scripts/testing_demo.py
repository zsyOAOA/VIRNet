#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2024-03-05 11:58:30

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parents[1]))

import cv2
import argparse
import numpy as np
from tqdm import tqdm
from collections import OrderedDict
from skimage import img_as_ubyte

import torch

from utils import util_image
from utils import util_common

def load_model(task, ckpt_path, sf=None):
    if task == 'denoising-syn':
        from networks.VIRNet import VIRAttResUNet
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
        if not ckpt_path:
            ckpt_path = str(Path('model_zoo') / 'virnet_denoising_syn.pth' )
    elif task == 'denoising-real':
        from networks.VIRNet import VIRAttResUNet
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
        if not ckpt_path:
            ckpt_path = str(Path('model_zoo') / 'virnet_denoising_real.pth' )
    else:
        from networks.VIRNet import VIRAttResUNetSR
        net = VIRAttResUNetSR(
                im_chn=3,
                sigma_chn=1,
                dep_S=5,
                dep_K=8,
                n_feat=[96, 160, 224],
                n_resblocks=2,
                extra_mode='Both',
                noise_avg=True,
                noise_cond=True,
                kernel_cond=True,
                ).cuda()
        if not ckpt_path:
            ckpt_path= str(Path('model_zoo') / f'virnet_sisr_x{sf}.pth')

    print(f"Loading model from {ckpt_path}...")
    ckpt = torch.load(ckpt_path)['model_state_dict']
    try:
        net.load_state_dict(ckpt, strict=True)
    except:
        net.load_state_dict(OrderedDict({key[7:]:value for key, value in ckpt.items()}), strict=True)
    net.eval()

    return net

def process_image(net, im_lq, task, sf=None):
    """
    Input:
        im: numpy array, [h, w, c] or [h, w], ranged in [0,1]
    """
    if im_lq.ndim == 2:
        im_lq = np.stack([im_lq,]*3, axis=2)

    inputs = torch.from_numpy(im_lq.transpose([2,0,1])).type(torch.float32).cuda().unsqueeze(0) # 1 x c x h x w
    if task in ['denoising-syn', 'denoising-real']:
        with torch.no_grad():
            im_pred, _ = net(inputs)
        if isinstance(im_pred, list):
            im_pred = im_pred[0]
    else:
        with torch.no_grad():
            im_pred, _, _ = net(inputs, sf)

    out = im_pred.clamp_(0.0, 1.0).cpu().squeeze(0).numpy().transpose([1,2,0]) # h x w x c, numpy array, [0,1]

    return out

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
            '--ckpt_path',
            default='',
            type=str,
            help="Path to the checkpoint!",
            )
    parser.add_argument(
            '-i',
            '--in_path',
            type=str,
            default='',
            help="Input image or folder"
            )
    parser.add_argument(
            '-o',
            '--out_path',
            type=str,
            default='outputs',
            help="Output folder"
            )
    parser.add_argument(
            '--task',
            type=str,
            default='denoising-syn',
            choices=['denoising-syn', 'denoising-real', 'sisr'],
            help="Task name."
            )
    parser.add_argument(
            '--prefix',
            type=str,
            default='',
            help="Prefix on the save results."
            )
    parser.add_argument('--sf', default=4, type=int, help="Downsampling Scale for SR.")
    args = parser.parse_args()

    # Loading the model
    net = load_model(args.task, args.ckpt_path, args.sf)

    # check the output path
    util_common.mkdir(args.out_path, delete=False, parents=True)

    if Path(args.in_path).is_dir():
        im_path_list = [x for x in Path(args.in_path).glob("*.[jJpP][pnPN]*[gG]")]
        for im_path in tqdm(im_path_list):
            im_name = im_path.stem
            im_lq = util_image.imread(str(im_path), chn='rgb', dtype='float32')
            im_pred = process_image(net, im_lq, args.task, args.sf)   # numpy array, [0,1], h x w x c

            # saving the results
            if args.prefix:
                im_path_save = Path(args.out_path) / f"{im_name}_{args.prefix}.png"
            else:
                im_path_save = Path(args.out_path) / f"{im_name}.png"
            util_image.imwrite(img_as_ubyte(im_pred), im_path_save, chn='rgb')
    else:
        im_name = Path(args.in_path).stem
        im_lq = util_image.imread(args.in_path, chn='rgb', dtype='float32')
        im_pred = process_image(net, im_lq, args.task, args.sf)   # numpy array, [0,1], h x w x c

        # saving the results
        if args.prefix:
            im_path_save = Path(args.out_path) / f"{im_name}_{args.prefix}.png"
        else:
            im_path_save = Path(args.out_path) / f"{im_name}.png"
        util_image.imwrite(img_as_ubyte(im_pred), im_path_save, chn='rgb')

    print(f"Please enjoy the result in {args.out_path}!")


if __name__ == '__main__':
    main()
