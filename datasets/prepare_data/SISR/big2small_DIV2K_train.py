#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2019-01-18 10:26:59

import os
import cv2
import numpy as np
import h5py as h5
import argparse
from pathlib import Path

parser = argparse.ArgumentParser(prog='DIV2KTrain dataset Generation')
# The orignal SIDD images: /ssd1t/DIV2K/DIV2K_train_HR
parser.add_argument('--data_dir', default='/ssd1t/DIV2K/DIV2K_train_HR', type=str, metavar='PATH',
                                      help="path to save the training set of DIV2K, (default: None)")
args = parser.parse_args()

path_all = list(Path(args.data_dir).glob('*.png'))
path_all = sorted([str(x) for x in path_all])
print('Number of big images: {:d}'.format(len(path_all)))

print('Training: Split the original images to small ones!')
path_h5 = Path(args.data_dir).parent / 'DIV2K_small_train_HR.hdf5'
if path_h5.exists():
    path_h5.unlink()
pch_size = 512
stride = 512-128
num_patch = 0
C = 3
with h5.File(str(path_h5), 'w') as h5_file:
    for ii in range(len(path_all)):
        if (ii+1) % 100 == 0:
            print('    The {:d} original images'.format(ii+1))
        im_HR_int8 = cv2.imread(path_all[ii])[:, :, ::-1]
        H, W, _ = im_HR_int8.shape
        ind_H = list(range(0, H-pch_size+1, stride))
        if ind_H[-1] < H-pch_size:
            ind_H.append(H-pch_size)
        ind_W = list(range(0, W-pch_size+1, stride))
        if ind_W[-1] < W-pch_size:
            ind_W.append(W-pch_size)
        for start_H in ind_H:
            for start_W in ind_W:
                pch = im_HR_int8[start_H:start_H+pch_size, start_W:start_W+pch_size, ]
                h5_file.create_dataset(name=str(num_patch), shape=pch.shape, dtype=pch.dtype, data=pch)
                num_patch += 1
print('Total {:d} small images in training set'.format(num_patch))
print('Finish!\n')


