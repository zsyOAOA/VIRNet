#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2019-01-18 10:26:59

import os
import cv2
import random
import argparse
import numpy as np
from glob import glob
from pathlib import Path
from joblib import Parallel, delayed

parser = argparse.ArgumentParser(prog='SIDD Train dataset Generation')
# The orignal SIDD images: /userhome/cs/zsyue/data/SIDD/
parser.add_argument('--data_dir', default=None, type=str, metavar='PATH',
                                      help="path to save the training set of SIDD, (default: None)")
parser.add_argument('--pch_size', default=256, type=int, help="Image patch size, (default: 256)")
parser.add_argument('--stride', default=128, type=int, help="Stride for croping, (default: 128)")
parser.add_argument('--per_num_pch', default=400, type=int,
                                       help="Cropped patchs in each original image, (default: 128)")
parser.add_argument('--seed', default=10000, type=int, help="Random Seed")
args = parser.parse_args()


# random seed
random.seed(args.seed)

path_all_noisy = glob(os.path.join(args.data_dir, '**/*NOISY*.PNG'), recursive=True)
path_all_noisy = sorted(path_all_noisy)
path_all_gt = [x.replace('NOISY', 'GT') for x in path_all_noisy]
print('Number of big images: {:d}'.format(len(path_all_gt)))

pch_dir_noisy = Path(args.data_dir) / f'patchs{args.pch_size}' / 'noisy'
if not pch_dir_noisy.exists():
    pch_dir_noisy.mkdir(parents=True)

pch_dir_gt = Path(args.data_dir) / f'patchs{args.pch_size}' / 'gt'
if not pch_dir_gt.exists():
    pch_dir_gt.mkdir(parents=True)

pch_size = args.pch_size
stride = args.stride

def save_files(ii):
    print('Processing: {:s}'.format(str(path_all_noisy[ii])))
    im_noisy = cv2.imread(path_all_noisy[ii])
    im_gt = cv2.imread(path_all_gt[ii])
    h, w = im_noisy.shape[:2]
    ind_h = list(range(0, h-pch_size, stride)) + [h-pch_size,]
    ind_w = list(range(0, w-pch_size, stride)) + [w-pch_size,]
    num_pch = 0
    for start_h in ind_h:
        for start_w in ind_w:
            num_pch += 1

            pch_noisy = im_noisy[start_h:start_h+pch_size, start_w:start_w+pch_size, ]
            pch_noisy_name = 'sidd_' + '{:04d}_{:04d}.png'.format(ii+1, num_pch)
            cv2.imwrite(str(pch_dir_noisy/pch_noisy_name), pch_noisy)

            pch_gt = im_gt[start_h:start_h+pch_size, start_w:start_w+pch_size, ]
            pch_gt_name = 'sidd_' + '{:04d}_{:04d}.png'.format(ii+1, num_pch)
            cv2.imwrite(str(pch_dir_gt/pch_gt_name), pch_gt)

def save_files_random(ii):
    print('Processing: {:s}'.format(str(path_all_noisy[ii])))
    im_noisy = cv2.imread(path_all_noisy[ii])
    im_gt = cv2.imread(path_all_gt[ii])
    h, w = im_noisy.shape[:2]
    for jj in range(args.per_num_pch):
        start_h = random.randint(0, h-pch_size)
        start_w = random.randint(0, w-pch_size)
        pch_noisy = im_noisy[start_h:start_h+pch_size, start_w:start_w+pch_size, ]
        pch_noisy_name = 'sidd_' + '{:04d}_{:04d}.png'.format(ii+1, jj+1)
        cv2.imwrite(str(pch_dir_noisy/pch_noisy_name), pch_noisy)

        pch_gt = im_gt[start_h:start_h+pch_size, start_w:start_w+pch_size, ]
        pch_gt_name = 'sidd_' + '{:04d}_{:04d}.png'.format(ii+1, jj+1)
        cv2.imwrite(str(pch_dir_gt/pch_gt_name), pch_gt)

num_workers = os.cpu_count()
Parallel(n_jobs=num_workers)(delayed(save_files_random)(ii) for ii in range(len(path_all_noisy)))

pch_noisy_path_list = [x for x in pch_dir_noisy.glob('*.png')]
print('{:d} patch pairs in SIDD'.format(len(pch_noisy_path_list)))  # 512-->30608

