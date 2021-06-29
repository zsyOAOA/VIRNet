#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2019-11-12 14:36:24

import sys
import argparse
import h5py as h5
from pathlib import Path
import cv2

parser = argparse.ArgumentParser(prog='Deblocking Train dataset Generation')
# The DIV2K images path: /ssd1t/DIV2K/DIV2K_train_HR/
parser.add_argument('--data_dir', default=None, type=str, metavar='PATH',
                                 help="path to save the original HR DIV2K dataset, (default: None)")
parser.add_argument('--save_dir', default=None, type=str, metavar='PATH',
                 help="path to save the Generated training and validation dataset, (default: None)")
args = parser.parse_args()

data_dir = Path(args.data_dir)
if not ((data_dir / DIV2K_train_HR).exists() and (data_dir / DIV2K_valid_HR).exists()):
    sys.exit('Please input the corrected data_dir')
save_dir = Path(args.save_dir)
if not (save_dir / 'train').is_dir():
    save_dir.mkdir('train')
if not (save_dir / 'valid').is_dir():
    save_dir.mkdir('valid')

im_list_train = (data_dir / DIV2K_train_HR).glob('*.png')
im_list_train = sorted([str(x) for x in im_list_train])

print('Begin making the training dataset: {:d} images'.format(len(im_list_train)))
h5_path_train = (data_dir / 'train.hdf5')
if h5_path_train.exists():
    print('Delete the existing hdf5 file: {:s}'.format(h5_path_train))
    h5_path_train.unlink()
with h5.File(h5_path_train, 'w') as h5_train:
    for ii, im_path in enumerate(im_list_train):
        if (ii+1) % 200 == 0:
            print('Processsing: {:d}/{:d}'.format(ii+1, len(im_list_train)))
        im_gt = cv2.imread(im_path, cv2.IMREAD_COLOR)
        save_path = data_dir / 'train' / (str(ii)+'.jpg')

