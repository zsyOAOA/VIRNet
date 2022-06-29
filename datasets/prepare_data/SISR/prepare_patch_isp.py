#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2021-12-22 19:48:58

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[3]))

import cv2
import pickle
import argparse
import numpy as np
from math import ceil
import multiprocessing
from camera_isp.ISP_implement import ISP
from skimage import img_as_ubyte, img_as_float32

parser = argparse.ArgumentParser(prog='SISR dataset Generation')
parser.add_argument('--pch_dir', default='/data2/zongsheng/data/sisr_hr_patchs',
                        type=str, metavar='PATH', help="Path to save the original hr patchs")
parser.add_argument('--isp_dir', default='/data2/zongsheng/data/sisr_hr_patchs_isp',
                   type=str, metavar='PATH', help="Path to save the isp processed hr patchs")
parser.add_argument('--num_workers', default=16, type=int, help="Number of processes")
args = parser.parse_args()

# check floder
if not Path(args.isp_dir).exists():
    Path(args.isp_dir).mkdir()

path_hr_list = sorted([x for x in Path(args.pch_dir).glob('*.png')])
path_hr_list = [{'path':x, 'index': ind} for ind, x in enumerate(path_hr_list)]
print('Number of hr patchs:{:d}'.format(len(path_hr_list)))

isp_dir_image = Path(args.isp_dir) / 'images'
if not isp_dir_image.exists():
    isp_dir_image.mkdir()
isp_dir_meta = Path(args.isp_dir) / 'meta'
if not isp_dir_meta.exists():
    isp_dir_meta.mkdir()

def isp_processing_pch(path_dict):
    im_path = path_dict['path']
    seed = path_dict['index']

    im_srgb = img_as_float32(cv2.imread(str(im_path), cv2.IMREAD_UNCHANGED)[:, :, ::-1]) # rgb float

    isp = ISP()
    # generate the config
    isp.reset_seed(seed)
    isp.random_config()

    # save the config
    pkl_path = isp_dir_meta / (im_path.stem + '.pkl')
    isp.save_config(str(pkl_path))

    im_srgb_syn = isp.simulate_clean(im_srgb, tone_type='func', demosaic_method='Menon') #rgb float
    im_srgb_syn = img_as_ubyte(np.clip(im_srgb_syn, 0.0, 1.0))[:, :, ::-1]               #bgr uint8

    im_path_syn = isp_dir_image / im_path.name 
    cv2.imwrite(str(im_path_syn), im_srgb_syn)

if __name__ == "__main__":
    pool = multiprocessing.Pool(args.num_workers) 
    pool.imap(func=isp_processing_pch,
              iterable=path_hr_list,
              chunksize=ceil(len(path_hr_list)/args.num_workers))
    pool.close()
    pool.join()

    num_pch = len([x for x in isp_dir_image.glob('*.png')])
    print('Total {:d} small patches in training set'.format(num_pch))  # 512-->84726, 256-->516792

    num_pkl = len([x for x in isp_dir_meta.glob('*.pkl')])
    assert num_pkl == num_pch

