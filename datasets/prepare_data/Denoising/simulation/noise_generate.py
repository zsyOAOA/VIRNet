#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2019-05-05 14:44:04

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[4]))

import cv2
import argparse
import h5py as h5
import numpy as np
from skimage import img_as_float

from utils import util_image
from utils import util_denoising

parser = argparse.ArgumentParser()
# trainning settings
parser.add_argument('--noise_type', type=str, default='NIID', help="NIID or IID")
parser.add_argument('--save_dir', type=str, default='', help="Path to save the noisy images")
args = parser.parse_args()

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
    print('Dataset: {:s}'.format(data_name))
    data_path = base_path / data_name
    im_list = sorted([str(x) for x in data_path.glob('*.'+ext_name)])

    for jj, sigma_base in enumerate(var_maps):
        # generate sigmaMap
        if args.noise_type.lower() == 'niid':
            sigma_base = sigma_min + (sigma_base-sigma_base.min())/(sigma_base.max()-sigma_base.min()) * (sigma_max-sigma_min)
        noise_dir = Path(args.save_dir) / args.noise_type.lower()
        if not noise_dir.is_dir(): noise_dir.mkdir()
        if args.noise_type.lower() == 'niid':
            h5_path = noise_dir / (data_name + '_case' + str(jj+1) + '.hdf5')
        else:
            h5_path = noise_dir / (data_name + '_sigma' + str(sigma_base) + '.hdf5')
        if h5_path.exists(): h5_path.unlink()
        with h5.File(h5_path, 'w') as h5_file:
            for im_path in im_list:
                im_name = 'im_' + Path(im_path).stem
                im_gt = util_image.imread(im_path, chn='rgb', dtype='float32')
                assert im_gt.ndim == 3 and im_gt.shape[-1] == 3
                h, w = im_gt.shape[:2]
                if args.noise_type.lower() == 'niid':
                    sigma = cv2.resize(sigma_base, (w, h), interpolation=cv2.INTER_NEAREST_EXACT).astype(np.float32) # H x W
                else:
                    sigma = np.ones([h, w], dtype=np.float32) * (sigma_base / 255.)
                noise = rng.standard_normal(size=im_gt.shape) * sigma[:, :, np.newaxis]
                im_noisy = im_gt + noise.astype(np.float32)
                data = np.concatenate((im_noisy, im_gt, sigma[:, :, np.newaxis].astype(np.float32)), axis=2)
                h5_file.create_dataset(name=im_name, dtype=data.dtype, shape=data.shape, data=data)

