#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2019-05-05 14:44:04

import sys
sys.path.append('./')

import numpy as np
import cv2
from skimage import img_as_float
from utils import generate_gauss_kernel_mix, peaks, sincos_kernel
import h5py as h5
from pathlib import Path

def prepare_test(chn):
    seed = 10000
    base_path = Path('./test_data')

    np.random.seed(seed)
    kernels = [peaks(256), sincos_kernel(), generate_gauss_kernel_mix(256, 256)]
    dep_U = 3

    sigma_max = 75/255.0
    sigma_min = 10/255.0
    # for data_name in ['CBSD68', 'Kodak24', 'McMaster']:
    for data_name in ['CBSD68', ]:
        print('Dataset: {:s}'.format(data_name))
        if data_name.lower() == 'mcmaster':
            im_list = sorted((base_path / data_name).glob('*.tif'))
        else:
            im_list = sorted((base_path / data_name).glob('*.png'))

        for jj, sigma in enumerate(kernels):
            # generate sigmaMap
            sigma = sigma_min + (sigma-sigma.min())/(sigma.max()-sigma.min()) * (sigma_max-sigma_min)
            noise_dir = base_path / ('noise_niid_chn'+str(chn))
            if not noise_dir.is_dir():
                noise_dir.mkdir()
            h5_path = noise_dir.joinpath(data_name + '_case' + str(jj+1) + '.hdf5')
            if h5_path.exists():
                h5_path.unlink()
            with h5.File(h5_path, 'a') as h5_file:
                for ii, im_name in enumerate(im_list):
                    if chn == 3:
                        im_gt = cv2.imread(str(im_list[ii]), cv2.IMREAD_COLOR)[:, :, ::-1]
                    else:
                        im_gt = cv2.imread(str(im_list[ii]), cv2.IMREAD_GRAYSCALE)[:, :, np.newaxis]
                    H, W, C = im_gt.shape
                    H -= int(H % pow(2, dep_U))
                    W -= int(W % pow(2, dep_U))
                    im_gt = img_as_float(im_gt[:H, :W])

                    sigma = cv2.resize(sigma, (W, H)).astype(np.float32) # H x W
                    noise = np.random.randn(H, W, C) * np.expand_dims(sigma, 2) # H x W x C
                    noise = noise.astype(np.float32)
                    data = np.concatenate((noise, sigma[:, :,np.newaxis]), axis=2)
                    im_name = im_list[ii].name.split('.')[0]
                    h5_file.create_dataset(name=im_name, dtype=data.dtype, shape=data.shape, data=data)

if __name__ == '__main__':
    prepare_test(3)

