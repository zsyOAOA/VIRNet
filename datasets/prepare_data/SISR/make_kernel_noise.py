#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2020-06-07 17:21:29

from pathlib import Path

import sys
sys.path.append(str(Path('./')))

from utils import getGaussianKernel2D
from datasets.data_tools import anisotropic_Gaussian

import numpy as np
from scipy.io import savemat, loadmat

p = 15
kernels = np.zeros([p, p, 8])
kernels_kai = loadmat('/home/oa/code/python/VDNet-TPAMI/test_data/kernels_SISR/kernels_12.mat')['kernels']

kernels[:, :, 0] = getGaussianKernel2D(p, 0.7)
kernels[:, :, 1] = getGaussianKernel2D(p, 1.2)
kernels[:, :, 2] = getGaussianKernel2D(p, 1.6)
kernels[:, :, 3] = getGaussianKernel2D(p, 2.0)

kernels[:, :, 4] = anisotropic_Gaussian(p, np.pi*0, 4, 1.5)
kernels[:, :, 5] = anisotropic_Gaussian(p, np.pi*0.75, 6, 1)
kernels[:, :, 6] = anisotropic_Gaussian(p, np.pi*0.25, 6, 1)
kernels[:, :, 7] = anisotropic_Gaussian(p, np.pi*0.1, 5, 3)


kernel_path = Path('./test_data') / 'kernels_SISR'
if not kernel_path.exists():
    kernel_path.mkdir()
savemat(str(kernel_path/'kernels_8.mat'), {'kernels':kernels})

np.random.seed(10000)
noise = np.zeros([1024, 1024, 3, 2])
noise[:, :, :, 0] = np.random.randn(1024, 1024, 3) * (2.55/255)
noise[:, :, :, 1] = np.random.randn(1024, 1024, 3) * (7.65/255)

noise_path = Path('./test_data') / 'noise_SISR'
if not noise_path.exists():
    noise_path.mkdir()
savemat(str(noise_path/'noise.mat'), {'noise':noise})

