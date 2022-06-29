#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2020-06-07 17:21:29

import sys
from pathlib import Path

import numpy as np
import scipy.io as sio

sys.path.append(str(Path(__file__).parents[3]))
from utils import util_sisr

kernel_dir = Path('./test_data') / 'kernels_sisr'
if not kernel_dir.exists():
    kernel_dir.mkdir()

p = 21
for sf in [2, 3, 4]:
    kernels = np.zeros([p, p, 8])

    kernels[:, :, 0] = util_sisr.shifted_anisotropic_Gaussian(p, sf, (0.40*sf)**2, (0.40*sf)**2, 0)[0]
    kernels[:, :, 1] = util_sisr.shifted_anisotropic_Gaussian(p, sf, (0.60*sf)**2, (0.60*sf)**2, 0)[0]
    kernels[:, :, 2] = util_sisr.shifted_anisotropic_Gaussian(p, sf, (0.80*sf)**2, (0.80*sf)**2, 0)[0]

    kernels[:, :, 3] = util_sisr.shifted_anisotropic_Gaussian(p, sf, (0.4*sf)**2, (0.2*sf)**2, 0)[0]
    kernels[:, :, 4] = util_sisr.shifted_anisotropic_Gaussian(p, sf, (0.6*sf)**2, (0.3*sf)**2, 0.75*np.pi)[0]
    kernels[:, :, 5] = util_sisr.shifted_anisotropic_Gaussian(p, sf, (0.8*sf)**2, (0.4*sf)**2, 0.25*np.pi)[0]
    kernels[:, :, 6] = util_sisr.shifted_anisotropic_Gaussian(p, sf, (0.8*sf)**2, (0.4*sf)**2, 0.50*np.pi)[0]

    kernel_path = kernel_dir / ('kernel_sf' + str(sf) + '.mat')
    sio.savemat(kernel_path, {'kernels':kernels})

