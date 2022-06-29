#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2021-12-14 17:38:52

import scipy.io as sio
import matplotlib.pyplot as plt

kernel_path = './test_data/kernels_sisr/kernel_sf4.mat'
kernel = sio.loadmat(kernel_path)['kernels']
for ii in range(7):
    plt.subplot(1, 7, ii+1)
    plt.imshow(kernel[:, :, ii], cmap='gray')

plt.show()

