#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2019-09-02 15:11:05

import cv2
import torch
import random
import numpy as np
from skimage import img_as_ubyte
import scipy.stats as ss

class MixUp_AUG:
    '''
    This mix up strategy is borrowed from https://github.com/swz30/MPRNet
    '''
    def __init__(self):
        self.dist = torch.distributions.beta.Beta(torch.tensor([0.6]), torch.tensor([0.6]))

    def aug(self, rgb_gt, rgb_noisy):
        bs = rgb_gt.size(0)
        indices = torch.randperm(bs)
        rgb_gt2 = rgb_gt[indices]
        rgb_noisy2 = rgb_noisy[indices]

        lam = self.dist.rsample((bs,1)).view(-1,1,1,1).cuda()

        rgb_gt    = lam * rgb_gt + (1-lam) * rgb_gt2
        rgb_noisy = lam * rgb_noisy + (1-lam) * rgb_noisy2

        return rgb_gt, rgb_noisy

def rgb2ycbcr(img):
    '''
    https://stackoverflow.com/questions/26480125/how-to-get-the-same-output-of-rgb2ycbcr-matlab-function-in-python-opencv
    Input:
        img: H x W x 3 tensor, np.uint8 format, range:[0,255]
    Output:
        out: np.float64 format, range:[0,1]
    '''
    W = np.array([[65.481,  -37.797, 112],
                  [128.553, -74.203, -93.786],
                  [24.966,  112.0,   -18.214]], dtype=np.float64)
    b = np.array([16, 128, 128], dtype=np.float64).reshape((1,1,3))
    Y = np.tensordot(img.astype(np.float64)/255.0, W, axes=[2, 0]) + b  # range [0,255]

    return img_as_ubyte(Y/255.0)

def anisotropic_Gaussian(ksize=25, theta=np.pi, l1=6, l2=6):
    """
    https://github.com/cszn/KAIR/blob/master/utils/utils_sisr.py
    Generate an anisotropic Gaussian kernel
    Args:
        ksize : e.g., 25, kernel size
        theta : [0,  pi], rotation angle range
        l1    : [0.1,50], scaling of eigenvalues
        l2    : [0.1,l1], scaling of eigenvalues
        If l1 = l2, will get an isotropic Gaussian kernel.
    Returns:
        k     : kernel
    """

    v = np.dot(np.array([[np.cos(theta), -np.sin(theta)],
                         [np.sin(theta), np.cos(theta)]]), np.array([1., 0.]))
    V = np.array([[v[0], v[1]], [v[1], -v[0]]])
    D = np.array([[l1, 0], [0, l2]])
    Sigma = np.dot(np.dot(V, D), np.linalg.inv(V))
    k = gm_blur_kernel(mean=[0, 0], cov=Sigma, size=ksize)

    return k

def gm_blur_kernel(mean, cov, size=25):
    center = size / 2.0 + 0.5
    k = np.zeros([size, size])
    for y in range(size):
        for x in range(size):
            cy = y - center + 1
            cx = x - center + 1
            k[y, x] = ss.multivariate_normal.pdf([cx, cy], mean=mean, cov=cov)

    k = k / np.sum(k)
    return k

