#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2021-11-24 17:15:34

import cv2 
import math
import numpy as np

import torch
import torch.nn.functional as F

def getGaussianKernel2DCenter(H, W, center, scale):
    '''
    Generating Gaussian kernel (H x W) with std=scale.

    '''
    centerH = center[0]
    centerW = center[1]
    ii, jj = [x.astype(np.float64) for x in np.meshgrid(np.arange(H), np.arange(W), indexing='ij')]
    kk = np.exp( (-(ii-centerH)**2-(jj-centerW)**2) / (2*scale**2) )
    kk /= kk.sum()
    return kk

def inverse_gamma_kernel(ksize, chn):
    '''
    Create the gauss kernel for inverge gamma prior.
    out:
        kernel: chn x 1 x k x k
    '''
    scale = 0.3 * ((ksize-1)*0.5 -1) + 0.8  # opencv setting
    kernel = getGaussianKernel2D(ksize, sigma=scale)
    kernel = np.tile(kernel[np.newaxis, np.newaxis,], [chn, 1, 1, 1])
    kernel = torch.from_numpy(kernel).type(torch.float32)
    return kernel

def getGaussianKernel2D(ksize, sigma=-1):
    kernel1D = cv2.getGaussianKernel(ksize, sigma)
    kernel2D = np.matmul(kernel1D, kernel1D.T)
    ZZ = kernel2D / kernel2D.sum()
    return ZZ

def conv_multi_chn(x, kernel):
    '''
    In:
        x: B x chn x h x w, tensor
        kernel: chn x 1 x k x k, tensor
    '''
    x_pad = F.pad(x, pad=[kernel.shape[-1]//2, ]*4, mode='reflect')
    y = F.conv2d(x_pad, kernel, padding=0, stride=1, groups=x.shape[1])

    return y

def noise_estimate_fun(im_noisy, im_gt, k_size):
    '''
    Estatmate the variance map.
    Input:
        im_noisy: N x c x h x w
    '''
    kernel = inverse_gamma_kernel(k_size, im_noisy.shape[1]).to(im_noisy.device)
    err2 = (im_noisy - im_gt) ** 2
    sigma_prior = conv_multi_chn(err2, kernel)
    sigma_prior.clamp_(min=1e-10)
    return sigma_prior

def noise_generator(seed=1000):
    rng = np.random.default_rng(seed=seed)
    return rng

def peaks(n):
    '''
    Implementation the peak function of matlab.
    '''
    X = np.linspace(-3, 3, n)
    Y = np.linspace(-3, 3, n)
    [XX, YY] = np.meshgrid(X, Y)
    ZZ = 3 * (1-XX)**2 * np.exp(-XX**2 - (YY+1)**2) \
            - 10 * (XX/5.0 - XX**3 -YY**5) * np.exp(-XX**2-YY**2) - 1/3.0 * np.exp(-(XX+1)**2 - YY**2)
    return ZZ

def generate_gauss_kernel_mix(H, W, rng=None):
    '''
    Generate a H x W mixture Gaussian kernel with mean (center) and std (scale).
    Input:
        H, W: interger
        center: mean value of x axis and y axis
        scale: float value
    '''
    pch_size = 32
    K_H = math.floor(H / pch_size)
    K_W = math.floor(W / pch_size)
    K = K_H * K_W
    # prob = np.random.dirichlet(np.ones((K,)), size=1).reshape((1,1,K))
    if rng is None:
        centerW = np.random.uniform(low=0, high=pch_size, size=(K_H, K_W))
    else:
        centerW = rng.uniform(low=0, high=pch_size, size=(K_H, K_W))
    ind_W = np.arange(K_W) * pch_size
    centerW += ind_W.reshape((1, -1))
    centerW = centerW.reshape((1,1,K)).astype(np.float32)
    if rng is None:
        centerH = np.random.uniform(low=0, high=pch_size, size=(K_H, K_W))
    else:
        centerH = rng.uniform(low=0, high=pch_size, size=(K_H, K_W))
    ind_H = np.arange(K_H) * pch_size
    centerH += ind_H.reshape((-1, 1))
    centerH = centerH.reshape((1,1,K)).astype(np.float32)
    if rng is None:
        scale = np.random.uniform(low=pch_size/2, high=pch_size, size=(1,1,K))
    else:
        scale = rng.uniform(low=pch_size/2, high=pch_size, size=(1,1,K))
    scale = scale.astype(np.float32)
    XX, YY = np.meshgrid(np.arange(0, W), np.arange(0,H))
    XX = XX[:, :, np.newaxis].astype(np.float32)
    YY = YY[:, :, np.newaxis].astype(np.float32)
    ZZ = 1./(2*np.pi*scale**2) * np.exp( (-(XX-centerW)**2-(YY-centerH)**2)/(2*scale**2) )
    out = ZZ.sum(axis=2, keepdims=False) / K

    return out

def sincos_kernel():
    # Nips Version
    [xx, yy] = np.meshgrid(np.linspace(1, 10, 256), np.linspace(1, 20, 256))
    zz = np.sin(xx) + np.cos(yy)
    return zz
