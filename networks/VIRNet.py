#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2019-09-01 19:35:06
import sys
sys.path.append('./datasets')

import random
import scipy
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from math import pi
from .DnCNN import DnCNN
from .UNet import UNet
from utils import PadUNet
from datasets.data_tools import anisotropic_Gaussian
from pathlib import Path
from scipy.io import loadmat, savemat

class VIRNetU(nn.Module):
    '''
    For Denoising task with UNet denoiser.
    '''
    def __init__(self, in_channels, wf=64, dep_S=8, dep_U=4, slope=0.2):
        super(VIRNetU, self).__init__()
        self.RNet = UNet(in_channels, in_channels*2, wf=wf, depth=dep_U, slope=slope)
        self.SNet = DnCNN(in_channels, in_channels*2, dep=dep_S)
        self.dep_U = dep_U

    def forward(self, x, mode='train'):
        C = x.shape[1]
        if mode.lower() == 'train':
            phi_Z = self.RNet(x)
            phi_Z[:, :C, ] = x - phi_Z[:, :C,]
            phi_sigma = self.SNet(x)
            return phi_Z, phi_sigma
        elif mode.lower() == 'test':
            padunet = PadUNet(x, dep_U=self.dep_U)
            x_pad = padunet.pad()
            phi_Z = self.RNet(x_pad)
            phi_Z[:, :C, ] = x_pad - phi_Z[:, :C,]
            phi_Z = padunet.pad_inverse(phi_Z)
            return phi_Z
        elif mode.lower() == 'sigma':
            phi_sigma = self.SNet(x)
            return phi_sigma

class VIRNetBlock(VIRNetU):
    '''
    For Deblocking task with UNet.
    '''
    def forward(self, x, mode='train'):
        C = x.shape[1]
        if mode.lower() == 'train':
            phi_Z = self.RNet(x)
            phi_Z[:, :C, ] = x + phi_Z[:, :C,]
            phi_sigma = self.SNet(x)
            return phi_Z, phi_sigma
        elif mode.lower() == 'test':
            phi_Z = self.RNet(x)
            phi_Z[:, :C, ] = x + phi_Z[:, :C,]
            return phi_Z
        elif mode.lower() == 'sigma':
            phi_sigma = self.SNet(x)
            return phi_sigma

class VIRNetSISR(nn.Module):
    '''
    For super-resolution task with UNet.
    '''
    def __init__(self, in_channels, wf=64, dep_U=4, dep_S=3, ksize=15, dim_pca=15):
        super(VIRNetSISR, self).__init__()
        self.dim_pca = dim_pca
        self.dep_U = dep_U
        pca_matrix = cal_pca_matrix(ksize, dim_pca=dim_pca)
        self.pca_matrix = torch.from_numpy(pca_matrix).type(torch.float32)
        self.RNet = UNet(in_channels+dim_pca, in_channels*2, wf=wf, depth=dep_U)
        self.SNet = DnCNN(in_channels, 2, dep=dep_S)

    def forward(self, x, kernel, scale, pad=False):
        '''
        Input:
            x: N x C x H x W
            kernel: N x 1 x k x k
            scale: int scalar
        '''
        phi_sigma = self.SNet(x)
        x_up = F.interpolate(x, scale_factor=scale, mode='nearest')
        if pad:
            padunet = PadUNet(x_up, dep_U=self.dep_U)
            x_up = padunet.pad()
        N, C, H, W = x_up.shape
        kernelmap = self.pca_matrix.to(x_up.device).matmul(kernel.view([N,-1,1])).unsqueeze(3)
        phi_Z = self.RNet(torch.cat([x_up,
                                     kernelmap.expand(N, self.dim_pca, H, W)], dim=1))
        phi_Z[:, :C,] = x_up + phi_Z[:, :C,]
        if pad:
            phi_Z = padunet.pad_inverse(phi_Z)
        return phi_Z, phi_sigma

def get_pca_matrix(x, dim_pca=15):
    """
    https://github.com/cszn/KAIR/blob/master/utils/utils_sisr.py
    Args:
        x: 225xN matrix
        dim_pca: 15
    Returns:
        pca_matrix: 15x225
    """
    C = np.dot(x, x.T)
    w, v = scipy.linalg.eigh(C)
    pca_matrix = v[:, -dim_pca:].T

    return pca_matrix

def cal_pca_matrix(ksize=15, l_max=12.0, dim_pca=15, num_samples=1000):
    '''
    Modified from https://github.com/cszn/KAIR/blob/master/utils/utils_sisr.py
    '''
    path = Path('./networks/pca_matrix.mat')
    if path.exists():
        pca_matrix = loadmat(str(path))['p']
    else:
        kernels = np.zeros([ksize*ksize, num_samples], dtype=np.float32)
        for i in range(num_samples):

            theta = pi*random.random()
            l1    = 0.1+l_max*random.random()
            l2    = 0.1+(l1-0.1)*random.random()

            k = anisotropic_Gaussian(ksize=ksize, theta=theta, l1=l1, l2=l2)

            # util.imshow(k)

            kernels[:, i] = np.reshape(k, (-1))  # C order

        pca_matrix = get_pca_matrix(kernels, dim_pca=dim_pca)

        savemat(path, {'p': pca_matrix})

    return pca_matrix

