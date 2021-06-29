#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2019-01-22 22:07:08

import cv2
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage.measure import compare_psnr, compare_ssim
from skimage import img_as_ubyte
import numpy as np
import sys
from math import floor
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.io import savemat, loadmat
from scipy.ndimage import correlate
from datasets.imresize import imresize as imresizenp
from loss.imresize_bicubic import imresize as imresizetorch

def rgb2ycbcr(im, only_y=True):
    '''
    same as matlab rgb2ycbcr
    Input:
        im: uint8 [0,255] or float [0,1]
        only_y: only return Y channel
    '''
    in_im_type = im.dtype
    im = im.astype(np.float64)
    if in_im_type != np.uint8:
        im *= 255.
    # convert
    if only_y:
        rlt = np.dot(im, np.array([65.481, 128.553, 24.966])/ 255.0) + 16.0
    else:
        rlt = np.matmul(im, np.array([[65.481,  -37.797, 112.0  ],
                                     [128.553, -74.203, -93.786],
                                     [24.966,  112.0,   -18.214]])/255.0) + [16, 128, 128]
    if in_im_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_im_type)

def rgb2ycbcrTorch(im, only_y=True):
    '''
    same as matlab rgb2ycbcr
    Input:
        im: float [0,1], N x 3 x H x W
        only_y: only return Y channel
    '''
    # transform to range [0,255.0]
    im_temp = im * 255.0
    im_temp = im_temp.permute([0,2,3,1])  # N x H x W x C --> N x H x W x C
    # convert
    if only_y:
        rlt = torch.matmul(im_temp, torch.tensor([65.481, 128.553, 24.966],
                                        device=im.device, dtype=im.dtype).view([3,1])/ 255.0) + 16.0
    else:
        rlt = torch.matmul(im_temp, torch.tensor([[65.481,  -37.797, 112.0  ],
                                                  [128.553, -74.203, -93.786],
                                                  [24.966,  112.0,   -18.214]],
                                                  device=im.device, dtype=im.dtype)/255.0) + \
                                                    torch.tensor([16, 128, 128]).view([-1, 1, 1, 3])
    rlt /= 255.0
    rlt.clamp_(0.0, 1.0)
    return rlt.permute([0, 3, 1, 2])

def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

def calculate_ssim(img1, img2, border=0):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    h, w = img1.shape[:2]
    img1 = img1[border:h-border, border:w-border]
    img2 = img2[border:h-border, border:w-border]

    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1[:,:,i], img2[:,:,i]))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')

def calculate_psnr(im1, im2, border=0):
    if not im1.shape == im2.shape:
        raise ValueError('Input images must have the same dimensions.')
    h, w = im1.shape[:2]
    im1 = im1[border:h-border, border:w-border]
    im2 = im2[border:h-border, border:w-border]

    im1_temp = im1.astype(np.float64)
    im2_temp = im2.astype(np.float64)
    mse = np.mean((im1_temp - im2_temp)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))

def im2patch(im, pch_size, stride=1):
    '''
    Transform image to patches.
    Input:
        im: 3 x H x W or 1 X H x W image, numpy format
        pch_size: (int, int) tuple or integer
        stride: (int, int) tuple or integer
    '''
    if isinstance(pch_size, tuple):
        pch_H, pch_W = pch_size
    elif isinstance(pch_size, int):
        pch_H = pch_W = pch_size
    else:
        sys.exit('The input of pch_size must be a integer or a int tuple!')

    if isinstance(stride, tuple):
        stride_H, stride_W = stride
    elif isinstance(stride, int):
        stride_H = stride_W = stride
    else:
        sys.exit('The input of stride must be a integer or a int tuple!')

    C, H, W = im.shape
    num_H = len(range(0, H-pch_H+1, stride_H))
    num_W = len(range(0, W-pch_W+1, stride_W))
    num_pch = num_H * num_W
    pch = np.zeros((C, pch_H*pch_W, num_pch), dtype=im.dtype)
    kk = 0
    for ii in range(pch_H):
        for jj in range(pch_W):
            temp = im[:, ii:H-pch_H+ii+1:stride_H, jj:W-pch_W+jj+1:stride_W]
            pch[:, kk, :] = temp.reshape((C, num_pch))
            kk += 1

    return pch.reshape((C, pch_H, pch_W, num_pch))

def batch_PSNR(img, imclean, border=0, ycbcr=False):
    if ycbcr:
        img = rgb2ycbcrTorch(img, True)
        imclean = rgb2ycbcrTorch(imclean, True)
    Img = img.data.cpu().numpy()
    Iclean = imclean.data.cpu().numpy()
    Img = img_as_ubyte(Img)
    Iclean = img_as_ubyte(Iclean)
    PSNR = 0
    h, w = Iclean.shape[2:]
    for i in range(Img.shape[0]):
        PSNR += calculate_psnr(Iclean[i,:,].transpose((1,2,0)), Img[i,:,].transpose((1,2,0)), border)
    return (PSNR/Img.shape[0])

def batch_SSIM(img, imclean, border=0, ycbcr=False):
    if ycbcr:
        img = rgb2ycbcrTorch(img, True)
        imclean = rgb2ycbcrTorch(imclean, True)
    Img = img.data.cpu().numpy()
    Iclean = imclean.data.cpu().numpy()
    Img = img_as_ubyte(Img)
    Iclean = img_as_ubyte(Iclean)
    SSIM = 0
    for i in range(Img.shape[0]):
        SSIM += calculate_ssim(Iclean[i,:,].transpose((1,2,0)), Img[i,:,].transpose((1,2,0)), border)
    return (SSIM/Img.shape[0])

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

def generate_gauss_kernel_mix(H, W):
    '''
    Generate a H x W mixture Gaussian kernel with mean (center) and std (scale).
    Input:
        H, W: interger
        center: mean value of x axis and y axis
        scale: float value
    '''
    pch_size = 32
    K_H = floor(H / pch_size)
    K_W = floor(W / pch_size)
    K = K_H * K_W
    # prob = np.random.dirichlet(np.ones((K,)), size=1).reshape((1,1,K))
    centerW = np.random.uniform(low=0, high=pch_size, size=(K_H, K_W))
    ind_W = np.arange(K_W) * pch_size
    centerW += ind_W.reshape((1, -1))
    centerW = centerW.reshape((1,1,K)).astype(np.float32)
    centerH = np.random.uniform(low=0, high=pch_size, size=(K_H, K_W))
    ind_H = np.arange(K_H) * pch_size
    centerH += ind_H.reshape((-1, 1))
    centerH = centerH.reshape((1,1,K)).astype(np.float32)
    scale = np.random.uniform(low=pch_size/2, high=pch_size, size=(1,1,K))
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
    # [xx, yy] = np.meshgrid(np.linspace(1, 10, 256), np.linspace(-10, 15, 256))
    zz = np.sin(xx) + np.cos(yy)
    return zz

def getGaussianKernel2DCenter(H, W, center, scale):
    centerH = center[0]
    centerW = center[1]
    XX, YY = [x.astype(np.float64) for x in np.meshgrid(np.arange(W), np.arange(H))]
    ZZ = np.exp( (-(XX-centerH)**2-(YY-centerW)**2) / (2*scale**2) )
    ZZ /= ZZ.sum()
    return ZZ

def getGaussianKernel2D(ksize, sigma=-1):
    kernel1D = cv2.getGaussianKernel(ksize, sigma)
    kernel2D = np.matmul(kernel1D, kernel1D.T)
    ZZ = kernel2D / kernel2D.sum()
    return ZZ

def gaussblur(x, kernel, p=7, chn=3):
    x_pad = F.pad(x, pad=[int((p-1)/2),]*4, mode='reflect')
    y = F.conv2d(x_pad, kernel, padding=0, stride=1, groups=chn)

    return y

def calculate_parameters(net):
    out = 0
    for param in net.parameters():
        out += param.numel()
    return out

def load_state_dict_cpu(net, state_dict0):
    state_dict1 = net.state_dict()
    for name, value in state_dict1.items():
        assert 'module.'+name in state_dict0
        state_dict1[name] = state_dict0['module.'+name]
    net.load_state_dict(state_dict1)

def str2bool(mode):
    if mode.lower() == 'true':
        return True
    else:
        return False

def inverse_gamma_kernel(radius, chn):
    '''
    Create the gauss kernel for inverge gamma prior.
    '''
    ksize = radius*2+1
    center = [radius, radius]
    scale = 0.3 * ((ksize-1)*0.5 -1) + 0.8  # opencv setting
    kernel = getGaussianKernel2D(ksize, sigma=scale)
    kernel = np.tile(kernel[np.newaxis, np.newaxis,], [chn, 1, 1, 1])
    kernel = torch.from_numpy(kernel).type(torch.float32)
    return kernel

def get_DIV2K_valid_LR_list(DIV2K_valid_HR_list, base_root, scale):
    DIV2K_valid_LR_list = []
    for ii, x in enumerate(DIV2K_valid_HR_list):
        x_HR = Path(x)
        im_name = x_HR.stem
        ext = x_HR.suffix
        x_LR = base_root / ('X'+str(int(scale))) / (im_name+'x'+str(int(scale))+ext)
        DIV2K_valid_LR_list.append(str(x_LR))

    return DIV2K_valid_LR_list

def modcrop(im, scale):
    H, W, _ = im.shape

    H -= (H % scale)
    W -= (W % scale)

    return im[:H, :W,]

def imshow(x, title=None, cbar=False):
    plt.imshow(np.squeeze(x), interpolation='nearest', cmap='gray')
    if title:
        plt.title(title)
    if cbar:
        plt.colorbar()
    plt.show()

def loadFixNoise():
    torch.manual_seed(10000)
    noise = torch.randn([1, 3, 1024, 1024], dtype=torch.float32)
    return noise

def degradeSingle(im_HR, blur_kernel, scale, downsampler='direct'):
    '''
    im_HR: 1 x 3 x h x w
    blur_kernel: 3 x 1 x k x k
    '''
    pad_len = int((blur_kernel.shape[2]-1)/2)
    im_HR_pad = F.pad(im_HR, (pad_len,)*4, mode='reflect')
    im_blur = F.conv2d(im_HR_pad, blur_kernel, groups=3)
    if downsampler.lower() == 'bicubic':
        im_LR = imresizetorch(im_blur, 1/scale)
    elif downsampler.lower() == 'direct':
        im_LR = im_blur[:, :, 0::scale, 0::scale]
    else:
        sys.exit('Please input corrected downsampler!')

    return im_LR

def degradeSingleNumpy(im_HR, blur_kernel, scale, downsampler='direct'):
    '''
    im_HR: h x w x 3 numpy array
    blur_kernel: k x k numpy array
    '''
    im_blur = correlate_im(im_HR, blur_kernel, mode='mirror')
    if downsampler.lower() == 'bicubic':
        im_LR = imresizenp(im_blur, 1/scale)
    elif downsampler.lower() == 'direct':
        im_LR = im_blur[0::scale, 0::scale,]
    else:
        sys.exit('Please input corrected downsampler!')

    return im_LR

def correlate_im(im, kernel, mode='mirror'):
    '''
    im: h x w x 3 or h x w numpy array
    '''
    if im.ndim == 2:
        out = correlate(im, kernel, mode=mode)
    elif im.ndim == 3:
        out = np.zeros_like(im)
        for ii in range(im.shape[2]):
            out[:, :, ii] = correlate(im[:, :, ii], kernel, mode=mode)
    else:
        sys.exit('Please input corrected image format')

    return out

def var_estimate(err, kernel):
    err2 = err**2
    err2_pad = F.pad(err2, pad=(int((kernel.shape[-1]-1)/2),)*4, mode='reflect')
    varmap_est = F.conv2d(err2_pad, kernel, groups=err.shape[1])

    return varmap_est

class PadUNet:
    '''
    im: N x C x H x W torch tensor
    dep_U: depth of UNet
    '''
    def __init__(self, im, dep_U, mode='reflect'):
        self.im_old = im
        self.dep_U = dep_U
        self.mode = mode
        self.H_old = im.shape[2]
        self.W_old = im.shape[3]

    def pad(self):
        lenU = 2 ** (self.dep_U-1)
        padH = 0 if ((self.H_old % lenU) == 0) else (lenU - (self.H_old % lenU))
        padW = 0 if ((self.W_old % lenU) == 0) else (lenU - (self.W_old % lenU))
        padding = (0, padW, 0, padH)
        out = F.pad(self.im_old, pad=padding, mode=self.mode)
        return out

    def pad_inverse(self, im_new):
        return im_new[:, :, :self.H_old, :self.W_old]

if __name__ == "__main__":
    # aa = np.array([[1, 2, 0, 0],
                   # [5, 3, 0, 4],
                   # [0, 0, 0, 7],
                   # [9, 3, 0, 0]])
    # aa = np.tile(aa[:, :, np.newaxis], (1, 1, 3))
    # kernel = np.array([[1,1,1], [1,1,0], [1, 0, 0]])
    np.random.seed(0)
    aa = np.random.randn(128, 128, 3)
    kernel = np.random.randn(5, 5)
    bb1 = degradeSingleNumpy(aa, kernel, 2, downsampler='direct')

    aa_temp = torch.from_numpy(aa.transpose((2,0,1))[np.newaxis,]).type(torch.float32)
    kernel_temp = torch.from_numpy(np.tile(kernel[np.newaxis, np.newaxis,], (3,1,1,1))).type(torch.float32)
    bb2_temp = degradeSingle(aa_temp, kernel_temp, 2, downsampler='direct')
    bb2 = bb2_temp.numpy().squeeze().transpose((1,2,0))

    print('Maximal Absolute Error: {:e}'.format(np.abs((bb1-bb2)).max()))

