#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2021-11-24 16:54:19

import sys
import cv2
import math
import torch
import lpips
import random
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from skimage import img_as_ubyte, img_as_float32, img_as_float64

# --------------------------Metrics----------------------------
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

def calculate_ssim(im1, im2, border=0, ycbcr=False):
    '''
    SSIM the same outputs as MATLAB's
    im1, im2: h x w x , [0, 255], uint8
    '''
    if not im1.shape == im2.shape:
        raise ValueError('Input images must have the same dimensions.')

    if ycbcr:
        im1 = rgb2ycbcr(im1, True)
        im2 = rgb2ycbcr(im2, True)

    h, w = im1.shape[:2]
    im1 = im1[border:h-border, border:w-border]
    im2 = im2[border:h-border, border:w-border]

    if im1.ndim == 2:
        return ssim(im1, im2)
    elif im1.ndim == 3:
        if im1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(im1[:,:,i], im2[:,:,i]))
            return np.array(ssims).mean()
        elif im1.shape[2] == 1:
            return ssim(np.squeeze(im1), np.squeeze(im2))
    else:
        raise ValueError('Wrong input image dimensions.')

def calculate_psnr(im1, im2, border=0, ycbcr=False):
    '''
    PSNR metric.
    im1, im2: h x w x , [0, 255], uint8
    '''
    if not im1.shape == im2.shape:
        raise ValueError('Input images must have the same dimensions.')

    if ycbcr:
        im1 = rgb2ycbcr(im1, True)
        im2 = rgb2ycbcr(im2, True)

    h, w = im1.shape[:2]
    im1 = im1[border:h-border, border:w-border]
    im2 = im2[border:h-border, border:w-border]

    im1 = im1.astype(np.float64)
    im2 = im2.astype(np.float64)
    mse = np.mean((im1 - im2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))

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

def normalize_lpips(im):
    '''
    Input:
        im1: h x w x c, [0, 255], uint8, numpy array
        im2: h x w x c, [0, 255], uint8, numpy array
    '''
    im_temp = (im.astype(np.float32) - 127.5) / 127.5
    out = torch.from_numpy(im_temp.transpose(2,0,1)).unsqueeze(0)
    return out

# ------------------------Image format--------------------------
def rgb2ycbcr(im, only_y=True):
    '''
    same as matlab rgb2ycbcr
    Input:
        im: uint8 [0,255] or float [0,1]
        only_y: only return Y channel
    '''
    # transform to float64 data type, range [0, 255]
    if im.dtype == np.uint8:
        im_temp = im.astype(np.float64)
    else:
        im_temp = (im * 255).astype(np.float64)

    # convert
    if only_y:
        rlt = np.dot(im_temp, np.array([65.481, 128.553, 24.966])/ 255.0) + 16.0
    else:
        rlt = np.matmul(im_temp, np.array([[65.481,  -37.797, 112.0  ],
                                           [128.553, -74.203, -93.786],
                                           [24.966,  112.0,   -18.214]])/255.0) + [16, 128, 128]
    if im.dtype == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(im.dtype)

def rgb2ycbcrTorch(im, only_y=True):
    '''
    same as matlab rgb2ycbcr
    Input:
        im: float [0,1], N x 3 x H x W
        only_y: only return Y channel
    '''
    # transform to range [0,255.0]
    im_temp = im.permute([0,2,3,1]) * 255.0  # N x H x W x C --> N x H x W x C
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

def bgr2rgb(im): return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

def rgb2bgr(im): return cv2.cvtColor(im, cv2.COLOR_RGB2BGR)

# ------------------------Image I/O-----------------------------
def read_img_lmdb(env, key, size, dtype='uint8'):
    '''
    Read image from give LMDB enviroment.
    out:
        im: h x w x c, numpy tensor, GRB channel
    '''
    with env.begin(write=False) as txn:
        im_buff = txn.get(key.encode())
        im_temp = np.frombuffer(im_buff, dtype=np.dtype(dtype))
        im = im_temp.reshape(size)
    return im

def imread(path, chn='rgb', dtype='float32'):
    '''
    Read image.
    out:
        im: h x w x c, numpy tensor
    '''
    im = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)  # BGR, uint8
    if chn.lower() == 'rgb' and im.ndim == 3:
        im = bgr2rgb(im)

    if dtype == 'float32':
        im = im.astype(np.float32) / 255.
    elif dtype ==  'float64':
        im = im.astype(np.float64) / 255.
    elif dtype == 'uint8':
        pass
    else:
        sys.exit('Please input corrected dtype: float32, float64 or uint8!')

    return im

def imwrite(im, path, chn='rgb', qf=None):
    '''
    Save image.
    Input:
        im: h x w x c, numpy tensor
        path: the saving path
        chn: the channel order of the im,
    '''
    if isinstance(path, str):
        path = Path(path)
    if chn.lower() == 'rgb' and im.ndim == 3:
        im = rgb2bgr(im)

    if qf is not None and path.suffix.lower() in ['.jpg', '.jpeg']:
        flag = cv2.imwrite(str(path), im, [int(cv2.IMWRITE_JPEG_QUALITY), int(qf)])
    else:
        flag = cv2.imwrite(str(path), im)

    return flag

def jpeg_compress(im, qf, chn_in='rgb'):
    '''
    Input:
        im: h x w x 3 array
        qf: compress factor, (0, 100]
        chn_in: 'rgb' or 'bgr'
    Return:
        Compressed Image with channel order: chn_in
    '''
    # transform to BGR channle and uint8 data type
    im_bgr = rgb2bgr(im) if chn_in.lower() == 'rgb' else im
    if im.dtype != np.dtype('uint8'): im_bgr = img_as_ubyte(im_bgr)

    # JPEG compress
    flag, encimg = cv2.imencode('.jpg', im_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), qf])
    assert flag
    im_jpg_bgr = cv2.imdecode(encimg, 1)    # uint8, BGR

    # transform back to original channel and the original data type
    im_out = bgr2rgb(im_jpg_bgr) if chn_in.lower() == 'rgb' else im_jpg_bgr
    if im.dtype != np.dtype('uint8'): im_out = img_as_float32(im_out).astype(im.dtype)
    return im_out

# ------------------------Image Crop-----------------------------
def random_crop_patch(im, pch_size):
    '''
    Randomly crop a patch from the give image.
    '''
    H, W = im.shape[:2]
    ind_H = random.randint(0, H-pch_size)
    ind_W = random.randint(0, W-pch_size)
    im_pch = im[ind_H:ind_H+pch_size, ind_W:ind_W+pch_size,]
    return im_pch

def center_crop(im, pch_size=1024):
    '''
    Crop a patch around the center from the original image.
    Input:
        im: h x w x c or h x w image
    '''
    h, w = im.shape[:2]
    if h > pch_size:
        h_start = (h - pch_size) // 2
        h_end = h_start + pch_size
        im = im[h_start:h_end,]
    if w > pch_size:
        w_start = (w - pch_size) // 2
        w_end = w_start + pch_size
        im = im[:, w_start:w_end,]
    return im

# ------------------------Augmentation-----------------------------
def flipud(x):
    '''
    Flip up and down for tensor.
    x: b x c x h x w
    '''
    ind = list(range(x.shape[2]))[::-1]
    return x[:, :, ind,]

def data_aug_tensor(image, mode):
    '''
    Performs data augmentation of the input image
    Input:
        image: B x c x h x w tensor,
        mode: int. Choice of transformation to apply to the image
                0 - no transformation
                1 - flip up and down
                2 - rotate 90  degree
                3 - rotate 90  degree, flip up and down
                4 - rotate 180 degree
                5 - rotate 180 degree, flip up and down
                6 - rotate 270 degree
                7 - rotate 270 degree, flip up and down
        ------------------------------------------------------------
        0:    A        1:     C
           D     B         D     B
              C               A
        ----------------------------
        2:    D        3:    D
           C     A        C     A
              B              B
        ----------------------------
        4:    C        5:    A
           B     D        B     D
              A              B
        ----------------------------
        6:    B        7:    D
           A     C        A     C
              D              B
    '''
    if mode == 0:
        # original
        out = image
    elif mode == 1:
        # flip up and down
        out = flipud(image)
    elif mode == 2:
        # rotate counterwise 90 degree
        out = torch.rot90(image, k=-1, dims=[2, 3])
    elif mode == 3:
        # rotate 90 degree and flip up and down
        out = torch.rot90(image, k=-1, dims=[2, 3])
        out = flipud(out.contiguous())
    elif mode == 4:
        # rotate 180 degree
        out = torch.rot90(image, k=-2, dims=[2, 3])
    elif mode == 5:
        # rotate 180 degree and flip
        out = torch.rot90(image, k=-2, dims=[2, 3])
        out = flipud(out.contiguous())
    elif mode == 6:
        # rotate 270 degree
        out = torch.rot90(image, k=-3, dims=[2, 3])
    elif mode == 7:
        # rotate 270 degree and flip
        out = torch.rot90(image, k=-3, dims=[2, 3])
        out = flipud(out.contiguous())
    else:
        raise Exception('Invalid choice of image transformation')

    return out

def data_inverse_aug_tensor(image, mode):
    '''
    Inverse process for "data_aug_tensor".
    Input:
        image: B x c x h x w tensor,
        mode: int. Choice of transformation to apply to the image
    '''
    if mode == 0:
        out = image
    elif mode == 1:
        out = flipud(image)
    elif mode == 2:
        out = torch.rot90(image, k=1, dims=[2, 3])
    elif mode == 3:
        out = flipud(image)
        out = torch.rot90(out.contiguous(), k=1, dims=[2, 3])
    elif mode == 4:
        out = torch.rot90(image, k=2, dims=[2, 3])
    elif mode == 5:
        out = flipud(image)
        out = torch.rot90(out.contiguous(), k=2, dims=[2, 3])
    elif mode == 6:
        out = torch.rot90(image, k=3, dims=[2, 3])
    elif mode == 7:
        # rotate 270 degree and flip
        out = flipud(image)
        out = torch.rot90(out.contiguous(), k=3, dims=[2, 3])
    else:
        raise Exception('Invalid choice of image transformation')

    return out

def data_aug_np(image, mode):
    '''
    Performs data augmentation of the input image
    Input:
        image: a cv2 (OpenCV) image
        mode: int. Choice of transformation to apply to the image
                0 - no transformation
                1 - flip up and down
                2 - rotate counterwise 90 degree
                3 - rotate 90 degree and flip up and down
                4 - rotate 180 degree
                5 - rotate 180 degree and flip
                6 - rotate 270 degree
                7 - rotate 270 degree and flip
    '''
    if mode == 0:
        # original
        out = image
    elif mode == 1:
        # flip up and down
        out = np.flipud(image)
    elif mode == 2:
        # rotate counterwise 90 degree
        out = np.rot90(image)
    elif mode == 3:
        # rotate 90 degree and flip up and down
        out = np.rot90(image)
        out = np.flipud(out)
    elif mode == 4:
        # rotate 180 degree
        out = np.rot90(image, k=2)
    elif mode == 5:
        # rotate 180 degree and flip
        out = np.rot90(image, k=2)
        out = np.flipud(out)
    elif mode == 6:
        # rotate 270 degree
        out = np.rot90(image, k=3)
    elif mode == 7:
        # rotate 270 degree and flip
        out = np.rot90(image, k=3)
        out = np.flipud(out)
    else:
        raise Exception('Invalid choice of image transformation')

    return out.copy()

def inverse_data_aug_np(image, mode):
    '''
    Performs inverse data augmentation of the input image
    '''
    if mode == 0:
        # original
        out = image
    elif mode == 1:
        out = np.flipud(image)
    elif mode == 2:
        out = np.rot90(image, axes=(1,0))
    elif mode == 3:
        out = np.flipud(image)
        out = np.rot90(out, axes=(1,0))
    elif mode == 4:
        out = np.rot90(image, k=2, axes=(1,0))
    elif mode == 5:
        out = np.flipud(image)
        out = np.rot90(out, k=2, axes=(1,0))
    elif mode == 6:
        out = np.rot90(image, k=3, axes=(1,0))
    elif mode == 7:
        # rotate 270 degree and flip
        out = np.flipud(image)
        out = np.rot90(out, k=3, axes=(1,0))
    else:
        raise Exception('Invalid choice of image transformation')

    return out

# ----------------------Visualization----------------------------
def imshow(x, title=None, cbar=False):
    plt.imshow(np.squeeze(x), interpolation='nearest', cmap='gray')
    if title:
        plt.title(title)
    if cbar:
        plt.colorbar()
    plt.show()
