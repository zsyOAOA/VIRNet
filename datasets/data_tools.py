#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2019-09-02 15:11:05

import cv2
import numpy as np
import random
from skimage import img_as_ubyte
from utils import imshow
import scipy.stats as ss

def data_augmentation(image, mode):
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

    return out

def inverse_data_augmentation(image, mode):
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

def random_augmentation(*args):
    out = []
    if random.randint(0,1) == 1:
        flag_aug = random.randint(1,7)
        for data in args:
            out.append(data_augmentation(data, flag_aug).copy())
    else:
        for data in args:
            out.append(data)
    return out

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

def random_scale():
    flag = random.randint(1,6)
    if flag == 1:
        scale = 2
    elif flag > 1 and flag < 4:
        scale = 3
    else:
        scale = 4
    return int(scale)

if __name__ == '__main__':
    # aa = np.random.randn(4,4)
    # for ii in range(8):
        # bb1 = data_augmentation(aa, ii)
        # bb2 = inverse_data_augmentation(bb1, ii)
        # if np.allclose(aa, bb2):
            # print('Flag: {:d}, Sccessed!'.format(ii))
        # else:
            # print('Flag: {:d}, Failed!'.format(ii))

    import matplotlib.pyplot as plt
    l1_max_root = 8
    print('L1_Max={:3d}'.format(l1_max_root))
    for ii in range(20):
        l1 = 0.1 + random.random() * l1_max_root
        l2 = 0.1 + random.random() * (l1-0.1)
        theta = random.random() * np.pi
        print('l1={:.2f}, l2={:.2f}, theta={:.2f}'.format(l1, l2, theta))
        kernel = anisotropic_Gaussian(25, theta, l1**2, l2**2)
        plt.figure('Kernel')
        imshow(kernel)
        plt.pause(0.5)

