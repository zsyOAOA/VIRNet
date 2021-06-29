#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2020-03-24 12:27:06

import cv2
import random
import torch
from math import sqrt
import torch.nn.functional as F
import numpy as np
import h5py as h5
from skimage import img_as_float32
from .imresize import imresize
from . import BaseDataSetImg, BaseDataSetH5
from .data_tools import anisotropic_Gaussian, random_scale, random_augmentation
from utils import getGaussianKernel2D

class GeneralTrainFloder(BaseDataSetImg):
    def __init__(self, im_list, length, HR_size=96, scale=2, kernel_size=15, sigma_range=[0,25]):
        super(GeneralTrainFloder, self).__init__(im_list, length)
        self.HR_size = HR_size
        self.num_images = len(im_list)
        assert sigma_range[0] < sigma_range[1]
        self.sigma_min = sigma_range[0]
        self.sigma_max = sigma_range[1]
        self.kernel_size = kernel_size
        if scale == 2:
            self.l1_max_root = 4
        elif scale == 3:
            self.l1_max_root = 6
        elif scale == 4:
            self.l1_max_root = 8
        else:
            sys.exit('Please input corrected scale factor')

        self.pad = int((kernel_size-1)*0.5)

    def crop_patch(self, im):
        H, W, _ = im.shape
        pch_size = self.HR_size + self.pad * 2
        ind_H = random.randint(0, H-pch_size)
        ind_W = random.randint(0, W-pch_size)
        im_crop = im[ind_H:ind_H+pch_size, ind_W:ind_W+pch_size]
        return im_crop

    def __getitem__(self, index):
        HR_size = self.HR_size
        ind_im = random.randint(0, self.num_images-1)
        im_ori = cv2.imread(self.im_list[ind_im], cv2.IMREAD_COLOR)[:, :, ::-1]
        im_ori = img_as_float32(im_ori)

        H, W, _ = im_ori.shape
        if H < (HR_size+self.pad*2) or W < (HR_size+self.pad*2):
            H = max(H, HR_size+self.pad*2)
            W = max(W, HR_size+self.pad*2)
            im_ori = cv2.resize(im_ori, [W, H], interpolation=cv2.INTER_CUBIC)

        # sample noise_level
        if random.random() < 0.1:
            noise_level = 0.01 / 255.0
        else:
            noise_level = random.uniform(self.sigma_min+0.01, self.sigma_max) / 255.0

        # sample blur kernel
        if random.random() < 0.5:
            theta = random.random() * np.pi
            l1_root = 0.1 + random.random()*(self.l1_max_root-0.1)
            l2_root = 0.1 + random.random()*(l1_root-0.1)
            kernel = anisotropic_Gaussian(self.kernel_size, theta, l1_root**2, l2_root**2)
        else:
            sigma = 0.1 + random.random() * (self.l1_max_root-0.1)
            kernel = getGaussianKernel2D(self.kernel_size, sigma)

        im_HR = self.crop_patch(im_ori)

        # augmentation
        im_HR = random_augmentation(im_HR)[0]

        im_HR = torch.from_numpy(im_HR.transpose([2,0,1])).type(torch.float32) # C x H x W
        kernel = torch.from_numpy(kernel[np.newaxis]).type(torch.float32)  # 1 x k x k
        noise_level = torch.tensor([noise_level]).reshape([1,1,1]).type(torch.float32) # 1 x 1 x 1

        return im_HR, noise_level, kernel

class GeneralTrainH5(BaseDataSetH5):
    def __init__(self, h5_path, length, HR_size=96, scale=2, kernel_size=15, sigma_range=[0,25]):
        super(GeneralTrainH5, self).__init__(h5_path, length)
        self.HR_size = HR_size
        assert sigma_range[0] < sigma_range[1]
        self.sigma_min = sigma_range[0]
        self.sigma_max = sigma_range[1]
        self.kernel_size = kernel_size
        self.l1_max = scale * 4.0
        self.pad = int((kernel_size-1)*0.5)

    def crop_patch(self, im):
        H, W, _ = im.shape
        pch_size = self.HR_size + self.pad * 2
        ind_H = random.randint(0, H-pch_size)
        ind_W = random.randint(0, W-pch_size)
        im_crop = im[ind_H:ind_H+pch_size, ind_W:ind_W+pch_size]
        return im_crop

    def __getitem__(self, index):
        HR_size = self.HR_size
        num_images = self.num_images
        ind_im = random.randint(0, num_images-1)
        with h5.File(self.h5_path, 'r') as h5_file:
            im_ori = np.array(h5_file[self.keys[ind_im]])

        im_ori = img_as_float32(im_ori)

        H, W, _ = im_ori.shape
        if H < (HR_size+self.pad*2) or W < (HR_size+self.pad*2):
            H = max(H, HR_size+self.pad*2)
            W = max(W, HR_size+self.pad*2)
            im_ori = cv2.resize(im_ori, [W, H], interpolation=cv2.INTER_CUBIC)

        # sample noise_level
        if random.random() < 0.1:
            noise_level = 0.01 / 255.0
        else:
            noise_level = random.uniform(self.sigma_min+0.01, self.sigma_max) / 255.0

        # sample blur kernel
        if random.random() < 0.5:
            theta = random.random() * np.pi
            l1 = 0.1 + random.random()*(self.l1_max-0.1)
            l2 = 0.1 + random.random()*(l1-0.1)
            kernel = anisotropic_Gaussian(self.kernel_size, theta, l1, l2)
        else:
            sigma = 0.1 + random.random() * (sqrt(self.l1_max)-0.1)
            kernel = getGaussianKernel2D(self.kernel_size, sigma)

        im_HR = self.crop_patch(im_ori)

        # augmentation
        im_HR = random_augmentation(im_HR)[0]

        im_HR = torch.from_numpy(im_HR.transpose([2,0,1])).type(torch.float32) # C x H x W
        kernel = torch.from_numpy(kernel[np.newaxis]).type(torch.float32)  # 1 x k x k
        noise_level = torch.tensor([noise_level]).reshape([1,1,1]).type(torch.float32) # 1 x 1 x 1

        return im_HR, noise_level, kernel

class GeneralTest(BaseDataSetImg):
    def __init__(self, HR_list, scale=2):
        super(GeneralTest, self).__init__(HR_list, length=None)
        self.HR_list = HR_list
        self.num_images = len(HR_list)
        self.scale = scale

    def __len__(self):
        return self.num_images

    def modecrop(self, im):
        scale = self.scale

        H, W , _ = im.shape
        H -= (H % scale)
        W -= (W % scale)

        return im[:H, :W,]

    def __getitem__(self, index):
        scale = self.scale

        im_HR = cv2.imread(self.im_list[index], cv2.IMREAD_COLOR)[:, :, ::-1]
        im_HR = self.modecrop(im_HR)
        im_HR = img_as_float32(im_HR)
        im_HR = torch.from_numpy(im_HR.transpose([2,0,1])).type(torch.float32)

        return im_HR
