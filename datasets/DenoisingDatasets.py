#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2019-09-02 15:51:11

import torch
import h5py as h5
import random
import cv2
import os
import numpy as np
import torch.utils.data as uData
from skimage import img_as_float32 as img_as_float
from .data_tools import random_augmentation
from utils import getGaussianKernel2DCenter
from . import BaseDataSetH5, BaseDataSetImg

# Benchmardk Datasets: Renoir and SIDD
class BenchmarkTrain(BaseDataSetH5):
    def __init__(self, h5_file, length, pch_size=128):
        super(BenchmarkTrain, self).__init__(h5_file, length)
        self.pch_size = pch_size

    def __getitem__(self, index):
        num_images = self.num_images
        ind_im = random.randint(0, num_images-1)

        with h5.File(self.h5_path, 'r') as h5_file:
            imgs_sets = h5_file[self.keys[ind_im]]
            im_gt, im_noisy = self.crop_patch(imgs_sets)
        im_gt = img_as_float(im_gt)
        im_noisy = img_as_float(im_noisy)

        # data augmentation
        im_gt, im_noisy = random_augmentation(im_gt, im_noisy)

        im_gt = torch.from_numpy(im_gt.transpose((2, 0, 1))).type(torch.float32)
        im_noisy = torch.from_numpy(im_noisy.transpose((2, 0, 1))).type(torch.float32)

        return im_noisy, im_gt

class PolyuTrain(BaseDataSetImg):
    def __init__(self, path_list, length, pch_size=128):
        super(PolyuTrain, self).__init__(path_list, length, pch_size)

    def __getitem__(self, index):
        num_images = self.num_images
        ind_im = random.randint(0, num_images-1)

        path_noisy = self.im_list[ind_im]
        head, tail = os.path.split(path_noisy)
        path_gt = os.path.join(head, tail.replace('real', 'mean'))
        im_noisy = img_as_float(cv2.imread(path_noisy, 1)[:, :, ::-1])
        im_gt = img_as_float(cv2.imread(path_gt, 1)[:, :, ::-1])
        im_noisy, im_gt = self.crop_patch(im_noisy, im_gt)

        # data augmentation
        im_gt, im_noisy = random_augmentation(im_gt, im_noisy)

        im_gt = torch.from_numpy(im_gt.transpose((2, 0, 1)))
        im_noisy = torch.from_numpy(im_noisy.transpose((2, 0, 1)))

        return im_noisy, im_gt

    def crop_patch(self, im_noisy, im_gt):
        pch_size = self.pch_size
        H, W, _ = im_noisy.shape
        ind_H = random.randint(0, H-pch_size)
        ind_W = random.randint(0, W-pch_size)
        im_pch_noisy = im_noisy[ind_H:ind_H+pch_size, ind_W:ind_W+pch_size,]
        im_pch_gt = im_gt[ind_H:ind_H+pch_size, ind_W:ind_W+pch_size,]
        return im_pch_noisy, im_pch_gt

class BenchmarkTest(BaseDataSetH5):
    def __getitem__(self, index):
        with h5.File(self.h5_path, 'r') as h5_file:
            imgs_sets = h5_file[self.keys[index]]
            C2 = imgs_sets.shape[2]
            C = int(C2/2)
            im_noisy = np.array(imgs_sets[:, :, :C])
            im_gt = np.array(imgs_sets[:, :, C:])
        im_gt = img_as_float(im_gt)
        im_noisy = img_as_float(im_noisy)

        im_gt = torch.from_numpy(im_gt.transpose((2, 0, 1)))
        im_noisy = torch.from_numpy(im_noisy.transpose((2, 0, 1)))

        return im_noisy, im_gt

# Simulation Datasets:
class SimulateTrain(BaseDataSetImg):
    def __init__(self, im_list, length,  pch_size=128, chn=3, mode='niid', clip=False):
        super(SimulateTrain, self).__init__(im_list, length,  pch_size)
        self.num_images = len(im_list)
        self.sigma_min = 0
        self.sigma_max = 75
        self.chn = chn
        self.mode = mode
        self.clip = clip

    def generate_sigma_niid(self):
        pch_size = self.pch_size
        center = [random.uniform(0, pch_size), random.uniform(0, pch_size)]
        scale = random.uniform(pch_size/4, pch_size/4*3)
        kernel = getGaussianKernel2DCenter(pch_size, pch_size, center, scale)
        up = random.uniform(self.sigma_min/255.0, self.sigma_max/255.0)
        down = random.uniform(self.sigma_min/255.0, self.sigma_max/255.0)
        if up < down:
            up, down = down, up
        up += 5/255.0
        sigma_map = down + (kernel-kernel.min())/(kernel.max()-kernel.min())  *(up-down)
        sigma_map = sigma_map.astype(np.float32)

        return sigma_map[:, :, np.newaxis]

    def generate_sigma_iid(self):
        pch_size = self.pch_size
        noise_level = random.uniform(self.sigma_min/255.0, self.sigma_max/255.0)
        sigma_map = np.ones([pch_size, pch_size]) * noise_level
        sigma_map = sigma_map.astype(np.float32)

        return sigma_map[:, :, np.newaxis]

    def __getitem__(self, index):
        pch_size = self.pch_size
        ind_im = random.randint(0, self.num_images-1)

        if self.chn == 3:
            im_ori = cv2.imread(self.im_list[ind_im], cv2.IMREAD_COLOR)[:, :, ::-1]
        else:
            im_ori = cv2.imread(self.im_list[ind_im], cv2.IMREAD_GRAYSCALE)[:, :, np.newaxis]
        im_gt = img_as_float(self.crop_patch(im_ori))  # H x W x C

        # generate sigmaMap
        if self.mode.lower() == 'niid':
            sigma_map = self.generate_sigma_niid()    # H x W x 1
        elif self.mode.lower() == 'iid':
            sigma_map = self.generate_sigma_iid()    # H x W x 1
        else:
            sys.exit('Plsase Input corrected noise type: iid or niid')

        # generate noise
        noise = torch.randn(im_gt.shape).numpy() * sigma_map
        if self.clip:
            im_noisy = np.clip(im_gt + noise.astype(np.float32), 0.0, 1.0)
        else:
            im_noisy = im_gt + noise.astype(np.float32)

        im_gt, im_noisy, sigma_map = random_augmentation(im_gt, im_noisy, sigma_map)

        if self.chn == 3:
            sigma_map_gt = np.tile(np.square(sigma_map), (1, 1, 3))  # H x W x 3
        else:
            sigma_map_gt = np.square(sigma_map)          # H x W x 3
        # Groundtruth SigmaMap
        sigma_map_gt = np.where(sigma_map_gt<1e-10, 1e-10, sigma_map_gt)
        sigma_map_gt = torch.from_numpy(sigma_map_gt.transpose((2, 0, 1)))

        im_gt = torch.from_numpy(im_gt.transpose((2, 0, 1)))
        im_noisy = torch.from_numpy(im_noisy.transpose((2, 0, 1)))

        return im_noisy, im_gt, sigma_map_gt

class SimulateTest(uData.Dataset):
    def __init__(self, im_list, h5_path, chn=3):
        super(SimulateTest, self).__init__()
        self.im_list = im_list
        self.h5_path = h5_path
        self.chn = chn

    def __len__(self):
        return len(self.im_list)

    def __getitem__(self, index):
        if self.chn == 3:
            im_gt = cv2.imread(self.im_list[index], cv2.IMREAD_COLOR)[:, :, ::-1]
        else:
            im_gt = cv2.imread(self.im_list[index], cv2.IMREAD_GRAYSCALE)[:, :, np.newaxis]
        im_key = os.path.basename(self.im_list[index]).split('.')[0]
        C = im_gt.shape[2]

        with h5.File(self.h5_path, 'r') as h5_file:
            noise = np.array(h5_file[im_key][:,:,:C])
        H, W, _ = noise.shape
        im_gt = img_as_float(im_gt[:H, :W])
        im_noisy = im_gt + noise

        im_gt = torch.from_numpy(im_gt.transpose((2, 0, 1))).type(torch.float32)
        im_noisy = torch.from_numpy(im_noisy.transpose((2, 0, 1))).type(torch.float32)

        return im_noisy, im_gt

