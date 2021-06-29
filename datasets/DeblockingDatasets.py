#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2019-11-12 16:25:04

import cv2
import random
import torch
import torch.utils.data as uData
import numpy as np
from .data_tools import random_augmentation, rgb2ycbcr
from pathlib import Path
from skimage import img_as_float32

class DeblockDatasetTrain(uData.Dataset):
    def __init__(self, im_list, tmp_dir, length, pch_size=128):
        '''
        Args:
            im_list (list): path of each image, Pathlib.Path format
            tmp_dir: path to save the JPEG compression images during training
            length (int): length of Datasets
            pch_size (int): patch size of the cropped patch from each image
        '''
        super(DeblockDatasetTrain, self).__init__()
        self.im_list = im_list
        self.tmp_dir = tmp_dir
        self.length = length
        self.pch_size = pch_size
        self.num_images = len(im_list)

    def __len__(self):
        return self.length

    def ranom_quality(self):
        '''
        https://ww2.mathworks.cn/help/images/jpeg-image-deblocking-using-deep-learning.html
        '''
        start = list(range(5, 45, 5)) + [50, 60, 70, 80]
        end = list(range(10, 45, 5)) + [50, 60, 70, 80, 99]
        ind_range = random.randint(0, len(start)-1)
        jpeg_quality = random.uniform(start[ind_range], end[ind_range])
        return jpeg_quality

    def crop_patch(self, im):
        H = im.shape[0]
        W = im.shape[1]
        if H < self.pch_size or W < self.pch_size:
            H = max(self.pch_size, H)
            W = max(self.pch_size, W)
            im = cv2.resize(im, (W, H))
        ind_H = random.randint(0, H-self.pch_size)
        ind_W = random.randint(0, W-self.pch_size)
        pch = im[ind_H:ind_H+self.pch_size, ind_W:ind_W+self.pch_size]
        return pch

    def __getitem__(self, index):
        ind_im = random.randint(0, self.num_images-1)
        im_path = Path(self.im_list[ind_im])
        im_gt_BGR = cv2.imread(str(im_path), cv2.IMREAD_COLOR)
        im_gt_BGR = self.crop_patch(im_gt_BGR)

        # data augmentation
        im_gt_BGR = random_augmentation(im_gt_BGR)[0]

        # jpeg compression
        save_path = self.tmp_dir / (im_path.name.split('.')[-2]+'_'+str(index)+'.jpg')
        jpeg_quality = self.ranom_quality()
        flag = cv2.imwrite(str(save_path), im_gt_BGR, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
        assert flag
        im_jpg_RGB = img_as_float32(cv2.imread(str(save_path), cv2.IMREAD_COLOR)[:, :, ::-1])

        # numpy to torch
        im_gt_RGB = img_as_float32(im_gt_BGR[:, :, ::-1])
        im_jpg = torch.from_numpy(im_jpg_RGB.transpose((2,0,1)).copy()).type(torch.float32)
        im_gt = torch.from_numpy(im_gt_RGB.transpose((2,0,1)).copy()).type(torch.float32)

        return im_jpg, im_gt

class DeblockDatasetTest(uData.Dataset):
    def __init__(self, im_list, tmp_dir, jpeg_quality):
        '''
        Args:
            im_list (list): path of each image, Pathlib.Path format
            tmp_dir: path to save the JPEG compression images during training
            jpeg_quality: integer
        '''
        super(DeblockDatasetTest, self).__init__()
        self.im_list = im_list
        self.tmp_dir = tmp_dir
        self.jpeg_quality = jpeg_quality

    def __len__(self):
        return len(self.im_list)

    def __getitem__(self, index):
        im_path = Path(self.im_list[index])
        im_gt_BGR = cv2.imread(str(im_path), cv2.IMREAD_COLOR)

        # jpeg compression
        save_path = self.tmp_dir / (im_path.name.split('.')[-2]+'.jpg')
        cv2.imwrite(str(save_path), im_gt_BGR, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        im_jpg_RGB = img_as_float32(cv2.imread(str(save_path), cv2.IMREAD_COLOR)[:, :, ::-1])

        # numpy to torch
        im_gt_RGB = img_as_float32(im_gt_BGR[:, :, ::-1])
        im_jpg = torch.from_numpy(im_jpg_RGB.transpose((2,0,1)).copy()).type(torch.float32)
        im_gt = torch.from_numpy(im_gt_RGB.transpose((2,0,1)).copy()).type(torch.float32)

        return im_jpg, im_gt
