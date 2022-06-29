#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2020-03-24 12:27:06

import cv2
import math
import random
import torch
import numpy as np
from pathlib import Path
from skimage import img_as_float32
import torch.utils.data as uData

from utils import util_sisr, util_image
from ResizeRight.resize_right import resize

class GeneralTrainFloder(uData.Dataset):
    def __init__(self, hr_dir, length,
                 hr_size=192,
                 sf=2,
                 k_size=21,
                 kernel_shift=False,
                 downsampler='bicubic',
                 noise_level=[0.1, 15],
                 noise_jpeg=[0.1, 10],
                 add_jpeg=False):
        super(GeneralTrainFloder, self).__init__()
        self.sf = sf
        self.hr_size = hr_size
        self.k_size = k_size
        self.kernel_shift = kernel_shift
        self.downsampler = downsampler
        self.length = length

        # get the image path
        self.hr_path_list = sorted([str(x) for x in Path(hr_dir).glob('*.png')])
        self.num_images = len(self.hr_path_list)

        self.noise_types = ['Gaussian',]
        if add_jpeg:
            self.noise_types.append('JPEG')

        # noise level for Gaussian noise
        assert noise_level[0] < noise_level[1]
        self.noise_level = noise_level
        assert noise_jpeg[0] < noise_jpeg[1]
        self.noise_jpeg = noise_jpeg

    def __len__(self):
        return self.length

    def random_qf(self):
        '''
        https://ww2.mathworks.cn/help/images/jpeg-image-deblocking-using-deep-learning.html
        '''
        start = list(range(30, 50, 5)) + [60, 70, 80]
        end = list(range(35, 50, 5)) + [60, 70, 80, 95]
        ind_range = random.randint(0, len(start)-1)
        qf = random.randint(start[ind_range], end[ind_range])
        return qf

    def reset_seed(self, epoch):
        random.seed(epoch*1000)
        torch.manual_seed(epoch*1000)

    def __getitem__(self, index):
        # obtain HR Image
        hr_size = self.hr_size
        ind_im = random.randint(0, self.num_images-1)
        im_path = self.hr_path_list[ind_im]
        im_ori = util_image.imread(im_path, dtype='float32', chn='rgb')
        im_hr = util_image.random_crop_patch(im_ori, self.hr_size)

        # random augmentation
        aug_flag = random.randint(0, 7)
        im_hr = util_image.data_aug_np(im_hr, aug_flag)

        # blur kernel
        lam1 = random.uniform(0.2, self.sf)
        lam2 = random.uniform(lam1, self.sf) if random.random() < 0.7 else lam1
        theta = random.uniform(0, np.pi)
        kernel, kernel_infos = util_sisr.shifted_anisotropic_Gaussian(k_size=self.k_size,
                                                                      sf=self.sf,
                                                                      lambda_1=lam1**2,
                                                                      lambda_2=lam2**2,
                                                                      theta=theta,
                                                                      shift=self.kernel_shift)

        # blurring
        im_blur = util_sisr.imconv_np(im_hr, kernel, padding_mode='reflect', correlate=False)
        im_blur = np.clip(im_blur, a_min=0.0, a_max=1.0)

        # downsampling
        if self.downsampler.lower() == 'direct':
            im_blur = im_blur[::self.sf, ::self.sf,]
        elif self.downsampler.lower() == 'bicubic':
            im_blur = resize(im_blur, scale_factors=1/self.sf).astype(np.float32)
        else:
            sys.exit('Please input corrected downsampler: Direct or Bicubic')

        # adding noise
        noise_type = random.sample(self.noise_types, k=1)[0]
        if noise_type == 'Gaussian':
            std = random.uniform(self.noise_level[0], self.noise_level[1]) / 255.0
            im_lr = im_blur + torch.randn(im_blur.shape, dtype=torch.float32).numpy() * std
            im_lr = np.clip(im_lr, a_min=0, a_max=1.0)
        elif noise_type == 'JPEG':
            qf = self.random_qf()
            std = random.uniform(self.noise_jpeg[0], self.noise_jpeg[1]) / 255.0
            im_noisy = im_blur + torch.randn(im_blur.shape, dtype=torch.float32).numpy() * std
            im_noisy = np.clip(im_noisy, a_min=0.0, a_max=1.0)
            im_lr = util_image.jpeg_compress(im_noisy, int(qf), chn_in='rgb')
        else:
            sys.exit('Please input corrected noise type: JPEG or Gaussian')

        im_hr = torch.from_numpy(im_hr.transpose([2,0,1])).type(torch.float32)        # c x h x w
        im_lr = torch.from_numpy(im_lr.transpose([2,0,1])).type(torch.float32)        # c x h x w
        im_blur = torch.from_numpy(im_blur.transpose([2,0,1])).type(torch.float32)    # c x h x w
        kernel_infos = torch.from_numpy(kernel_infos).type(torch.float32)             # 3
        nlevel = torch.tensor([std], dtype=torch.float32).unsqueeze(-1).unsqueeze(-1) # 1 x 1 x 1

        return im_hr, im_lr, im_blur, kernel_infos, nlevel

class GeneralTest(uData.Dataset):
    def __init__(self, hr_dir,
                 sf=2,
                 k_size=21,
                 kernel_shift=False,
                 downsampler='bicubic',
                 seed=10000,
                 noise_type='Gaussian'):
        super(GeneralTest, self).__init__()
        self.sf = sf
        self.k_size = k_size
        self.kernel_shift = kernel_shift
        self.downsampler = downsampler
        self.seed = seed

        # get the image path
        self.hr_path_list = sorted([str(x) for x in Path(hr_dir).glob('*.bmp')])
        self.num_images = len(self.hr_path_list)

        self.noise_type = noise_type

        # generate fixed Gaussian noise
        self.fixed_noise = self.generate_noise()

    def __len__(self):
        return self.num_images

    def generate_noise(self):
        h_max, w_max = 1, 1
        for im_path in self.hr_path_list:
            im = util_image.imread(im_path, chn='bgr', dtype='uint8')
            h, w = im.shape[:2]
            if h_max < h: h_max = h
            if w_max < w: w_max = w
        h_down, w_down = math.ceil(h_max / self.sf), math.ceil(w_max / self.sf)

        g =torch.Generator()
        g.manual_seed(self.seed)
        noise = torch.randn([h_down, w_down, 3], generator=g, dtype=torch.float32).numpy()
        return noise

    def __getitem__(self, index):
        im_hr = util_image.imread(self.hr_path_list[index], chn='rgb', dtype='float32')
        if im_hr.ndim == 2 or im_hr.shape[2] == 1:
            im_hr = np.stack([im_hr, im_hr, im_hr], axis=2)
        im_hr = util_sisr.modcrop(im_hr, self.sf)

        # blur kernel
        kernel, kernel_infos = util_sisr.shifted_anisotropic_Gaussian(k_size=self.k_size,
                                                                      sf=self.sf,
                                                                      lambda_1=1.6**2,
                                                                      lambda_2=1.6**2,
                                                                      theta=0,
                                                                      shift=self.kernel_shift)

        # blurring and downsampling
        im_blur = util_sisr.imconv_np(im_hr, kernel, padding_mode='reflect', correlate=False)
        im_blur = np.clip(im_blur, a_min=0.0, a_max=1.0)

        # downsampling
        if self.downsampler.lower() == 'direct':
            im_blur = im_blur[::self.sf, ::self.sf,]
        elif self.downsampler.lower() == 'bicubic':
            im_blur = resize(im_blur, scale_factors=1/self.sf)
        else:
            sys.exit('Please input corrected downsampler: Direct or Bicubic')

        # add noise
        h, w = im_blur.shape[:2]
        if self.noise_type == 'Gaussian':
            im_lr = im_blur + self.fixed_noise[:h, :w,] * (2.55/255)
            im_lr = np.clip(im_lr, a_min=0., a_max=1.0)
        elif self.noise_type == 'JPEG':
            im_noisy = im_blur + self.fixed_noise[:h, :w,] * (2.55/255)
            im_noisy = np.clip(im_noisy, a_min=0.0, a_max=1.0).astype(np.float32)
            im_lr = util_image.jpeg_compress(im_noisy, 40, chn_in='rgb')
        else:
            sys.exit('Please input corrected noise type: JPEG or Gaussian')

        im_hr = torch.from_numpy(im_hr.transpose([2,0,1])).type(torch.float32) # c x h x w
        im_lr = torch.from_numpy(im_lr.transpose([2,0,1])).type(torch.float32) # c x h x w
        kernel_infos = torch.from_numpy(kernel_infos).type(torch.float32)          # 3

        return im_hr, im_lr, kernel_infos

