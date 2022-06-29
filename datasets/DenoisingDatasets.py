#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2019-09-02 15:51:11

import os
import torch
import h5py as h5
import random
import cv2
import lmdb
import numpy as np
from pathlib import Path
from scipy.io import loadmat
import torch.utils.data as uData
from skimage import img_as_float32

from . import BaseDataSetH5, BaseDataSetImg
from utils import util_image
from utils import util_denoising

class DataLMDB(uData.Dataset):
    def __init__(self, lmdb_dir, length, pch_size=128, sidd=True, renoir=False, polyu=False):
        super(DataLMDB, self).__init__()
        self.sidd_flag = sidd
        self.renoir_flag = renoir
        self.polyu_flag = polyu
        self.lmdb_dir = lmdb_dir
        self.length = length
        self.pch_size = pch_size

        self.env, self.keys_noisy, self.keys_gt = None, None, None
        self.num_images = self.get_num_images()

    def get_num_images(self):
        if self.keys_noisy is None:
            self._get_keys()
        return len(self.keys_noisy)

    def _init_lmdb(self):
        self.env = lmdb.open(self.lmdb_dir,
                             readonly=True,
                             lock=False,
                             readahead=False,
                             meminit=False)

    def _get_keys(self):
        if self.env is None:
            self._init_lmdb()

        with self.env.begin(write=False) as txn:
            with txn.cursor() as curs:
                keys_all = []
                if self.sidd_flag:
                    keys_all += [x.decode() for x, _ in curs if 'sidd' in x.decode()]
                if self.renoir_flag:
                    keys_all += [x.decode() for x, _ in curs if 'renoir' in x.decode()]
                if self.polyu_flag:
                    keys_all += [x.decode() for x, _ in curs if 'polyu' in x.decode()]

        self.keys_noisy = sorted([x for x in keys_all if 'noisy' in x])
        self.keys_gt = [x.replace('noisy', 'gt') for x in self.keys_noisy]

        assert len(self.keys_noisy) == len(self.keys_gt)
        self.close_env()

    def close_env(self):
        if self.env is not None:
            self.env.close()
            self.env = None

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        if self.env is None or self.keys_noisy is None:
            self._init_lmdb()

        ind_im = random.randint(0, self.num_images-1)

        # read noisy image
        key_noisy = self.keys_noisy[ind_im]
        im_noisy = util_image.read_img_lmdb(self.env, key_noisy)  # h x w x 3, uint8, numpy

        # read clean image
        key_gt = self.keys_gt[ind_im]
        im_gt = util_image.read_img_lmdb(self.env, key_gt)        # h x w x 3, uint8, numpy

        # random crop
        temp = util_image.random_crop_patch(np.concatenate([im_noisy, im_gt], axis=2), self.pch_size)
        im_noisy, im_gt = np.split(temp, 2, axis=2)

        # random augmentation
        aug_flag = random.randint(0, 7)
        im_noisy, im_gt = [util_image.data_aug_np(x, aug_flag) for x in [im_noisy, im_gt]]

        im_noisy = torch.from_numpy(img_as_float32(im_noisy.transpose([2, 0, 1])))
        im_gt = torch.from_numpy(img_as_float32(im_gt.transpose([2, 0, 1])))

        return im_noisy, im_gt

class RealTrain(uData.Dataset):
    def __init__(self, noisy_dir, length, pch_size=128, sidd=True, renoir=False, polyu=False):
        super(RealTrain, self).__init__()
        self.sidd_flag = sidd
        self.renoir_flag = renoir
        self.polyu_flag = polyu
        self.length = length
        self.noisy_dir = noisy_dir
        self.pch_size = pch_size

        self.noisy_path_list, self.gt_path_list = self.extract_im_path()
        self.num_images = len(self.noisy_path_list)

    def extract_im_path(self):
        noisy_path_list_all = [x for x in Path(self.noisy_dir).glob('*.png')]
        noisy_path_list = []
        if self.sidd_flag:
            noisy_path_list.extend([str(x) for x in noisy_path_list_all if 'sidd' in x.stem])
        if self.renoir_flag:
            noisy_path_list.extend([str(x) for x in noisy_path_list_all if 'renoir' in x.stem])
        if self.polyu_flag:
            noisy_path_list.extend([str(x) for x in noisy_path_list_all if 'polyu' in x.stem])

        gt_path_list = []
        for noisy_path in noisy_path_list:
            gt_path = str(Path(noisy_path).parents[1] / 'gt' / Path(noisy_path).name)
            gt_path_list.append(gt_path)

        return noisy_path_list, gt_path_list

    def reset_seed(self, seed):
        random.seed(seed)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        ind_im = random.randint(0, self.num_images-1)

        # read image
        im_noisy = util_image.imread(self.noisy_path_list[ind_im], chn='rgb', dtype='uint8')  # h x w x 3, uint8, numpy
        im_gt = util_image.imread(self.gt_path_list[ind_im], chn='rgb', dtype='uint8')

        # random crop
        temp = util_image.random_crop_patch(np.concatenate([im_noisy, im_gt], axis=2), self.pch_size)
        im_noisy, im_gt = np.split(temp, 2, axis=2)

        # random augmentation
        aug_flag = random.randint(0, 7)
        im_noisy, im_gt = [util_image.data_aug_np(x, aug_flag) for x in [im_noisy, im_gt]]

        im_noisy = torch.from_numpy(img_as_float32(im_noisy.transpose([2, 0, 1])))
        im_gt = torch.from_numpy(img_as_float32(im_gt.transpose([2, 0, 1])))

        return im_noisy, im_gt

class BenchmarkTest(uData.Dataset):
    def __init__(self, noisy_path, gt_path):
        super(BenchmarkTest, self).__init__()
        self.im_noisy_all = loadmat(noisy_path)['ValidationNoisyBlocksSrgb']
        self.im_gt_all = loadmat(gt_path)['ValidationGtBlocksSrgb']

        h, w, c = self.im_noisy_all.shape[2:]
        self.im_noisy_all = self.im_noisy_all.reshape([-1, h, w, c])
        self.im_gt_all = self.im_gt_all.reshape([-1, h, w, c])

    def __len__(self):
        return self.im_noisy_all.shape[0]

    def __getitem__(self, index):
        im_gt = img_as_float32(self.im_gt_all[index])
        im_noisy = img_as_float32(self.im_noisy_all[index])

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
        kernel = util_denoising.getGaussianKernel2DCenter(pch_size, pch_size, center, scale)
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

    def reset_seed(self, seed):
        random.seed(seed)
        torch.manual_seed(seed)

    def __getitem__(self, index):
        pch_size = self.pch_size
        ind_im = random.randint(0, self.num_images-1)

        if self.chn == 3:
            im_ori = cv2.imread(self.im_list[ind_im], cv2.IMREAD_COLOR)[:, :, ::-1]
        else:
            im_ori = cv2.imread(self.im_list[ind_im], cv2.IMREAD_GRAYSCALE)[:, :, np.newaxis]
        im_gt = img_as_float32(self.crop_patch(im_ori))  # H x W x C

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

        aug_flag = random.randint(0, 7)
        im_gt, im_noisy, sigma_map = [util_image.data_aug_np(x, aug_flag) for x in [im_gt, im_noisy, sigma_map]]

        sigma_map_gt = np.square(sigma_map)          # H x W x 1
        # Groundtruth SigmaMap
        sigma_map_gt = np.where(sigma_map_gt<1e-10, 1e-10, sigma_map_gt)
        sigma_map_gt = torch.from_numpy(sigma_map_gt.transpose((2, 0, 1)))

        im_gt = torch.from_numpy(im_gt.transpose((2, 0, 1)).copy())
        im_noisy = torch.from_numpy(im_noisy.transpose((2, 0, 1)).copy())

        return im_noisy, im_gt, sigma_map_gt

class SimulateTest(uData.Dataset):
    def __init__(self, im_list, seed=1000):
        super(SimulateTest, self).__init__()
        self.im_list = im_list
        self.seed = seed
        self.noise = self.generate_noise()
        self.sigma_map = self.generate_sigma_map()

    def __len__(self):
        return len(self.im_list)

    def generate_noise(self):
        h_max, w_max = 1, 1
        for im_path in self.im_list:
            im = util_image.imread(im_path, chn='bgr', dtype='uint8')
            h, w = im.shape[:2]
            if h_max < h: h_max = h
            if w_max < w: w_max = w

        rng = np.random.default_rng(seed=self.seed)
        noise = rng.standard_normal(size=[h_max, w_max, 3], dtype=np.float32)
        return noise

    def generate_sigma_map(self):
        kernel = util_denoising.peaks(256)
        down, up = 10/255., 75/255.
        sigma_map = down + (kernel-kernel.min())/(kernel.max()-kernel.min())  *(up-down)
        sigma_map = sigma_map.astype(np.float32)

        return sigma_map

    def __getitem__(self, index):
        im_gt = util_image.imread(self.im_list[index], chn='rgb', dtype='float32')

        h, w, _ = im_gt.shape
        sigma = cv2.resize(self.sigma_map, (w, h), interpolation=cv2.INTER_NEAREST_EXACT) # H x W
        im_noisy = im_gt + self.noise[:h, :w,] * sigma[:, :, np.newaxis]

        im_gt = torch.from_numpy(im_gt.transpose((2, 0, 1))).type(torch.float32)
        im_noisy = torch.from_numpy(im_noisy.transpose((2, 0, 1))).type(torch.float32)

        return im_noisy, im_gt

