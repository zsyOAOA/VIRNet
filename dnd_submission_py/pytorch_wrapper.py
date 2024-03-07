 # Author: Tobias Plötz, TU Darmstadt (tobias.ploetz@visinf.tu-darmstadt.de)

 # This file is part of the implementation as described in the CVPR 2017 paper:
 # Tobias Plötz and Stefan Roth, Benchmarking Denoising Algorithms with Real Photographs.
 # Please see the file LICENSE.txt for the license governing this code.

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

import torch
import numpy as np
from utils import util_image

def pytorch_denoiser(denoiser, use_cuda, flip=False):
    def wrap_denoiser(Inoisy, nlf):
        if flip:
            H, W, C = Inoisy.shape
            denoised_all = np.zeros((H, W, C, 8), dtype = Inoisy.dtype)
            time_all = 0
            for mode in range(8):
                noisy_temp = util_image.data_aug_np(Inoisy, mode)
                noisy_temp = torch.from_numpy(noisy_temp.copy())
                noisy_temp = noisy_temp.permute(2,0,1).unsqueeze_(0)
                if use_cuda:
                    noisy_temp = noisy_temp.cuda()

                with torch.autograd.set_grad_enabled(False):
                    denoised_temp, time_temp = denoiser(noisy_temp, nlf)
                time_all += time_temp
                denoised_temp = denoised_temp[0,...].cpu().numpy().transpose((1,2,0))
                denoised_all[:, :, :, mode] = util_image.inverse_data_aug_np(denoised_temp, mode)
            denoised = np.clip(denoised_all.mean(axis=3, keepdims=False), 0.0, 1.0)
        else:
            noisy = torch.from_numpy(Inoisy)
            noisy = noisy.permute(2,0,1).unsqueeze_(0)
            if use_cuda:
                noisy = noisy.cuda()

            with torch.autograd.set_grad_enabled(False):
                denoised, time_all = denoiser(noisy, nlf)

            denoised = denoised[0,...].cpu().numpy()
            denoised = np.transpose(denoised, [1,2,0])
            denoised = np.clip(denoised, 0.0, 1.0)
        print('Time: {:.2f}'.format(time_all))
        return denoised

    return wrap_denoiser
