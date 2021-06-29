#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2019-05-16 16:20:01

import torch
import numpy as np
from utils import imshow
from skimage import img_as_float
from scipy.io import loadmat
from networks.VIRNet import VIRNetU

print('Load the testing data')
im_noisy = loadmat('./test_data/DND/1.mat')['InoisySRGB']

C = 3
# load the pretrained model
print('Loading the Model')
net = VIRNetU(C, wf=64, dep_U=4, dep_S=8).cuda()
net.load_state_dict(torch.load('./model_zoo/model_denoising_real.pt', map_location='cpu'))
net.eval()

# begin testing
inputs = torch.from_numpy(im_noisy.transpose((2,0,1))[np.newaxis,]).type(torch.float32).cuda()
with torch.autograd.set_grad_enabled(False):
    phi_Z = net(inputs, 'test')
    im_denoise = np.clip(phi_Z[:, :C,].cpu().numpy().squeeze().transpose([1,2,0]), 0.0, 1.0)

im_all = np.concatenate([im_noisy, im_denoise], 1)
imshow(im_all)

