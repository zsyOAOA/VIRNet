#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2019-01-10 22:41:49

import os
import commentjson as json
import sys
import time
import random
import warnings
import numpy as np
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.utils as vutils
from utils import *
from torch.utils.tensorboard import SummaryWriter
from math import ceil
from loss.ELBO import elbo_sisr, degrademodel
from networks.VIRNet import VIRNetSISR
from datasets.SISRDatasets import GeneralTrainH5, GeneralTest
from pathlib import Path

# filter warnings
warnings.simplefilter('ignore', Warning, lineno=0)

_C = 3
_modes = ['train', 'val']
_fix_noise = loadFixNoise()

def train_model(net, datasets, optimizer, lr_scheduler, blur_kernel_test, criterion, args):
    clip_grad = args['clip_grad']
    batch_size = {_modes[0]: args['batch_size'], _modes[1]: 1}
    data_loader = {phase: torch.utils.data.DataLoader(datasets[phase], batch_size=batch_size[phase],
       shuffle=True, num_workers=args['num_workers'], pin_memory=True) for phase in datasets.keys()}
    num_data = {phase: len(datasets[phase]) for phase in datasets.keys()}
    num_iter_epoch = {phase: ceil(num_data[phase] / batch_size[phase]) for phase in datasets.keys()}
    writer = SummaryWriter(args['log_dir'])
    x = vutils.make_grid(blur_kernel_test, normalize=True, scale_each=True)
    writer.add_image('Test'+' Blur kernel', x, 1)
    if args['resume']:
        step = args['step']
        step_img = args['step_img']
    else:
        step = 0
        step_img = {x: 0 for x in _modes}

    ksize = 2*args['radius'] + 1
    kernel_avg = torch.ones([1, 1, ksize, ksize], dtype=torch.float32).expand([3,1,ksize,ksize])/(ksize**2)
    for epoch in range(args['epoch_start'], args['epochs']):
        loss_per_epoch = {x: 0 for x in ['Loss', 'lh', 'KLG', 'KLIG']}
        mse_per_epoch = grad_norm = 0
        tic = time.time()
        # train stage
        net.train()
        lr = optimizer.param_groups[0]['lr']
        phase = 'train'
        for ii, data in enumerate(data_loader[phase]):
            # prepare data
            im_HR, nlevel, blur_kernel = [x.cuda() for x in data] # NxCxHxW, Nx1x1x1, Nx1xkxk
            im_LR, im_blur = degrademodel(im_HR, blur_kernel, args['scale'], padding=False,
                                           noise_level=nlevel, downsampler=args['downsampler'])
            pad = int((blur_kernel.shape[-1]-1)*0.5)
            im_HR = im_HR[:, :, pad:im_HR.shape[2]-pad, pad:im_HR.shape[3]-pad]
            varmap_est = var_estimate(im_LR-im_blur, kernel_avg.to(im_LR.device))
            Nv, _, Hv, Wv = varmap_est.shape
            varmap_est = varmap_est.mean(dim=[1,2,3], keepdim=True).expand([Nv, 1, Hv, Wv])
            # network optimization
            optimizer.zero_grad()
            out_Z, out_sigma = net(im_LR, blur_kernel, args['scale'], pad=False)
            loss, g_lh, kl_g, kl_Ig = criterion(out_Z, out_sigma, im_LR, im_HR, blur_kernel, varmap_est,
                                      args['scale'], args['downsampler'], args['eps2'], args['radius'])
            loss.backward()
            # clip the gradnorm
            total_norm_D = nn.utils.clip_grad_norm_(net.parameters(), clip_grad)
            grad_norm = (grad_norm*(ii/(ii+1)) + total_norm_D/(ii+1))
            optimizer.step()

            varmap_pre = torch.exp(out_sigma.data[:, 1,]) / (torch.exp(out_sigma.data[:, 0,])+1)
            loss_per_epoch['Loss'] += loss.item() / num_iter_epoch[phase]
            loss_per_epoch['lh'] += g_lh.item() / num_iter_epoch[phase]
            loss_per_epoch['KLG'] += kl_g.item() / num_iter_epoch[phase]
            loss_per_epoch['KLIG'] += kl_Ig.item() / num_iter_epoch[phase]
            im_HR_est = out_Z[:, :_C, ].detach().data
            mse = F.mse_loss(im_HR_est, im_HR)
            mse_per_epoch += mse
            if (ii+1) % args['print_freq'] == 0:
                log_str = '[Epoch:{:>2d}/{:<2d}] {:s}:{:0>4d}/{:0>4d}, lh={:>6.2f}, ' + \
                            'KLG={:+>7.2f}, KLIG={:+>7.2f}, mse={:.2e}, NLevel:T/E/P:{:.2e}/{:.2e}/{:.2e},'+\
                                                                 ' GNorm_D:{:.1e}/{:.1e}, lr={:.1e}'
                print(log_str.format(epoch+1, args['epochs'], phase, ii+1, num_iter_epoch[phase],
                                       g_lh.item(), kl_g.item(), kl_Ig.item(), mse, (nlevel).mean(),
                                                          torch.sqrt(varmap_est[:, :, 0, 0]).mean(),
                                                           torch.sqrt(varmap_pre).mean(), clip_grad,
                                                                                  total_norm_D, lr))
                writer.add_scalar('Train Loss Iter', loss.item(), step)
                writer.add_scalar('Train MSE Iter', mse, step)
                step += 1
            if (ii+1) % (10*args['print_freq']) == 0:
                x1 = vutils.make_grid(im_HR_est, normalize=True, scale_each=True)
                writer.add_image(phase+' Recover Image', x1, step_img[phase])
                x2 = vutils.make_grid(im_HR, normalize=True, scale_each=True)
                writer.add_image(phase+' HR Image', x2, step_img[phase])
                x3 = vutils.make_grid(blur_kernel, normalize=True, scale_each=True)
                writer.add_image(phase+' Blur Kernel', x3, step_img[phase])
                x4 = vutils.make_grid(im_LR, normalize=True, scale_each=True)
                writer.add_image(phase+' LR Image', x4, step_img[phase])
                step_img[phase] += 1

        mse_per_epoch /= (ii+1)
        log_str ='{:s}: Loss={:+.2e}, lh={:+.2e}, KL_Guass={:+.2e}, mse={:.3e}, GNorm_D={:.1e}/{:.1e}'
        print(log_str.format(phase, loss_per_epoch['Loss'], loss_per_epoch['lh'],
                                        loss_per_epoch['KLG'], mse_per_epoch, clip_grad, grad_norm))
        writer.add_scalar('Loss_epoch', loss_per_epoch['Loss'], epoch)
        clip_grad = min(clip_grad, grad_norm)
        print('-'*105)

        # test stage
        phase = 'val'
        net.eval()
        psnr_per_epoch = ssim_per_epoch = mse_per_epoch = 0
        for ii, data in enumerate(data_loader[phase]):
            im_HR = data.cuda()
            with torch.set_grad_enabled(False):
                im_LR, _ = degrademodel(im_HR, blur_kernel_test, args['scale'], padding=True,
                                                  noise_level=None, downsampler=args['downsampler'])
                im_LR += _fix_noise[:, :, :im_LR.shape[2], :im_LR.shape[3]].to(im_LR.device)*(args['noise_level_test']/255.0)
                out_sisr, _ = net(im_LR, blur_kernel_test, args['scale'], pad=True)
                im_HR_est = out_sisr[:, :_C,]

            mse = F.mse_loss(im_HR_est, im_HR)
            mse_per_epoch += mse
            psnr_iter = batch_PSNR(im_HR_est, im_HR, args['scale']**2, True)
            ssim_iter = batch_SSIM(im_HR_est, im_HR, args['scale']**2, True)
            psnr_per_epoch += psnr_iter
            ssim_per_epoch += ssim_iter
            # print statistics every log_interval mini_batches
            if (ii+1) % 3 == 0:
                log_str = '[Epoch:{:>3d}/{:<3d}] {:s}:{:0>5d}/{:0>5d}, mse={:.2e}, ' + \
                                                        'Nlevel:{:.2f}, psnr={:4.2f}, ssim={:5.4f}'
                print(log_str.format(epoch+1, args['epochs'], phase, ii+1, num_iter_epoch[phase],
                                               mse, args['noise_level_test'], psnr_iter, ssim_iter))
            # tensorboardX summary
                x1 = vutils.make_grid(im_HR_est, normalize=True, scale_each=True)
                writer.add_image(phase+' Recover images', x1, step_img[phase])
                x2 = vutils.make_grid(im_HR, normalize=True, scale_each=True)
                writer.add_image(phase+' HR Image', x2, step_img[phase])
                x3 = vutils.make_grid(im_LR, normalize=True, scale_each=True)
                writer.add_image(phase+' LR Image', x3, step_img[phase])
                step_img[phase] += 1

        psnr_per_epoch /= (ii+1)
        ssim_per_epoch /= (ii+1)
        mse_per_epoch /= (ii+1)
        log_str = '{:s}: mse={:.3e}, PSNR={:4.2f}, SSIM={:5.4f}'
        print(log_str.format(phase, mse_per_epoch, psnr_per_epoch, ssim_per_epoch))
        print('-'*105)

        # adjust the learning rate
        lr_scheduler.step()
        # save model
        model_prefix = 'model_'
        save_path_model = os.path.join(args['model_dir'], model_prefix+str(epoch+1))
        torch.save({
            'epoch': epoch+1,
            'step': step+1,
            'step_img': {x: step_img[x] for x in _modes},
            'grad_norm': clip_grad,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'lr_scheduler_state_dict': lr_scheduler.state_dict()
        }, save_path_model)
        model_state_prefix = 'model_state_'
        save_path_model_state = os.path.join(args['model_dir'], model_state_prefix+str(epoch+1)+'.pt')
        torch.save(net.state_dict(), save_path_model_state)

        writer.add_scalar('PSNR_epoch_test', psnr_per_epoch, epoch)
        writer.add_scalar('SSIM_epoch_test', ssim_per_epoch, epoch)
        toc = time.time()
        print('This epoch take time {:.2f}'.format(toc-tic))
    writer.close()
    print('Reach the maximal epochs! Finish training')

def main():
    # set parameters
    with open('./configs/sisr_general.json', 'r') as f:
        args = json.load(f)

    # set the available GPUs
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args['gpu_id'])

    # build the model
    net = VIRNetSISR(_C, wf=args['wf'], dep_U=args['dep_U'], dep_S=args['dep_S'],
                                                         ksize=args['kernel_size']).cuda()

    # optimizer
    optimizer = optim.Adam(net.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, args['milestones'], gamma=0.5)

    # resume from specific epoch
    if args['resume']:
        if os.path.isfile(args['resume']):
            print('=> Loading checkpoint {:s}'.format(args['resume']))
            checkpoint = torch.load(args['resume'])
            args['epoch_start'] = checkpoint['epoch']
            args['step'] = checkpoint['step']
            args['step_img'] = checkpoint['step_img']
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
            net.load_state_dict(checkpoint['model_state_dict'])
            args['clip_grad'] = checkpoint['grad_norm']
            print('=> Loaded checkpoint {:s} (epoch {:d})'.format(args['resume'], checkpoint['epoch']))
        else:
            sys.exit('Please provide corrected model path!')
    else:
        args['epoch_start'] = 0
        if os.path.isdir(args['log_dir']):
            shutil.rmtree(args['log_dir'])
        os.makedirs(args['log_dir'])
        if os.path.isdir(args['model_dir']):
            shutil.rmtree(args['model_dir'])
        os.makedirs(args['model_dir'])

    # print the arg pamameters
    for key, value in args.items():
        print('{:<15s}: {:s}'.format(key, str(value)))

    # prepare training data
    train_h5_path = args['DIV2K_HR_train_path']

    # prepare tesing data
    test_im_list_HR = Path(args['DIV2K_HR_valid_path']).glob('*.bmp')
    test_im_list_HR = sorted([str(x) for x in test_im_list_HR])

    # make dataset
    datasets = {'train':GeneralTrainH5(train_h5_path, 8000*args['batch_size'], args['HR_size'],
             scale=args['scale'], kernel_size=args['kernel_size'], sigma_range=args['noise_level']),
                                                  'val':GeneralTest(test_im_list_HR, args['scale'])}

    # get the gauss kernle for inverse gamma prior
    blur_kernel_test = getGaussianKernel2D(args['kernel_size'], args['kernel_size_test'])[np.newaxis, np.newaxis,]
    blur_kernel_test = torch.from_numpy(blur_kernel_test).type(torch.float32).cuda()

    # train model
    print('\nBegin training with GPU: ' + str(args['gpu_id']))
    train_model(net, datasets, optimizer, scheduler, blur_kernel_test, elbo_sisr, args)

if __name__ == '__main__':
    main()
