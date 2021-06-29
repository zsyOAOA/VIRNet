#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2019-01-10 22:41:49

import os
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
from torch.utils.tensorboard import SummaryWriter
from utils import *
import commentjson as json
from math import ceil
from loss.ELBO import elbo_denoising as loss_fn
from networks.VIRNet import VIRNetU
from datasets.DenoisingDatasets import SimulateTrain, SimulateTest
from pathlib import Path

# filter warnings
warnings.simplefilter('ignore', Warning, lineno=0)

_C = 3
_modes = ['train', 'test_bsd68_case1', 'test_bsd68_case2', 'test_bsd68_case3']

def train_model(net, datasets, optimizer, lr_scheduler, criterion, kernel, args):
    clip_grad_D = args['clip_grad_D']
    clip_grad_S = args['clip_grad_S']
    batch_size = {_modes[0]: args['batch_size'], _modes[1]: 1, _modes[2]: 1, _modes[3]: 1}
    data_loader = {phase: torch.utils.data.DataLoader(datasets[phase], batch_size=batch_size[phase],
                shuffle=True, num_workers=args['num_workers'], pin_memory=True) for phase in _modes}
    num_data = {phase: len(datasets[phase]) for phase in datasets.keys()}
    num_iter_epoch = {phase: ceil(num_data[phase] / batch_size[phase]) for phase in _modes}
    writer = SummaryWriter(args['log_dir'])
    if args['resume']:
        step = args['step']
        step_img = args['step_img']
    else:
        step = 0
        step_img = {x: 0 for x in _modes}
    param_D = [x for name, x in net.named_parameters() if 'dnet' in name.lower()]
    param_S = [x for name, x in net.named_parameters() if 'snet' in name.lower()]
    for epoch in range(args['epoch_start'], args['epochs']):
        loss_per_epoch = {x: 0 for x in ['Loss', 'lh', 'KLG', 'KLIG']}
        mse_per_epoch = {x: 0 for x in _modes}
        grad_norm_D = grad_norm_S = 0
        tic = time.time()
        # train stage
        net.train()
        lr = optimizer.param_groups[0]['lr']
        phase = 'train'
        for ii, data in enumerate(data_loader[phase]):
            im_noisy, im_gt, sigmaMapGt = [x.cuda() for x in data]
            with torch.set_grad_enabled(False):
                err2 = (im_noisy - im_gt) ** 2
                sigmaMapEst = gaussblur(err2, kernel, p=2*args['radius']+1, chn=_C)
                sigmaMapEst.clamp_(min=1e-10)
            optimizer.zero_grad()
            phi_Z, phi_sigma = net(im_noisy, 'train')
            loss, g_lh, kl_g, kl_Igam = criterion(phi_Z, phi_sigma, im_noisy, im_gt,
                                                  sigmaMapEst, args['eps2'], radius=args['radius'])
            loss.backward()
            # clip the gradnorm
            total_norm_D = nn.utils.clip_grad_norm_(param_D, clip_grad_D)
            total_norm_S = nn.utils.clip_grad_norm_(param_S, clip_grad_S)
            grad_norm_D = (grad_norm_D*(ii/(ii+1)) + total_norm_D/(ii+1))
            grad_norm_S = (grad_norm_S*(ii/(ii+1)) + total_norm_S/(ii+1))
            optimizer.step()

            loss_per_epoch['Loss'] += loss.item() / num_iter_epoch[phase]
            loss_per_epoch['lh'] += g_lh.item() / num_iter_epoch[phase]
            loss_per_epoch['KLG'] += kl_g.item() / num_iter_epoch[phase]
            loss_per_epoch['KLIG'] += kl_Igam.item() / num_iter_epoch[phase]
            im_denoise = phi_Z[:, :_C, ].detach().data
            mse = F.mse_loss(im_denoise, im_gt)
            im_denoise.clamp_(0.0, 1.0)
            mse_per_epoch[phase] += mse
            if (ii+1) % args['print_freq'] == 0:
                log_str = '[Epoch:{:>2d}/{:<2d}] {:s}:{:0>4d}/{:0>4d}, lh={:+4.2f}, ' + \
                        'KLG={:+>7.2f}, KLIG={:+>6.2f}, mse={:.2e}, GNorm_D:{:.1e}/{:.1e}, ' + \
                                                                  'GNorm_S:{:.1e}/{:.1e}, lr={:.1e}'
                print(log_str.format(epoch+1, args['epochs'], phase, ii+1, num_iter_epoch[phase],
                                         g_lh.item(), kl_g.item(), kl_Igam.item(), mse, clip_grad_D,
                                                       total_norm_D, clip_grad_S, total_norm_S, lr))
                writer.add_scalar('Train Loss Iter', loss.item(), step)
                writer.add_scalar('Train MSE Iter', mse, step)
                step += 1
            if (ii+1) % (20*args['print_freq']) == 0:
                alpha = torch.exp(phi_sigma[:, :_C, ])
                beta = torch.exp(phi_sigma[:, _C:, ])
                sigmaMap_pred = beta / (alpha+1)
                x1 = vutils.make_grid(im_denoise, normalize=True, scale_each=True)
                writer.add_image(phase+' Denoised images', x1, step_img[phase])
                x2 = vutils.make_grid(im_gt, normalize=True, scale_each=True)
                writer.add_image(phase+' GroundTruth', x2, step_img[phase])
                x3 = vutils.make_grid(sigmaMap_pred, normalize=True, scale_each=True)
                writer.add_image(phase+' Predict Sigma', x3, step_img[phase])
                x4 = vutils.make_grid(sigmaMapGt, normalize=True, scale_each=True)
                writer.add_image(phase+' Groundtruth Sigma', x4, step_img[phase])
                x5 = vutils.make_grid(im_noisy, normalize=True, scale_each=True)
                writer.add_image(phase+' Noisy Image', x5, step_img[phase])
                step_img[phase] += 1

        mse_per_epoch[phase] /= (ii+1)
        log_str ='{:s}: Loss={:+.2e}, lh={:+.2e}, KL_Guass={:+.2e}, KLIG={:+.2e}, mse={:.3e}, ' + \
                                                      'GNorm_D={:.1e}/{:.1e}, GNorm_S={:.1e}/{:.1e}'
        print(log_str.format(phase, loss_per_epoch['Loss'], loss_per_epoch['lh'],
                                loss_per_epoch['KLG'], loss_per_epoch['KLIG'], mse_per_epoch[phase],
                                                clip_grad_D, grad_norm_D, clip_grad_S, grad_norm_S))
        writer.add_scalar('Loss_epoch', loss_per_epoch['Loss'], epoch)
        writer.add_scalar('Mean Grad Norm_D epoch', grad_norm_D, epoch)
        writer.add_scalar('Mean Grad Norm_S epoch', grad_norm_S, epoch)
        clip_grad_D = min(clip_grad_D, grad_norm_D)
        clip_grad_S = min(clip_grad_S, grad_norm_S)
        print('-'*150)

        # test stage
        net.eval()
        psnr_per_epoch = {x: 0 for x in _modes[1:]}
        ssim_per_epoch = {x: 0 for x in _modes[1:]}
        for phase in _modes[1:]:
            for ii, data in enumerate(data_loader[phase]):
                im_noisy, im_gt = [x.cuda() for x in data]
                with torch.set_grad_enabled(False):
                    phi_Z, phi_sigma = net(im_noisy, 'train')

                im_denoise = torch.clamp(phi_Z[:, :_C, ].detach().data, 0.0, 1.0)
                mse = F.mse_loss(im_denoise, im_gt)
                mse_per_epoch[phase] += mse
                psnr_iter = batch_PSNR(im_denoise, im_gt)
                ssim_iter = batch_SSIM(im_denoise, im_gt)
                psnr_per_epoch[phase] += psnr_iter
                ssim_per_epoch[phase] += ssim_iter
                # print statistics every log_interval mini_batches
                if (ii+1) % 10 == 0:
                    log_str = '[Epoch:{:>3d}/{:<3d}] {:s}:{:0>5d}/{:0>5d}, mse={:.2e}, ' + \
                        'psnr={:4.2f}, ssim={:5.4f}'
                    print(log_str.format(epoch+1, args['epochs'], phase, ii+1, num_iter_epoch[phase],
                                                                         mse, psnr_iter, ssim_iter))
                # tensorboardX summary
                    alpha = torch.exp(phi_sigma[:, :_C, ])
                    beta = torch.exp(phi_sigma[:, _C:, ])
                    sigmaMap_pred = beta / (alpha+1)
                    x1 = vutils.make_grid(im_denoise, normalize=True, scale_each=True)
                    writer.add_image(phase+' Denoised images', x1, step_img[phase])
                    x2 = vutils.make_grid(im_gt, normalize=True, scale_each=True)
                    writer.add_image(phase+' GroundTruth', x2, step_img[phase])
                    x3 = vutils.make_grid(sigmaMap_pred, normalize=True, scale_each=True)
                    writer.add_image(phase+' Predict Sigma', x3, step_img[phase])
                    x4 = vutils.make_grid(im_noisy, normalize=True, scale_each=True)
                    writer.add_image(phase+' Noise Image', x4, step_img[phase])
                    step_img[phase] += 1

            psnr_per_epoch[phase] /= (ii+1)
            ssim_per_epoch[phase] /= (ii+1)
            mse_per_epoch[phase] /= (ii+1)
            log_str = '{:s}: mse={:.3e}, PSNR={:4.2f}, SSIM={:5.4f}'
            print(log_str.format(phase, mse_per_epoch[phase], psnr_per_epoch[phase],
                                 ssim_per_epoch[phase]))
            print('-'*90)

        # adjust the learning rate
        lr_scheduler.step()
        # save model
        model_prefix = 'model_'
        save_path_model = os.path.join(args['model_dir'], model_prefix+str(epoch+1))
        torch.save({
            'epoch': epoch+1,
            'step': step+1,
            'step_img': {x: step_img[x] for x in _modes},
            'grad_norm_D': clip_grad_D,
            'grad_norm_S': clip_grad_S,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'lr_scheduler_state_dict': lr_scheduler.state_dict()
        }, save_path_model)
        model_state_prefix = 'model_state_'
        save_path_model_state = os.path.join(args['model_dir'], model_state_prefix+str(epoch+1)+'.pt')
        torch.save(net.state_dict(), save_path_model_state)

        writer.add_scalars('MSE_epoch', mse_per_epoch, epoch)
        writer.add_scalars('PSNR_epoch_test', psnr_per_epoch, epoch)
        writer.add_scalars('SSIM_epoch_test', ssim_per_epoch, epoch)
        toc = time.time()
        print('This epoch take time {:.2f}'.format(toc-tic))
    writer.close()
    print('Reach the maximal epochs! Finish training')

def main():
    # set parameters
    with open('./configs/denoising_simulation_niid.json', 'r') as f:
        args = json.load(f)

    # set the available GPUs
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args['gpu_id'])

    # build the model
    net = VIRNetU(_C, wf=args['wf'], dep_U=args['dep_U'], dep_S=args['dep_S']).cuda()

    # optimizer
    optimizer = optim.Adam(net.parameters(), lr=args['lr'])
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, args['milestones'], gamma=0.5)

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
            args['clip_grad_D'] = checkpoint['grad_norm_D']
            args['clip_grad_S'] = checkpoint['grad_norm_S']
            print('=> Loaded checkpoint {:s} (epoch {:d})'.format(args['resume'], checkpoint['epoch']))
        else:
            sys.exit('Please provide corrected model path!')
    else:
        args['epoch_start '] = 0
        if os.path.isdir(args['log_dir']):
            shutil.rmtree(args['log_dir'])
        os.makedirs(args['log_dir'])
        if os.path.isdir(args['model_dir']):
            shutil.rmtree(args['model_dir'])
        os.makedirs(args['model_dir'])

    # print the arg pamameters
    for key, value in args.items():
        print('{:<15s}: {:s}'.format(key, str(value)))

    # prepare traing data
    train_im_list = list(Path(args['CBSD_path']).glob('*.jpg'))
    train_im_list += list(Path(args['WED_path']).glob('*.bmp'))
    train_im_list += list(Path(args['CImageNet']).glob('*.png'))
    train_im_list = sorted([str(x) for x in train_im_list])

    # prepare tesing data
    test_case1_h5 = Path('test_data').joinpath('noise_niid_chn3', 'CBSD68_case1.hdf5')
    test_case2_h5 = Path('test_data').joinpath('noise_niid_chn3', 'CBSD68_case2.hdf5')
    test_case3_h5 = Path('test_data').joinpath('noise_niid_chn3', 'CBSD68_case3.hdf5')
    test_im_list = (Path('test_data') / 'CBSD68').glob('*.png')
    test_im_list = sorted([str(x) for x in test_im_list])

    # make datasets
    datasets = {'train':SimulateTrain(train_im_list, 8000*args['batch_size'], args['patch_size'],
                                  mode='niid'), _modes[1]:SimulateTest(test_im_list, test_case1_h5),
                                               _modes[2]: SimulateTest(test_im_list, test_case2_h5),
                                               _modes[3]: SimulateTest(test_im_list, test_case3_h5)}

    # get the gauss kernle for inverse gamma prior
    kernel = inverse_gamma_kernel(args['radius'], _C).cuda()

    # train model
    print('\nBegin training with GPU: ' + str(args['gpu_id']))
    train_model(net, datasets, optimizer, scheduler, loss_fn, kernel, args)

if __name__ == '__main__':
    main()
