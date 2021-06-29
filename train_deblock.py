#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2019-01-10 22:41:49

import os
import sys
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils
from utils import *
from math import ceil
from loss.ELBO import elbo_denoising as loss_fn
from networks.VIRNet import VIRNetBlock
from datasets.DeblockingDatasets import DeblockDatasetTrain, DeblockDatasetTest
from pathlib import Path
import commentjson as json
import warnings
import time
import random
import numpy as np
import shutil

# filter warnings
warnings.simplefilter('ignore', Warning, lineno=0)

_C = 3
_modes = ['train', 'test']

def train_model(net, datasets, optimizer, lr_scheduler, kernel, criterion, args):
    clip_grad_R = args['clip_grad_R']
    clip_grad_S = args['clip_grad_S']
    batch_size = {'train': args['batch_size'], 'test': 1}
    data_loader = {'train': torch.utils.data.DataLoader(datasets['train'],
       batch_size=batch_size['train'], shuffle=True, num_workers=args['num_workers'], pin_memory=True)}
    data_loader['test'] = {str(x):torch.utils.data.DataLoader(datasets['test'][str(x)],
                   batch_size['test'], shuffle=True, num_workers=args['num_workers'], pin_memory=True)
                                                                          for x in [10, 20, 30, 40]}
    num_data = {'train': len(datasets['train'])}
    num_data['test'] = len(datasets['test'][str(10)])
    num_iter_epoch = {phase: ceil(num_data[phase] / batch_size[phase]) for phase in _modes}
    writer = SummaryWriter(args['log_dir'])
    if args['resume']:
        step = args['step']
        step_img = args['step_img']
    else:
        step = 0
        step_img = {x: 0 for x in _modes}
    param_R = [x for name, x in net.named_parameters() if 'rnet' in name.lower()]
    param_S = [x for name, x in net.named_parameters() if 'snet' in name.lower()]
    for epoch in range(args['epoch_start'], args['epochs']):
        loss_per_epoch = {x: 0 for x in ['Loss', 'lh', 'KLG', 'KLIG']}
        mse_per_epoch = {x: 0 for x in ['train', '10', '20', '30', '40']}
        grad_norm_R = grad_norm_S = 0
        tic = time.time()
        # train stage
        net.train()
        lr = optimizer.param_groups[0]['lr']
        phase = 'train'
        for ii, data in enumerate(data_loader[phase]):
            im_jpg, im_gt = [x.cuda() for x in data]
            with torch.set_grad_enabled(False):
                err2 = (im_jpg - im_gt) ** 2
                sigmaMap = gaussblur(err2, kernel, p=2*args['radius']+1, chn=_C)
                sigmaMap.clamp_(min=1e-10)
            optimizer.zero_grad()
            phi_Z, phi_sigma = net(im_jpg, 'train')
            loss, g_lh, kl_g, kl_Igam = criterion(phi_Z, phi_sigma, im_jpg, im_gt,
                                                      sigmaMap, args['eps2'], radius=args['radius'])
            loss.backward()
            # clip the gradnorm
            total_norm_R = nn.utils.clip_grad_norm_(param_R, clip_grad_R)
            total_norm_S = nn.utils.clip_grad_norm_(param_S, clip_grad_S)
            grad_norm_R = (grad_norm_R*(ii/(ii+1)) + total_norm_R/(ii+1))
            grad_norm_S = (grad_norm_S*(ii/(ii+1)) + total_norm_S/(ii+1))
            optimizer.step()

            loss_per_epoch['Loss'] += loss.item() / num_iter_epoch[phase]
            loss_per_epoch['lh'] += g_lh.item() / num_iter_epoch[phase]
            loss_per_epoch['KLG'] += kl_g.item() / num_iter_epoch[phase]
            loss_per_epoch['KLIG'] += kl_Igam.item() / num_iter_epoch[phase]
            im_deblock = phi_Z[:, :_C, ].detach().data
            mse = F.mse_loss(im_deblock, im_gt)
            mse_per_epoch[phase] += mse
            if (ii+1) % args['print_freq'] == 0:
                log_str = '[Epoch:{:>2d}/{:<2d}] {:s}:{:0>4d}/{:0>4d}, lh={:+4.2f}, ' + \
                        'KLG={:+>7.2f}, KLIG={:+>6.2f}, mse={:.2e}, GNorm_R:{:.1e}/{:.1e}, ' + \
                                                                  'GNorm_S:{:.1e}/{:.1e}, lr={:.1e}'
                print(log_str.format(epoch+1, args['epochs'], phase, ii+1, num_iter_epoch[phase],
                                         g_lh.item(), kl_g.item(), kl_Igam.item(), mse, clip_grad_R,
                                                       total_norm_R, clip_grad_S, total_norm_S, lr))
                writer.add_scalar('Train Loss Iter', loss.item(), step)
                writer.add_scalar('Train MSE Iter', mse, step)
                step += 1
            if (ii+1) % (20*args['print_freq']) == 0:
                alpha = torch.exp(phi_sigma[:, :_C, ])
                beta = torch.exp(phi_sigma[:, _C:, ])
                sigmaMap_pred = beta / (alpha-1)
                x1 = vutils.make_grid(im_deblock, normalize=True, scale_each=True)
                writer.add_image(phase+' Deblocked images', x1, step_img[phase])
                x2 = vutils.make_grid(im_gt, normalize=True, scale_each=True)
                writer.add_image(phase+' GroundTruth', x2, step_img[phase])
                x5 = vutils.make_grid(im_jpg, normalize=True, scale_each=True)
                writer.add_image(phase+' JPEG Compress Image', x5, step_img[phase])
                x3 = vutils.make_grid(sigmaMap_pred, normalize=True, scale_each=True)
                writer.add_image(phase+' Predict Sigma', x3, step_img[phase])
                x4 = vutils.make_grid(sigmaMap, normalize=True, scale_each=True)
                writer.add_image(phase+' Prior Sigma', x4, step_img[phase])
                step_img[phase] += 1

        mse_per_epoch[phase] /= (ii+1)
        log_str ='{:s}: Loss={:+.2e}, lh={:+.2e}, KL_Guass={:+.2e}, KLIG={:+.2e}, mse={:.3e}, ' + \
                                                      'GNorm_R={:.1e}/{:.1e}, GNorm_S={:.1e}/{:.1e}'
        print(log_str.format(phase, loss_per_epoch['Loss'], loss_per_epoch['lh'],
                                loss_per_epoch['KLG'], loss_per_epoch['KLIG'], mse_per_epoch[phase],
                                                clip_grad_R, grad_norm_R, clip_grad_S, grad_norm_S))
        writer.add_scalar('Loss_epoch', loss_per_epoch['Loss'], epoch)
        clip_grad_R = min(clip_grad_R, grad_norm_R)
        clip_grad_S = min(clip_grad_S, grad_norm_S)
        print('-'*135)

        # test stage
        phase = 'test'
        net.eval()
        psnr_per_epoch = {str(x): 0 for x in [10, 20, 30, 40]}
        ssim_per_epoch = {str(x): 0 for x in [10, 20, 30, 40]}
        for jpeg_quality in [10, 20, 30, 40]:
            print('Begin test on JPEG Factor: {:d}'.format(jpeg_quality))
            for ii, data in enumerate(data_loader[phase][str(jpeg_quality)]):
                im_jpg, im_gt = [x.cuda() for x in data]
                with torch.set_grad_enabled(False):
                    padunet = PadUNet(im_jpg, args['dep_U'])
                    im_jpg_pad = padunet.pad()
                    phi_Z, phi_sigma = net(im_jpg_pad, 'train')
                    phi_Z = padunet.pad_inverse(phi_Z)
                    phi_sigma = padunet.pad_inverse(phi_sigma)

                im_deblock = phi_Z[:, :_C, ].detach().data
                mse = F.mse_loss(im_deblock, im_gt)
                mse_per_epoch[str(jpeg_quality)] += mse
                psnr_iter = batch_PSNR(im_deblock, im_gt, border=0, ycbcr=True)
                ssim_iter = batch_SSIM(im_deblock, im_gt, border=0, ycbcr=True)
                psnr_per_epoch[str(jpeg_quality)] += psnr_iter
                ssim_per_epoch[str(jpeg_quality)] += ssim_iter
                # print statistics every log_interval mini_batches
                if (ii+1) % 10 == 0:
                    log_str = '[Epoch:{:>2d}/{:<2d}] {:s}:{:0>5d}/{:0>5d}, mse={:.2e}, ' + \
                        'psnr={:4.2f}, ssim={:5.4f}'
                    print(log_str.format(epoch+1, args['epochs'], phase, ii+1, num_iter_epoch[phase],
                                                                         mse, psnr_iter, ssim_iter))
                # tensorboardX summary
                    alpha = torch.exp(phi_sigma[:, :_C, ])
                    beta = torch.exp(phi_sigma[:, _C:, ])
                    sigmaMap_pred = beta / (alpha-1)
                    x1 = vutils.make_grid(im_deblock, normalize=True, scale_each=True)
                    writer.add_image(phase + 'JPEG ' + str(jpeg_quality) + ' Deblocked images', x1,
                                                                                    step_img[phase])
                    x2 = vutils.make_grid(im_gt, normalize=True, scale_each=True)
                    writer.add_image(phase + 'JPEG ' + str(jpeg_quality) + ' GroundTruth', x2,
                                                                                    step_img[phase])
                    x4 = vutils.make_grid(im_jpg, normalize=True, scale_each=True)
                    writer.add_image(phase + 'JPEG ' + str(jpeg_quality) + 'Compress Image', x4,
                                                                                    step_img[phase])
                    x3 = vutils.make_grid(sigmaMap_pred, normalize=True, scale_each=True)
                    writer.add_image(phase + 'JPEG ' + str(jpeg_quality) + ' Predict Sigma', x3,
                                                                                    step_img[phase])
                    step_img[phase] += 1

            psnr_per_epoch[str(jpeg_quality)] /= (ii+1)
            ssim_per_epoch[str(jpeg_quality)] /= (ii+1)
            mse_per_epoch[str(jpeg_quality)] /= (ii+1)
            log_str = 'mse={:.3e}, PSNR={:4.2f}, SSIM={:5.4f}'
            print(log_str.format(mse_per_epoch[str(jpeg_quality)], psnr_per_epoch[str(jpeg_quality)],
                                                                 ssim_per_epoch[str(jpeg_quality)]))
            print('-'*80)

        # adjust the learning rate
        lr_scheduler.step()
        # save model
        model_prefix = 'model_'
        save_path_model = os.path.join(args['model_dir'], model_prefix+str(epoch+1))
        torch.save({
            'epoch': epoch+1,
            'step': step+1,
            'step_img': {x: step_img[x] for x in _modes},
            'grad_norm_R': clip_grad_R,
            'grad_norm_S': clip_grad_S,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'lr_scheduler_state_dict': lr_scheduler.state_dict()
        }, save_path_model)
        model_prefix = 'model_state_'
        save_path_model_state = os.path.join(args['model_dir'], model_prefix+str(epoch+1)+'.pt')
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
    with open('./configs/deblocking.json', 'r') as f:
        args = json.load(f)

    # seting the available GPUs
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args['gpu_id'])

    # move the model to GPU
    net = VIRNetBlock(_C, wf=args['wf'], dep_U=args['dep_U'], dep_S=args['dep_S']).cuda()

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
            args['clip_grad_R'] = checkpoint['grad_norm_R']
            args['clip_grad_S '] = checkpoint['grad_norm_S']
            print('=> Loaded checkpoint {:s} (epoch {:d})'.format(args['resume'], checkpoint['epoch']))
        else:
            sys.exit('Please provide corrected model path!')
    else:
        args['epoch_start'] = 0
        if os.path.isdir(args['log_dir']):
            print('Delete the log dir:{:s}'.format(args['log_dir']))
            shutil.rmtree(args['log_dir'])
        os.makedirs(args['log_dir'])
        if os.path.isdir(args['model_dir']):
            print('Delete the model dir:{:s}'.format(args['model_dir']))
            shutil.rmtree(args['model_dir'])
        os.makedirs(args['model_dir'])

    # print the arg pamameters
    for key, value in args.items():
        print('{:<15s}: {:s}'.format(key, str(value)))

    # make training data
    train_tmp_dir = Path(args['train_path']).parent / ('train_tmp_eps'+str(args['eps2'])+'_r'+str(args['radius']))
    if train_tmp_dir.exists():
        shutil.rmtree(str(train_tmp_dir))
        train_tmp_dir.mkdir()
    else:
        train_tmp_dir.mkdir()
    train_list = Path(args['train_path']).glob('*.png')
    train_list = sorted([str(x) for x in train_list])

    # make testing data
    test_tmp_dir = Path(args['val_path']).parent / ('test_tmp_eps'+str(args['eps2'])+'_r'+str(args['radius']))
    if test_tmp_dir.exists():
        shutil.rmtree(str(test_tmp_dir))
        test_tmp_dir.mkdir()
    else:
        test_tmp_dir.mkdir()
    test_list = Path(args['val_path']).glob('*.bmp')
    test_list = sorted([str(x) for x in test_list])
    datasets = {'train':DeblockDatasetTrain(train_list, train_tmp_dir, 5000*args['batch_size'],
                                                                                args['patch_size'])}
    datasets['test'] = {str(x):DeblockDatasetTest(test_list, test_tmp_dir, x)
                                                                          for x in [10, 20, 30, 40]}

    # get the gauss kernle for inverse gamma prior
    kernel = inverse_gamma_kernel(args['radius'], _C).cuda()

    # train model
    print('\nBegin training with GPU: ' + str(args['gpu_id']))
    train_model(net, datasets, optimizer, scheduler, kernel, loss_fn, args)
    shutil.rmtree(str(train_tmp_dir))
    shutil.rmtree(str(test_tmp_dir))

if __name__ == '__main__':
    main()
