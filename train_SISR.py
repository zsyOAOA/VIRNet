#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2019-01-10 22:41:49

import argparse
import os
import commentjson as json
import sys
import time
import math
import shutil
from pathlib import Path
from collections import OrderedDict
from loss.ELBO_simple import elbo_sisr
from networks.VIRNet import VIRAttResUNetSR
from datasets.SISRDatasets import GeneralTrainFloder, GeneralTest

from utils import util_net
from utils import util_sisr
from utils import util_image
from utils import util_denoising
from utils import util_opts
from utils import util_common

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
import torch.utils.data as udata
import torch.multiprocessing as mp
import torchvision.utils as vutils
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP

def init_dist(backend='nccl', **kwargs):
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method('spawn')
    rank = int(os.environ["RANK"])
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(backend=backend, **kwargs)

def close_dist():
    dist.destroy_process_group()

def main():
    # set parameters
    with open(opts_parser.config, 'r') as f:
        args = json.load(f)
    util_opts.update_args(args, opts_parser)

    # set the available GPUs
    num_gpus = torch.cuda.device_count()
    args['dist'] = True if num_gpus > 1 else False

    # noise types
    noise_types_list = ['Gaussian', ]
    if util_opts.str2bool(args['add_jpeg']): noise_types_list.append('JPEG')

    # distributed settings
    if args['dist']:
        init_dist()
        rank = dist.get_rank()
    else:
        rank = 0

    # print the arg pamameters
    if rank == 0:
        for key, value in args.items():
            print('{:<20s}: {:s}'.format(key, str(value)))

    # set the seed before initilizing the network
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)

    # build the model
    net = VIRAttResUNetSR(im_chn=args['im_chn'],
                          sigma_chn=args['sigma_chn'],
                          dep_K=args['dep_K'],
                          dep_S=args['dep_S'],
                          n_feat=args['n_feat'],
                          n_resblocks=args['n_resblocks'],
                          noise_cond=util_opts.str2bool(args['noise_cond']),
                          kernel_cond=util_opts.str2bool(args['kernel_cond']),
                          extra_mode=args['extra_mode'],
                          noise_avg= (not util_opts.str2bool(args['add_jpeg']))).cuda()

    if rank == 0:
        print('Number of parameters in SNet: {:.2f}M'.format(util_net.calculate_parameters(net.SNet)/(1000**2)), flush=True)
        print('Number of parameters in KNet: {:.2f}M'.format(util_net.calculate_parameters(net.KNet)/(1000**2)), flush=True)
        print('Number of parameters in RNet: {:.2f}M'.format(util_net.calculate_parameters(net.RNet)/(1000**2)), flush=True)
        print(net)
    if args['dist']:
        net = DDP(net, device_ids=[rank])  # wrap the network

    # optimizer
    optimizer = optim.Adam(net.parameters(), lr=args['lr'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,
                                                     T_max=args['epochs'],
                                                     eta_min=args['lr_min'])
    if rank == 0:
        print('T_max = {:d}, eta_min={:.2e}'.format(args['epochs'], args['lr_min']))
        # util_net.test_scheduler(scheduler, optimizer, args['epochs'])

    # resume from specific epoch
    if rank == 0:
        log_dir = Path(args['save_dir']) / 'logs'
        model_dir = Path(args['save_dir']) / 'models'
    if args['resume']:
        if os.path.isfile(args['resume']):
            checkpoint = torch.load(args['resume'], map_location='cuda:%d'%rank)
            args['epoch_start'] = checkpoint['epoch']
            try:
                net.load_state_dict(checkpoint['model_state_dict'])
            except:
                net.load_state_dict(OrderedDict({'module.'+key:value for key, value in checkpoint['model_state_dict'].items()}))
            for _ in range(args['epoch_start']):
                scheduler.step()
            if rank == 0:
                args['step'] = checkpoint['step']
                args['step_img'] = checkpoint['step_img']
                print('=> Loaded checkpoint {:s} (epoch {:d})'.format(args['resume'], checkpoint['epoch']), flush=True)
        else:
            sys.exit('Please provide corrected model path!')
    else:
        args['epoch_start'] = 0
        if rank == 0:
            util_common.mkdir(log_dir, delete=True)
            util_common.mkdir(model_dir, delete=False)

    # prepare training data
    train_dataset = GeneralTrainFloder(hr_dir=args['train_hr_patchs'],
                                       sf=args['sf'],
                                       length=10000*args['batch_size'],
                                       hr_size=args['hr_size'],
                                       k_size=args['k_size'],
                                       kernel_shift=util_opts.str2bool(args['kernel_shift']),
                                       downsampler=args['downsampler'],
                                       add_jpeg=util_opts.str2bool(args['add_jpeg']),
                                       noise_jpeg=args['noise_jpeg'],
                                       noise_level=args['noise_level'])
    if rank == 0:
        print('Number of Patches in training data set: {:d}'.format(train_dataset.num_images), flush=True)
    if num_gpus > 1:
        shuffle_flag = False
        train_sampler = udata.distributed.DistributedSampler(train_dataset, num_replicas=num_gpus, rank=rank)
    else:
        shuffle_flag = True
        train_sampler = None
    train_dataloader = udata.DataLoader(train_dataset,
                                        batch_size=args['batch_size'] // num_gpus,
                                        shuffle=shuffle_flag,
                                        drop_last=False,
                                        num_workers=args['num_workers'] // num_gpus,
                                        pin_memory=True,
                                        prefetch_factor=args['prefetch_factor'],
                                        sampler=train_sampler)

    # prepare tesing data
    test_datasets = {x:GeneralTest(args['val_hr_path'],
                                   sf=args['sf'],
                                   k_size=args['k_size'],
                                   kernel_shift=util_opts.str2bool(args['kernel_shift']),
                                   downsampler=args['downsampler'],
                                   noise_type=x) for x in noise_types_list}
    test_dataloaders = {x:udata.DataLoader(test_datasets[x],
                                           batch_size=1,
                                           shuffle=False,
                                           pin_memory=True,
                                           num_workers=0,
                                           drop_last=False) for x in noise_types_list}

    if rank == 0:
        num_iter_epoch = {'train': math.ceil(len(train_dataset) / args['batch_size']), 'test': len(test_datasets['Gaussian'])}
        writer = SummaryWriter(str(log_dir))
        step = args['step'] if args['resume'] else 0
        step_img = args['step_img'] if args['resume'] else {phase:0 for phase in ['train',]+noise_types_list}
    chn = args['im_chn']
    alpha0 = 0.5 * torch.tensor([args['var_window']**2], dtype=torch.float32).cuda()
    kappa0 = torch.tensor([args['kappa0']], dtype=torch.float32).cuda()
    param_rnet = [x for name, x in net.named_parameters() if 'rnet' in name.lower()]
    param_snet = [x for name, x in net.named_parameters() if 'snet' in name.lower()]
    param_knet = [x for name, x in net.named_parameters() if 'knet' in name.lower()]
    for epoch in range(args['epoch_start'], args['epochs']):
        train_dataset.reset_seed(epoch)
        if num_gpus > 1:
            train_sampler.set_epoch(epoch)
        if rank == 0:
            loss_per_epoch = {x: 0 for x in ['Loss', 'lh', 'KLR', 'KLS', 'KLK']}
            tic = time.time()
        # train stage
        phase = 'train'
        net.train()
        lr = optimizer.param_groups[0]['lr']
        phase = 'train'
        for ii, data in enumerate(train_dataloader):
            # prepare data
            im_hr, im_lr, im_blur, kinfo_gt, nlevel = [x.cuda() for x in data]
            if util_opts.str2bool(args['add_jpeg']):
                sigma_prior = util_denoising.noise_estimate_fun(im_lr, im_blur, args['var_window'])
            else:
                sigma_prior = nlevel     # N x 1 x 1 x1 for Gaussian noise

            # network optimization
            optimizer.zero_grad()
            mu, kinfo_est, sigma_est = net(im_lr, args['sf'])
            loss, loss_detail = elbo_sisr(mu=mu,
                                          sigma_est=sigma_est,
                                          kinfo_est=kinfo_est,
                                          im_hr=im_hr,
                                          im_lr=im_lr,
                                          sigma_prior=sigma_prior,
                                          alpha0=alpha0,
                                          kinfo_gt=kinfo_gt,
                                          kappa0=kappa0,
                                          r2=args['r2'],
                                          eps2=args['eps2'],
                                          sf=args['sf'],
                                          k_size=args['k_size'],
                                          penalty_K=args['penalty_K'],
                                          downsampler=args['downsampler'],
                                          shift=util_opts.str2bool(args['kernel_shift']))
            loss.backward()
            # clip the gradnorm
            total_norm_R = nn.utils.clip_grad_norm_(param_rnet, args['clip_grad_R'])
            total_norm_S = nn.utils.clip_grad_norm_(param_snet, args['clip_grad_S'])
            total_norm_K = nn.utils.clip_grad_norm_(param_knet, args['clip_grad_K'])
            optimizer.step()

            if rank == 0:
                loss_per_epoch['Loss'] += loss.item() / num_iter_epoch[phase]
                loss_per_epoch['lh'] += loss_detail[0].item() / num_iter_epoch[phase]
                loss_per_epoch['KLR'] += loss_detail[1].item() / num_iter_epoch[phase]
                loss_per_epoch['KLS'] += loss_detail[2].item() / num_iter_epoch[phase]
                loss_per_epoch['KLK'] += loss_detail[3].item() / num_iter_epoch[phase]
                im_hr_est = mu[0].detach() if isinstance(mu, list) else mu.detach()
                if ((ii+1) % args['print_freq'] == 0 or ii==0) and rank == 0:
                    log_str = '[Epoch:{:>3d}/{:<3d}] {:s}:{:0>5d}/{:0>5d}, lh:{:+>5.2f}, KL:{:+>7.2f}/{:+>6.2f}/{:+>6.2f}, ' +\
                                                                                          'Grad:{:.1e}/{:.1e}/{:.1e}, lr={:.1e}'
                    print(log_str.format(epoch+1, args['epochs'], phase, ii+1, num_iter_epoch[phase], loss_detail[0].item(),
                                                            loss_detail[1].item(), loss_detail[2].item(), loss_detail[3].item(),
                                                                                  total_norm_R, total_norm_S, total_norm_K, lr))
                    writer.add_scalar('Train Loss Iter', loss.item(), step)
                    step += 1
                if (ii+1) % (20*args['print_freq']) == 0 and rank == 0:
                    x1 = vutils.make_grid(im_hr_est, normalize=True, scale_each=True)
                    writer.add_image(phase+' Recover Image', x1, step_img[phase])
                    x2 = vutils.make_grid(im_hr, normalize=True, scale_each=True)
                    writer.add_image(phase+' HR Image', x2, step_img[phase])
                    kernel_blur = util_sisr.kinfo2sigma(kinfo_gt, k_size=args['k_size'], sf=args['sf'])
                    x3 = vutils.make_grid(kernel_blur, normalize=True, scale_each=True)
                    writer.add_image(phase+' GT Blur Kernel', x3, step_img[phase])
                    x4 = vutils.make_grid(im_lr, normalize=True, scale_each=True)
                    writer.add_image(phase+' LR Image', x4, step_img[phase])
                    x5 = vutils.make_grid(loss_detail[7].detach(), normalize=True, scale_each=True)
                    writer.add_image(phase+' Est Blur Kernel Resample', x5, step_img[phase])
                    kernel_blur_est = util_sisr.kinfo2sigma(kinfo_est.detach(),
                                                            k_size=args['k_size'],
                                                            sf=args['sf'],
                                                            shift=util_opts.str2bool(args['kernel_shift']))
                    x6 = vutils.make_grid(kernel_blur_est, normalize=True, scale_each=True)
                    writer.add_image(phase+' Est Blur Kernel', x6, step_img[phase])
                    step_img[phase] += 1

        if rank == 0:
            log_str ='{:s}: Loss={:+.2e}, lh={:>4.2f}, KLR={:+>7.2f}, KLS={:+>6.2f}, KLK={:+>5.2f}'
            print(log_str.format(phase, loss_per_epoch['Loss'], loss_per_epoch['lh'], loss_per_epoch['KLR'],
                                                                        loss_per_epoch['KLS'], loss_per_epoch['KLK']))
            writer.add_scalar('Loss_epoch', loss_per_epoch['Loss'], epoch)
            print('-'*105)

        # test stage
        if rank == 0:
            phase = 'test'
            net.eval()
            for noise_type in noise_types_list:
                psnr_per_epoch = ssim_per_epoch = 0
                for ii, data in enumerate(test_dataloaders[noise_type]):
                    im_hr, im_lr, kinfo_gt = [x.cuda() for x in data]
                    with torch.set_grad_enabled(False):
                        mu, kinfo_est, sigma_est = net(im_lr, args['sf'])
                        im_hr_est = mu[0] if isinstance(mu, list) else mu

                    psnr_iter = util_image.batch_PSNR(im_hr_est, im_hr, args['sf']**2, True)
                    ssim_iter = util_image.batch_SSIM(im_hr_est, im_hr, args['sf']**2, True)
                    psnr_per_epoch += psnr_iter
                    ssim_per_epoch += ssim_iter
                    # print statistics every log_interval mini_batches
                    if (ii+1) % 3 == 0:
                        log_str = 'Noise: {:s}, Epoch:{:>3d}/{:<3d}] {:s}:{:0>3d}/{:0>3d}, psnr={:4.2f}, ssim={:5.4f}'
                        print(log_str.format(noise_type, epoch+1, args['epochs'], phase, ii+1, num_iter_epoch[phase],
                                                                                                  psnr_iter, ssim_iter))
                        # tensorboardX summary
                        x1 = vutils.make_grid(im_hr_est, normalize=True, scale_each=True)
                        writer.add_image('Test '+noise_type+' Recover images', x1, step_img[noise_type])
                        x2 = vutils.make_grid(im_hr, normalize=True, scale_each=True)
                        writer.add_image('Test '+noise_type+' HR Image', x2, step_img[noise_type])
                        x3 = vutils.make_grid(im_lr, normalize=True, scale_each=True)
                        writer.add_image('Test '+noise_type+' LR Image', x3, step_img[noise_type])
                        kernel_blur = util_sisr.kinfo2sigma(kinfo_gt,
                                                            k_size=args['k_size'],
                                                            sf=args['sf'],
                                                            shift=util_opts.str2bool(args['kernel_shift']))
                        x4 = vutils.make_grid(kernel_blur, normalize=True, scale_each=True)
                        writer.add_image('Test '+noise_type+' GT Blur Kernel', x4, step_img[noise_type])
                        kernel_blur_est = util_sisr.kinfo2sigma(kinfo_est,
                                                                k_size=args['k_size'],
                                                                sf=args['sf'],
                                                                shift=util_opts.str2bool(args['kernel_shift']))
                        x5 = vutils.make_grid(kernel_blur_est, normalize=True, scale_each=True)
                        writer.add_image('Test '+noise_type+' Est Blur Kernel', x5, step_img[noise_type])
                        step_img[noise_type] += 1

                psnr_per_epoch /= (ii+1)
                ssim_per_epoch /= (ii+1)
                writer.add_scalar(noise_type + ' PSNR Test', psnr_per_epoch, epoch)
                writer.add_scalar(noise_type + ' SSIM Test', ssim_per_epoch, epoch)
                log_str = 'Noise: {:s}, {:s}: PSNR={:4.2f}, SSIM={:5.4f}'
                print(log_str.format(noise_type, phase, psnr_per_epoch, ssim_per_epoch))
                print('-'*60)

        # adjust the learning rate
        scheduler.step()
        # save model
        if rank == 0:
            save_path_model = str(model_dir / ('model_'+str(epoch+1)+'.pth'))
            torch.save({'epoch': epoch+1,
                        'step': step+1,
                        'step_img': {x: step_img[x] for x in ['train',]+noise_types_list},
                        'model_state_dict': net.state_dict()}, save_path_model)
            toc = time.time()
            print('This epoch take time {:.2f}'.format(toc-tic))

    if rank == 0:
        writer.close()
    if num_gpus > 1:
        close_dist()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # DDP parameters
    parser.add_argument('--local_rank', type=int, default=0, help="Passed by launch.py")
    # GPU settings
    parser.add_argument('--config', type=str, default="./configs/sisr_x4.json",
                                                                    help="Path for the config file")
    parser.add_argument('--save_dir', default='./train_save', type=str, metavar='PATH',
                                                                   help="Path to save the log file")
    opts_parser  = parser.parse_args()

    main()
