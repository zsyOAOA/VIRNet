#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2019-01-10 22:41:49

import argparse
import os
import sys
import shutil
import time
import math
from pathlib import Path
from thop import profile
import commentjson as json
from networks.VIRNet import VIRAttResUNet
from datasets.data_tools import MixUp_AUG
from loss.ELBO_simple import elbo_denoising_simple
from datasets.DenoisingDatasets import RealTrain, BenchmarkTest
from gradual_warmup_lr.warmup_scheduler import GradualWarmupScheduler

from utils import util_net
from utils import util_opts
from utils import util_image
from utils import util_common
from utils import util_denoising

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.utils.data as udata
import torchvision.utils as vutils
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP

def dist_setup(port, rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = port

    dist.init_process_group(backend='nccl', init_method='env://', rank=rank, world_size=world_size)

def dist_close():
    dist.destroy_process_group()

def train_model(rank, args):
    if args['num_gpus'] > 1:
        dist_setup(str(args['port']), rank=rank, world_size=args['num_gpus'])

    # set current gpu
    torch.cuda.set_device(rank)

    # set the seed before initilizing the network
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)

    net = VIRAttResUNet(im_chn=args['im_chn'],
                        sigma_chn=args['sigma_chn'],
                        n_feat=args['n_feat'],
                        dep_S=args['dep_S'],
                        n_resblocks=args['n_resblocks'],
                        extra_mode=args['extra_mode'],
                        noise_cond=util_opts.str2bool(args['noise_cond']),
                        noise_avg=False).cuda()

    if rank == 0:
        print('Number of parameters in SNet: {:.2f}M'.format(util_net.calculate_parameters(net.SNet)/(1000**2)), flush=True)
        print('Number of parameters in RNet: {:.2f}M'.format(util_net.calculate_parameters(net.RNet)/(1000**2)), flush=True)
        print(net)
    if args['num_gpus'] > 1:
        net = DDP(net, device_ids=[rank])  # wrap the network

    #define optimizer and scheduler
    optimizer = optim.Adam(net.parameters(), lr=args['lr'])
    T_max = args['epochs'] - args['warmup_epochs']
    lr_min = 1e-6
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=T_max, eta_min=lr_min)
    if rank == 0: print('T_max = {:d}, eta_min={:.2e}'.format(T_max, lr_min))
    scheduler = GradualWarmupScheduler(optimizer, multiplier=1,
                                       total_epoch=args['warmup_epochs'],
                                       after_scheduler=scheduler)

    # resume
    if rank == 0:
        model_dir = Path(args['save_dir']) / 'models'
        log_dir = Path(args['save_dir']) / 'logs'
    if args['resume']:
        if os.path.isfile(args['resume']):
            checkpoint = torch.load(args['resume'], map_location='cuda:%d'%rank)
            args['epoch_start'] = checkpoint['epoch']
            net.load_state_dict(checkpoint['model_state_dict'])
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
            util_common.mkdir(model_dir, delete=True)
            util_common.mkdir(log_dir, delete=True)

    # prepare the training data
    train_dataset = RealTrain(noisy_dir=args['train_pch_dir'],
                              length=10000*args['batch_size'],
                              pch_size=args['patch_size'],
                              renoir=util_opts.str2bool(args['add_renoir']),
                              polyu=util_opts.str2bool(args['add_polyu']))
    if rank == 0:
        print('Number of Patches in training data set: {:d}'.format(train_dataset.num_images), flush=True)
    if args['num_gpus'] > 1:
        shuffle_flag = False
        train_sampler = udata.distributed.DistributedSampler(train_dataset, num_replicas=args['num_gpus'], rank=rank)
    else:
        shuffle_flag = True
        train_sampler = None
    train_dataloader = udata.DataLoader(train_dataset,
                                        batch_size=args['batch_size'] // args['num_gpus'],
                                        shuffle=shuffle_flag,
                                        drop_last=False,
                                        num_workers=args['num_workers'] // args['num_gpus'],
                                        pin_memory=True,
                                        prefetch_factor=args['prefetch_factor'],
                                        sampler=train_sampler)

    # prepare the testing data
    batch_size_test = 4
    test_dataset = BenchmarkTest(args['test_noisy_path'], args['test_gt_path'])
    test_dataloader = udata.DataLoader(test_dataset,
                                       batch_size=batch_size_test,
                                       shuffle=False,
                                       drop_last=False,
                                       num_workers=0,
                                       pin_memory=True)

    if rank == 0:
        num_iter_epoch = {'train': math.ceil(len(train_dataset) / args['batch_size']),
                          'test': math.ceil(len(test_dataset) / batch_size_test)}
        writer = SummaryWriter(str(log_dir))
        step = args['step'] if args['resume'] else 0
        step_img = args['step_img'] if args['resume'] else {phase:0 for phase in ['train', 'test']}
    param_R = [x for name, x in net.named_parameters() if 'rnet' in name.lower()]
    param_S = [x for name, x in net.named_parameters() if 'snet' in name.lower()]
    chn = args['im_chn']
    mixup = MixUp_AUG()
    alpha0 = 0.5 * torch.tensor([args['var_window']**2], dtype=torch.float32).cuda()
    for epoch in range(args['epoch_start'], args['epochs']):
        train_dataset.reset_seed(epoch)
        if args['num_gpus'] > 1:
            train_sampler.set_epoch(epoch)
        if rank == 0:
            loss_per_epoch = {x: 0 for x in ['Loss', 'lh', 'KLG', 'KLIG']}
            tic = time.time()
        # train stage
        phase = 'train'
        net.train()
        lr = optimizer.param_groups[0]['lr']
        for ii, data in enumerate(train_dataloader):
            im_noisy, im_gt = [x.cuda() for x in data]
            # mixup, borrowed from MPRNet, a little performance gain
            im_gt, im_noisy = mixup.aug(im_gt, im_noisy)
            sigma_gt = util_denoising.noise_estimate_fun(im_noisy, im_gt, args['var_window'])
            beta0 = alpha0 * sigma_gt

            # foreward
            optimizer.zero_grad()
            mu, sigma = net(im_noisy)
            loss, g_lh, kl_g, kl_Igam = elbo_denoising_simple(mu, sigma, im_noisy, im_gt,
                                                                    args['eps2'], alpha0, beta0)
            loss.backward()

            # clip the gradient norm of D-Net
            total_norm_R = nn.utils.clip_grad_norm_(param_R, args['clip_grad_R'])
            total_norm_S = nn.utils.clip_grad_norm_(param_S, args['clip_grad_S'])
            optimizer.step()

            if rank == 0:
                loss_per_epoch['Loss'] += loss.item() / num_iter_epoch[phase]
                loss_per_epoch['lh'] += g_lh.item() / num_iter_epoch[phase]
                loss_per_epoch['KLG'] += kl_g.item() / num_iter_epoch[phase]
                loss_per_epoch['KLIG'] += kl_Igam.item() / num_iter_epoch[phase]
                im_denoise = mu[0][:, :chn, ].data if isinstance(mu, list) else mu[:, :chn,].data
                im_denoise.clamp_(0.0, 1.0)
                if (ii+1) % args['print_freq'] == 0 or ii == 0:
                    log_str = '[Epoch:{:>2d}/{:<2d}] {:s}:{:0>5d}/{:0>5d}, lh={:+4.2f}, KLG={:+>7.2f}, ' + \
                                   'KLIG={:+>6.2f}, GNorm_D:{:.1e}/{:.1e}, GNorm_S:{:.1e}/{:.1e}, lr={:.2e}'
                    print(log_str.format(epoch+1, args['epochs'], phase, ii+1, num_iter_epoch[phase],
                                g_lh.item(), kl_g.item(), kl_Igam.item(),  args['clip_grad_R'], total_norm_R,
                                                          args['clip_grad_S'], total_norm_S, lr), flush=True)
                    writer.add_scalar('Train Loss Iter', loss.item(), step)
                    step += 1
                if (ii+1) % (30*args['print_freq']) == 0:
                    x1 = vutils.make_grid(im_denoise, normalize=True, scale_each=True)
                    writer.add_image(phase+' Denoised images', x1, step_img[phase])
                    x2 = vutils.make_grid(im_gt, normalize=True, scale_each=True)
                    writer.add_image(phase+' GroundTruth', x2, step_img[phase])
                    x3 = vutils.make_grid(sigma.detach(), normalize=True, scale_each=True)
                    writer.add_image(phase+' Predict Sigma', x3, step_img[phase])
                    x4 = vutils.make_grid(sigma_gt, normalize=True, scale_each=True)
                    writer.add_image(phase+' Groundtruth Sigma', x4, step_img[phase])
                    x5 = vutils.make_grid(im_noisy, normalize=True, scale_each=True)
                    writer.add_image(phase+' Noisy Image', x5, step_img[phase])
                    step_img[phase] += 1
        if rank == 0:
            log_str ='{:s}: Loss={:+.2e}, lh={:+.2e}, KL_Guass={:+.2e}, KLIG={:+.2e}'
            print(log_str.format(phase, loss_per_epoch['Loss'], loss_per_epoch['lh'], loss_per_epoch['KLG'],
                                                                        loss_per_epoch['KLIG']), flush=True)
            writer.add_scalar('Loss_epoch', loss_per_epoch['Loss'], epoch)
            print('-'*140, flush=True)

        # test stage
        if rank == 0:
            phase = 'test'
            net.eval()
            psnr_per_epoch = ssim_per_epoch = 0
            for ii, data in enumerate(test_dataloader):
                im_noisy, im_gt = [x.cuda() for x in data]
                with torch.set_grad_enabled(False):
                    mu, sigma = net(im_noisy)

                im_denoise = mu[0][:, :chn, ].data if isinstance(mu, list) else mu[:, :chn,].data
                im_denoise.clamp_(0.0, 1.0)
                psnr_iter = util_image.batch_PSNR(im_denoise, im_gt)
                ssim_iter = util_image.batch_SSIM(im_denoise, im_gt)
                psnr_per_epoch += psnr_iter
                ssim_per_epoch += ssim_iter
                # print statistics every log_interval mini_batches
                if (ii+1) % 40 == 0:
                    log_str = '[Epoch:{:>3d}/{:<3d}] {:s}:{:0>4d}/{:0>4d}, psnr={:4.2f}, ssim={:5.4f}'
                    print(log_str.format(epoch+1, args['epochs'], phase, ii+1, num_iter_epoch[phase],
                                                                        psnr_iter, ssim_iter), flush=True)
                    # tensorboardX summary
                    x1 = vutils.make_grid(im_denoise, normalize=True, scale_each=True)
                    writer.add_image(phase+' Denoised images', x1, step_img[phase])
                    x2 = vutils.make_grid(im_gt, normalize=True, scale_each=True)
                    writer.add_image(phase+' GroundTruth', x2, step_img[phase])
                    x3 = vutils.make_grid(sigma, normalize=True, scale_each=True)
                    writer.add_image(phase+' Predict Sigma', x3, step_img[phase])
                    x5 = vutils.make_grid(im_noisy, normalize=True, scale_each=True)
                    writer.add_image(phase+' Noisy Image', x5, step_img[phase])
                    step_img[phase] += 1

            psnr_per_epoch /= (ii+1)
            ssim_per_epoch /= (ii+1)
            log_str ='{:s}: PSNR={:4.2f}, SSIM={:5.4f}'
            print(log_str.format(phase, psnr_per_epoch, ssim_per_epoch), flush=True)
            print('-'*90, flush=True)

        # adjust the learning rate
        scheduler.step()
        # save model
        if rank == 0:
            save_path_model = str(model_dir / ('model_'+str(epoch+1)+'.pth'))
            torch.save({
                'epoch': epoch+1,
                'step': step+1,
                'step_img': {x:step_img[x] for x in ['train', 'test']},
                'model_state_dict': net.state_dict()}, save_path_model)

            writer.add_scalar('PSNR_epoch_test', psnr_per_epoch, epoch)
            writer.add_scalar('SSIM_epoch_test', ssim_per_epoch, epoch)
            toc = time.time()
            print('This epoch take time {:.2f}'.format(toc-tic), flush=True)

    if rank == 0:
        writer.close()
    if args['num_gpus'] > 1:
        dist_close()

def main():
    # set parameters
    with open(opts_parser.config, 'r') as f:
        args = json.load(f)
    util_opts.update_args(args, opts_parser)

    # set the available GPUs
    num_gpus = len(args['gpu_id'])
    gpus = ','.join([args['gpu_id'][i] for i in range(num_gpus)])
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = gpus
    args['num_gpus'] = num_gpus

    # print the arg pamameters
    for key, value in args.items():
        print('{:<20}: {:s}'.format(key, str(value)))

    if num_gpus > 1:
        mp.spawn(train_model, nprocs=num_gpus, args=(args,))
    else:
        train_model(rank=0, args=args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=str, default=0, help="GPU ID, which allow multiple GPUs")
    parser.add_argument('--save_dir', default='./models', type=str, metavar='PATH',
                                                             help="Path to save the model and log file")
    parser.add_argument('--config', type=str, default="./configs/denoising_real.json",
                                                                    help="Path for the config file")
    opts_parser = parser.parse_args()

    main()

