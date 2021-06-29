#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2019-03-20 19:48:14

import torch
from math import pi, log
from .loggamma_op import LogGamma
from .imresize_bicubic import imresize
import torch.nn.functional as F

log_gamma = LogGamma.apply

# clip bound
log_max = log(1e4)
log_min = log(1e-8)

def elbo_denoising(out_denoise, out_sigma, im_noisy, im_gt, sigmaMap, eps2, radius=3):
    '''
    Input:
        eps2: variance of the Gaussian prior of Z
        radius: radius for guided filter in the Inverse Gamma prior
    '''
    C = im_gt.shape[1]
    p = 2*radius+1
    p2 = p**2
    alpha0 = 0.5 * torch.tensor([p2-2]).type(sigmaMap.dtype).to(device=sigmaMap.device)
    beta0 = 0.5 * p2 * sigmaMap

    # parameters predicted of Gaussain distribution
    mu = out_denoise[:, :C,]
    out_denoise[:, C:,].clamp_(min=log_min, max=log_max)
    m2 = torch.exp(out_denoise[:, C:,])   # variance

    # parameters predicted of Inverse Gamma distribution
    out_sigma.clamp_(min=log_min, max=log_max)
    log_alpha = out_sigma[:, :C,]
    alpha = torch.exp(log_alpha)
    log_beta = out_sigma[:, C:,]
    alpha_div_beta = torch.exp(log_alpha - log_beta)

    # KL divergence for Gauss distribution
    m2_div_eps = torch.div(m2, eps2)
    kl_gauss = 0.5 * torch.mean((mu-im_gt)**2/eps2 + (m2_div_eps - 1 - torch.log(m2_div_eps)))

    # KL divergence for Inv-Gamma distribution
    kl_Igamma = torch.mean((alpha-alpha0)*torch.digamma(alpha) + (log_gamma(alpha0) - log_gamma(alpha))
                           + alpha0*(log_beta - torch.log(beta0)) + beta0 * alpha_div_beta - alpha)

    # likelihood of im_gt
    lh = 0.5 * log(2*pi) + 0.5 * torch.mean((log_beta - torch.digamma(alpha)) + \
                                                              ((im_noisy-mu)**2+m2) * alpha_div_beta)

    loss = lh + kl_gauss + kl_Igamma

    return loss, lh, kl_gauss, kl_Igamma

def elbo_sisr(phi_Z, out_sigma, im_LR, im_HR, blur_kernel, varmap_est, scale, downsampler, eps2, radius):
    '''
    Input:
        eps2: variance of the Gaussian prior of Z
    '''
    C = im_LR.shape[1]
    p = 2*radius+1
    p2 = p**2
    alpha0 = 0.5 * torch.tensor([p2]).type(varmap_est.dtype).to(device=varmap_est.device) - 1
    beta0 = 0.5 * p2 * varmap_est

    # parameters predicted of Gaussain distribution
    mu = phi_Z[:, :C,]
    phi_Z[:, C:,].clamp_(max=log(100), min=log(1e-4))
    m = torch.exp(phi_Z[:, C:,])   # std

    # parameters predicted of inverse gamma distribution
    cc = int(out_sigma.shape[1]/2)
    out_sigma.clamp_(min=log(1e-10), max=log_max)
    log_alpha = out_sigma[:, :cc,]
    alpha = torch.exp(log_alpha)
    log_beta = out_sigma[:, cc:,]
    alpha_div_beta = torch.exp(log_alpha - log_beta)

    # KL divergence for Gauss distribution
    m2_div_eps = torch.div(m**2, eps2)
    kl_gauss = 0.5 * torch.mean((im_HR-mu)**2/eps2 + (m2_div_eps - 1 - torch.log(m2_div_eps)))

    # KL divergence for Inv-Gamma distribution
    kl_Igamma = torch.mean((alpha-alpha0)*torch.digamma(alpha) + (log_gamma(alpha0) - log_gamma(alpha))
                + alpha0*(log_beta - torch.log(beta0)) + beta0 * alpha_div_beta - alpha)

    # reparameter z
    Z = mu + torch.randn_like(m) * m

    #Degradation process
    Z_LR, _ = degrademodel(Z, blur_kernel, scale, padding=True, noise_level=None, downsampler=downsampler)

    # likelihood of im_gt
    lh = 0.5 * log(2*pi) + 0.5 * torch.mean((log_beta - torch.digamma(alpha)) + \
                                                              ((im_LR-Z_LR)**2) * alpha_div_beta)

    loss = lh + kl_gauss + kl_Igamma

    return loss, lh, kl_gauss, kl_Igamma

def degrademodel(im_HR, blur_kernel, scale, padding=True, noise_level=None, downsampler='bicubic'):
    B = im_HR.shape[0]
    pad_len = int((blur_kernel.shape[2]-1)/2)
    if padding:
        im_HR_pad = F.pad(im_HR, (pad_len,)*4, mode='reflect')
    else:
        im_HR_pad = im_HR
    im_blur = F.conv3d(im_HR_pad.unsqueeze(0), blur_kernel.unsqueeze(1), groups=B)
    if downsampler.lower() == 'bicubic':
        im_blur = imresize(im_blur[0,], 1/scale)
    elif downsampler.lower() == 'direct':
        im_blur = im_blur[0, :, :, 0::scale, 0::scale]
    else:
        sys.exit('Please input corrected downsampler!')
    if not noise_level is None:
        im_LR = im_blur + torch.randn_like(im_blur)*noise_level
    else:
        im_LR = im_blur

    return im_LR, im_blur

