#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2019-03-20 19:48:14

import torch
from math import pi, log, sqrt
import torch.nn.functional as F
import torch.distributions as tdist
from utils import util_sisr 
from ResizeRight.resize_right import resize

def cal_kl_inverse_gamma_simple(beta_q, alpha_p, beta_p):
    out = alpha_p*(beta_p.div(beta_q)-1) + alpha_p*(beta_q.log()-beta_p.log())
    return out.mean()

def cal_kl_gauss_simple(mu_q, mu_p, var_p): return 0.5 * ((mu_q-mu_p)**2/var_p).mean()

def cal_likelihood(x, mu_q, var_q, alpha_q, beta_q):
    temp =  0.5 * (beta_q.log()-alpha_q.digamma() + alpha_q.div(beta_q)*((x-mu_q)**2+var_q))  
    out = temp + 0.5*log(2*pi)
    return out.mean()

def elbo_denoising_simple(mu, sigma_est, im_noisy, im_gt, eps2, alpha0, beta0):
    '''
    Input:
        mu: output of the RNet
        sigma_est: output of SNet
    '''
    # KL divergence for Gauss distribution
    if isinstance(mu, list):
        kl_gauss = cal_kl_gauss_simple(mu[0], im_gt, eps2)
        for jj in range(1, len(mu)):
            kl_gauss += cal_kl_gauss_simple(mu[jj], im_gt, eps2)
        kl_gauss /= len(mu)
    else:
        kl_gauss = cal_kl_gauss_simple(mu, im_gt, eps2)

    # KL divergence for Inv-Gamma distribution
    beta = sigma_est * alpha0
    kl_Igamma = cal_kl_inverse_gamma_simple(beta, alpha0-1, beta0)

    # likelihood 
    if isinstance(mu, list):
        lh = cal_likelihood(im_noisy, mu[0], eps2, alpha0-1, beta)
        for jj in range(1, len(mu)):
            lh += cal_likelihood(im_noisy, mu[jj], eps2, alpha0-1, beta)
        lh /= len(mu)
    else:
        lh = cal_likelihood(im_noisy, mu, eps2, alpha0-1, beta)

    loss = lh + kl_gauss + kl_Igamma

    return loss, lh, kl_gauss, kl_Igamma

def cal_likelihood_sisr(x, kernel, sf, mu_q, var_q, alpha_q, beta_q, downsampler):
    zz = mu_q + torch.randn_like(mu_q) * sqrt(var_q)
    zz_blur = util_sisr.conv_multi_kernel_tensor(zz, kernel, sf, downsampler)   
    out = 0.5*log(2*pi) +  0.5*(beta_q.log()-alpha_q.digamma()) +  0.5*alpha_q.div(beta_q)*(x-zz_blur)**2  
    return out.mean()

def reparameter_inv_gamma(alpha, beta):
    dist_gamma = tdist.gamma.Gamma(alpha, beta)
    out = 1 / dist_gamma.rsample()
    return out

def reparameter_cov_mat(kinfo_est, kappa0, rho_var):
    '''
    Reparameterize kernelo.
    Input:
        kinfo_est: N x 3
    '''
    alpha_k = torch.ones_like(kinfo_est[:, :2]) * (kappa0-1)
    beta_k = kinfo_est[:, :2] * kappa0
    k_var = reparameter_inv_gamma(alpha_k, beta_k)
    k_var1, k_var2 = torch.chunk(k_var, 2, dim=1)    # N x 1, resampled variance along x and y axis
    rho_mean = kinfo_est[:, 2].unsqueeze(1)          # N x 1, mean of the correlation coffecient
    rho = rho_mean + sqrt(rho_var)*torch.randn_like(rho_mean)  # resampled correlation coffecient
    direction = k_var1.detach().sqrt() * k_var2.detach().sqrt() * torch.clamp(rho, min=-1, max=1)   # N x 1
    k_cov = torch.cat([k_var1, direction, direction, k_var2], dim=1).view(-1, 1, 2, 2) # N x 1 x 2 x 2
    return k_cov

def elbo_sisr(mu, sigma_est,
              kinfo_est,
              im_hr,
              im_lr,
              sigma_prior,
              alpha0,
              kinfo_gt,
              kappa0,
              r2,
              eps2,
              sf,
              k_size,
              penalty_K,
              shift,
              downsampler):
    '''
    Input:
        mu: output of RNet, mean value of the Gaussian posterior of Z
        sigma_est: output of SNet, estimated sigma map of the noise
        kinfo_est: output of KNet, estimated kernel information, sigma1, sigma2, rho
    '''
    # KL divergence for Gauss distribution
    if isinstance(mu, list):
        kl_rnet = cal_kl_gauss_simple(mu[0], im_hr, eps2)
        for jj in range(1, len(mu)):
            kl_rnet += cal_kl_gauss_simple(mu[jj], im_hr, eps2)
        kl_rnet /= len(mu)
    else:
        kl_rnet = cal_kl_gauss_simple(mu, im_hr, eps2)

    # KL divergence for Inv-Gamma distribution of the sigma map for noise
    beta0 = sigma_prior * alpha0
    beta = sigma_est * alpha0
    kl_snet = cal_kl_inverse_gamma_simple(beta, alpha0-1, beta0)

    # KL divergence for the kernel
    kl_knet0 = cal_kl_inverse_gamma_simple(kappa0*kinfo_est[:, 0], kappa0-1, kappa0*kinfo_gt[:, 0])
    kl_knet1 = cal_kl_inverse_gamma_simple(kappa0*kinfo_est[:, 1], kappa0-1, kappa0*kinfo_gt[:, 1])
    kl_knet2 = cal_kl_gauss_simple(kinfo_est[:, 2], kinfo_gt[:, 2], r2) * penalty_K[0]
    kl_knet = (kl_knet0 + kl_knet1 + kl_knet2) / 3 * penalty_K[1]

    # reparameter kernel
    k_cov = reparameter_cov_mat(kinfo_est, kappa0, r2)        # resampled covariance matrix, N x 1 x 2 x 2
    kernel = util_sisr.sigma2kernel(k_cov, k_size, sf, shift)        # N x 1 x k x k

    # likelihood
    if isinstance(mu, list):
        lh = cal_likelihood_sisr(im_lr, kernel, sf, mu[0], eps2, alpha0-1, beta, downsampler)
        for jj in range(1, len(mu)):
            lh += cal_likelihood_sisr(im_lr, kernel, sf, mu[jj], eps2, alpha0-1, beta, downsampler)
        lh /= len(mu)
    else:
        lh = cal_likelihood_sisr(im_lr, kernel, sf, mu, eps2, alpha0-1, beta, downsampler)

    loss = lh + kl_rnet + kl_snet + kl_knet

    return loss, [lh, kl_rnet, kl_snet, kl_knet, kl_knet0, kl_knet1, kl_knet2, kernel]

