#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2020-03-30 15:10:57

import torch
import torch.nn.functional as F
from math import ceil

def cubic(x):
    absx = torch.abs(x)
    absx2 = torch.abs(x)**2
    absx3 = torch.abs(x)**3

    condition1 = (absx<=1).to(torch.float32)
    condition2 = ((1<absx)&(absx<=2)).to(torch.float32)

    f = (1.5*absx3 - 2.5*absx2 +1)*condition1+(-0.5*absx3 + 2.5*absx2 -4*absx +2)*condition2
    return f

def contribute(in_size, out_size, scale, device):
    kernel_width = 4
    if scale<1:
        kernel_width /= scale
    x = torch.arange(start=1, end=out_size+1, dtype=torch.float32, device=device)
    u = (x-0.5) / scale + 0.5
    left = torch.floor(u-kernel_width/2)
    P = ceil(kernel_width)+2
    indice = left.unsqueeze(1) + torch.arange(start=0, end=P, dtype=torch.float32, device=device).unsqueeze(0) - 1
    indice = indice.type(torch.int64)
    mid = u.unsqueeze(1) - indice - 1  # H x P
    if scale < 1:
        weight = scale * cubic(mid*scale)
    else:
        weight = cubic(mid)   # H x P
    weight = weight / (torch.sum(weight, 1, keepdim=True)) # H x P

    aux = torch.cat((torch.arange(end=in_size, dtype=torch.int64, device=device),
                     torch.arange(start=in_size-1, end=-1, step=-1, dtype=torch.int64, device=device)))
    indice = aux[(indice % aux.numel())]

    mask = torch.gt(weight[0,].abs(), 0)
    weight = weight[:, mask]
    indice = indice[:, mask]

    return weight, indice

def imresize(x, scale=2):
    '''
    matlab imresize function, only bicubic interpolation
    Input:
        x: b x c x h x w torch tensor, torch.float32
        scale: scalar

    '''
    [b, c, h, w] = x.shape
    H = int(h*scale)
    W = int(w*scale)

    # scale the height dimension
    # weight0: H x 4, indice0: H x 4
    weight0, indice0 = contribute(h, H, scale, x.device)
    out = x[:, :, indice0, :]*(weight0.unsqueeze(0).unsqueeze(1).unsqueeze(4)) # b x c x H x 4 x w
    out = torch.sum(out, dim=3)   # b x c x H x w

    # scale the width dimension
    A = out.permute(0,1,3,2)
    # weight1: W x 4, indice0: W x 4
    weight1, indice1 = contribute(w, W, scale, x.device)
    out = A[:, :, indice1, :]*(weight1.unsqueeze(0).unsqueeze(1).unsqueeze(4))
    out = torch.sum(out,dim = 3).permute(0,1,3,2)

    return out

