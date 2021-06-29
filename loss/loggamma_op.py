#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2020-03-30 14:59:29

import torch
from scipy.special import gammaln
from torch.autograd import Function as autoF

class LogGamma(autoF):
    '''
    Implement of the logarithm of gamma Function.
    '''
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        if input.is_cuda:
            input_np = input.detach().cpu().numpy()
        else:
            input_np = input.detach().numpy()
        out = gammaln(input_np)
        out = torch.from_numpy(out).to(device=input.device).type(dtype=input.dtype)

        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = torch.digamma(input) * grad_output

        return grad_input
