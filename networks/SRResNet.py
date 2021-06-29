#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2020-04-12 21:23:53

import torch
import torch.nn.init as init
from torch import nn
from .SubBlocks import conv3x3

class SRResNetV1(nn.Module):
    '''
    SRResNet for Denoising task.
    https://arxiv.org/abs/1609.04802
    '''
    def __init__(self, in_chn=3, out_chn=3, nb=16, nc=64):
        super(SRResNetV1, self).__init__()
        self.head = nn.Sequential(
            conv3x3(3, nc, bias=True),
            nn.LeakyReLU(0.2, True)
        )

        self.body = nn.ModuleList()
        for _ in range(nb):
            self.body.append(ResidualBlock(nc))
        self.body.append(conv3x3(nc, nc, bias=True))
        self.body.append(nn.BatchNorm2d(nc))

        self.tail = conv3x3(nc, out_chn, bias=True)

        self._initialize()

    def forward(self, x):
        x1 = self.head(x)

        x2 = x1.clone()
        for block in self.body:
            x2 = block(x2)
        x3 = x2 + x1

        out = self.tail(x3)

        return out

    def _initialize(self):
        gain = nn.init.calculate_gain('leaky_relu', 0.20)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal(m.weight, gain=gain)
                if not m.bias is None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

class SRResNetV2(nn.Module):
    '''
    SRResNet for Super-Resolution task.
    https://arxiv.org/abs/1609.04802
    '''
    def __init__(self, in_chn=3, nb=16, nc=64, scale=2):
        super(SRResNetV2, self).__init__()
        self.head = nn.Sequential(
            conv3x3(3, nc, bias=True),
            nn.LeakyReLU(0.2, True)
        )

        self.body = nn.ModuleList()
        for _ in range(nb):
            self.body.append(ResidualBlock(nc))
        self.body.append(conv3x3(nc, nc, bias=True))
        self.body.append(nn.BatchNorm2d(nc))

        if scale == 4:
            self.upblock1 = UpsampleBLock4(nc)
            self.upblock2 = UpsampleBLock4(nc)
        else:
            self.upblock1 = UpsampleBLock(nc, scale)
            self.upblock2 = UpsampleBLock(nc, scale)

        self.tail1 = nn.Conv2d(nc, in_chn, kernel_size=1, bias=True, stride=1, padding=0)
        self.tail2 = nn.Conv2d(nc, in_chn, kernel_size=1, bias=True, stride=1, padding=0)

        self._initialize()

    def forward(self, x, mode='train'):
        x1 = self.head(x)

        x2 = x1.clone()
        for block in self.body:
            x2 = block(x2)
        x3 = x2 + x1

        x_up1 = self.upblock1(x3)
        out1 = self.tail1(x_up1)

        if mode.lower() == 'train':
            x_up2 = self.upblock2(x3)
            out2 = self.tail2(x_up2)
            return torch.cat([out1, out2], dim=1)
        else:
            return out1

    def _initialize(self):
        gain = nn.init.calculate_gain('leaky_relu', 0.20)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal(m.weight, gain=gain)
                if not m.bias is None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(channels, channels, bias=True)
        self.bn1 = nn.BatchNorm2d(channels)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.conv2 = conv3x3(channels, channels, bias=True)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.lrelu(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)

        return x + residual

class UpsampleBLock(nn.Module):
    def __init__(self, in_chn, up_scale):
        super(UpsampleBLock, self).__init__()
        self.conv = conv3x3(in_chn,  in_chn*(up_scale**2), bias=True)
        self.pixel_shuffle = nn.PixelShuffle(up_scale)
        self.lrelu = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.lrelu(x)
        return x

class UpsampleBLock4(nn.Module):
    def __init__(self, in_chn):
        super(UpsampleBLock4, self).__init__()
        self.conv1 = conv3x3(in_chn, in_chn*4, bias=True)
        self.pixel_shuffle1 = nn.PixelShuffle(2)
        self.lrelu1 = nn.LeakyReLU(0.2, True)
        self.conv2 = conv3x3(in_chn, in_chn*4, bias=True)
        self.pixel_shuffle2 = nn.PixelShuffle(2)
        self.lrelu2 = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        x1 = self.lrelu1(self.pixel_shuffle1(self.conv1(x)))
        out = self.lrelu2(self.pixel_shuffle2(self.conv2(x1)))
        return out
