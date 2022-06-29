import torch
import torch.nn as nn
from math import log

log_max = log(1e2)
log_min = log(1e-4)

def num2bool(ii, mod=2):
    out = True if (ii+1) % mod == 1 else False
    return out

class CALayer(nn.Module):
    def __init__(self, nf, reduction=16):
        super(CALayer, self).__init__()
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.body = nn.Sequential(
            nn.Conv2d(nf, nf // reduction, 1, 1, 0),
            nn.LeakyReLU(0.2),
            nn.Conv2d(nf // reduction, nf, 1, 1, 0),
            nn.Sigmoid(),
        )

    def forward(self, x):
        y = self.avg(x)
        y = self.body(y)
        return torch.mul(x, y)

class RB_Layer(nn.Module):
    def __init__(self, nf):
        super(RB_Layer, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(nf, nf, 3, 1, 1),
                                  nn.LeakyReLU(0.2, True),
                                  nn.Conv2d(nf, nf, 3, 1, 1),
                                  CALayer(nf))

    def forward(self, x):
        out = self.body(x) + x
        return out

class KernelNet(nn.Module):
    def __init__(self, in_nc=3, out_chn=3, nf=64, num_blocks=8, scale=4):
        super(KernelNet, self).__init__()

        self.head = nn.Conv2d(in_nc, nf, kernel_size=9, stride=4, padding=4, bias=False)

        self.body = nn.Sequential(*[RB_Layer(nf) for ii in range(num_blocks)])

        self.tail = nn.Sequential(nn.Conv2d(nf, out_chn, 3, 1, 1),
                                  nn.AdaptiveAvgPool2d((1, 1)))

    def forward(self, x):
        x_head = self.head(x)
        x_body = self.body(x_head)
        out = self.tail(x_body)                     # N x 3 x 1 x 1
        lam12 = torch.exp(torch.clamp(out[:, :2,], min=log_min, max=log_max)) # N x 2 x 1 x 1
        rho = torch.tanh(out[:, -1, ]).unsqueeze(1) # N x 1 x 1 x 1
        Lam = torch.cat((lam12, rho), dim=1)        # N x 3 x 1 x 1
        return Lam
