
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parents[1]))

import torch
import lpips
import pickle
import logging
import argparse
import numpy as np
from collections import OrderedDict
from skimage import img_as_float32, img_as_ubyte

from thop import profile
from networks.VIRNet import VIRAttResUNetSR

from utils import util_net
from utils import util_sisr
from utils import util_image
from utils import util_common
from utils.util_opts import str2bool

parser = argparse.ArgumentParser()
# trainning settings
parser.add_argument('--ckpt_path', type=str, default="", help="Checkpoint path")
parser.add_argument('--sf', type=int, default=4, help="Downsampling scale")
parser.add_argument('--nlevel', type=float, default=0, help="Noise Level")
parser.add_argument('--save_dir', type=str, default='', help="Path to save the results")
args = parser.parse_args()

# logging settings
nl_str = str(int(args.nlevel)) if args.nlevel == 0 else str(args.nlevel).replace('.', '')
log_name = 'virnet_sf{:d}_nl{:s}.log'.format(args.sf, nl_str)
log_path = Path(args.save_dir) / log_name
util_common.mkdir(args.save_dir, delete=False)
if log_path.exists():
    log_path.unlink()
logger = util_common.make_log(str(log_path), file_level=logging.INFO, stream_level=logging.INFO)

logger.info('=========================Optional settings=========================')
for key in vars(args):
    value = getattr(args, key)
    logger.info('{:12s}: {:s}'.format(key, str(value)))
logger.info('===================================================================')

# load the pretrained model
if args.ckpt_path:
    ckpt_path = args.ckpt_path
else:
    ckpt_path= str(Path('model_zoo') / f'virnet_sisr_x{args.sf}.pth')
logger.info('Loading the Model from: {:s}'.format(ckpt_path))
net = VIRAttResUNetSR(im_chn=3,
                      sigma_chn=1,
                      dep_S=5,
                      dep_K=8,
                      n_feat=[96, 160, 224],
                      n_resblocks=2,
                      extra_mode='Both',
                      noise_avg=True,
                      noise_cond=True,
                      kernel_cond=True,
                      ).cuda()
ckpt = torch.load(ckpt_path)['model_state_dict']
try:
    net.load_state_dict(ckpt)
except:
    net.load_state_dict(OrderedDict({key[7:]:value for key, value in ckpt.items()}), strict=True)
net.eval()

logger.info('\n')
logger.info('===========================Model Analysis==========================')
inputs1 = torch.randn(1, 3, int(256 / args.sf), int(256 / args.sf)).cuda()
flops1, _ = profile(net, inputs=(inputs1, args.sf, ))
inputs2 = torch.randn(1, 3, int(512 / args.sf), int(512 / args.sf)).cuda()
flops2, _ = profile(net, inputs=(inputs2, args.sf, ))
num_params = util_net.calculate_parameters(net) / 1000**2
elapsed_time = util_net.measure_time(net, inputs=(inputs1, args.sf,), num_forward=10)
logger.info('Number of parameters: {:.2f}M'.format(num_params))
logger.info('FLOPs for 256: {:.2f}G'.format(flops1 / 1000**3))
logger.info('FLOPs for 512: {:.2f}G'.format(flops2 / 1000**3))
logger.info('===================================================================')
logger.info('\n')

base_path = './test_data'
# data_sets = ['Set14', 'CBSD68', 'DIV2K100']
# exts      = ['.bmp',  '.png',   '.png']
data_sets = ['Set14', 'CBSD68']
exts      = ['.bmp',  '.png']

# kernel settings the kernel
num_kernel = 7
p = 21

# LPIPS setttings
lpips_calculator = lpips.LPIPS(net='alex').cuda()

logger.info('------------------------------Evaluation---------------------------------')
sf = args.sf
for data, ext in zip(data_sets, exts):
    im_path_list = sorted(list((Path(base_path) / data).glob('*' + ext)))
    assert len(im_path_list) in [5, 14, 68, 100]
    psnr_mean_kernel_y = ssim_mean_kernel_y = lpips_mean_kernel_rgb = 0
    for ind_kernel in range(num_kernel):
        if ind_kernel == 0:
            kernel = util_sisr.shifted_anisotropic_Gaussian(p, sf, (0.40*sf)**2, (0.40*sf)**2, 0, False)[0]
        elif ind_kernel == 1:
            kernel = util_sisr.shifted_anisotropic_Gaussian(p, sf, (0.60*sf)**2, (0.60*sf)**2, 0, False)[0]
        elif ind_kernel == 2:
            kernel = util_sisr.shifted_anisotropic_Gaussian(p, sf, (0.80*sf)**2, (0.80*sf)**2, 0, False)[0]
        elif ind_kernel == 3:
            kernel = util_sisr.shifted_anisotropic_Gaussian(p, sf, (0.4*sf)**2, (0.2*sf)**2, 0, False)[0]
        elif ind_kernel == 4:
            kernel = util_sisr.shifted_anisotropic_Gaussian(p, sf, (0.6*sf)**2, (0.3*sf)**2, 0.75*np.pi, False)[0]
        elif ind_kernel == 5:
            kernel = util_sisr.shifted_anisotropic_Gaussian(p, sf, (0.8*sf)**2, (0.4*sf)**2, 0.25*np.pi, False)[0]
        else:
            kernel = util_sisr.shifted_anisotropic_Gaussian(p, sf, (0.8*sf)**2, (0.4*sf)**2, 0.50*np.pi, False)[0]

        im_sr_res = {}
        im_lr_res = {}
        psnr_y = {}
        ssim_y = {}
        lpips_rgb = {}
        psnr_mean_y = ssim_mean_y = lpips_mean_rgb = 0
        for im_path in im_path_list:
            im_name = im_path.stem
            im_gt = util_image.imread(im_path, chn='rgb', dtype='uint8')
            if data == 'DIV2K100':
                im_gt = util_image.center_crop(im_gt, 1024)
            else:
                im_gt = util_sisr.modcrop(im_gt, args.sf)
            if im_gt.ndim == 2: im_gt = np.stack([im_gt,]*3, axis=2)

            # degradation
            im_lr = util_sisr.degrade_virnet(img_as_float32(im_gt),
                                             kernel=kernel,
                                             sf=sf,
                                             nlevel=args.nlevel,
                                             qf=None,
                                             downsampler='Bicubic')  # h x w x c, float32
            im_lr_res['im_'+im_name] = img_as_ubyte(np.clip(im_lr, 0.0, 1.0))
            inputs = torch.from_numpy(im_lr.transpose((2,0,1))[np.newaxis,]).type(torch.float32).cuda()
            with torch.set_grad_enabled(False):
                outputs, _, _ = net(inputs, args.sf)

            im_sr = img_as_ubyte(outputs.clamp(0.0, 1.0).cpu().squeeze(0).numpy().transpose((1,2,0)))  # h x w x c, uint8
            im_sr_res['im_'+im_name] = im_sr

            psnr_y_iter = util_image.calculate_psnr(im_sr, im_gt, args.sf**2, True)
            psnr_y['im_'+im_name] = psnr_y_iter
            psnr_mean_y += psnr_y_iter

            ssim_y_iter = util_image.calculate_ssim(im_sr, im_gt, args.sf**2, True)
            ssim_y['im_'+im_name] = ssim_y_iter
            ssim_mean_y += ssim_y_iter

            lpips_rgb_iter = lpips_calculator(util_image.normalize_lpips(im_sr).cuda(),
                                              util_image.normalize_lpips(im_gt).cuda()).item()
            lpips_rgb['im_'+im_name] = lpips_rgb_iter
            lpips_mean_rgb += lpips_rgb_iter

        psnr_mean_y /= len(im_path_list)
        psnr_y['mean'] = psnr_mean_y
        ssim_mean_y /= len(im_path_list)
        ssim_y['mean'] = ssim_mean_y
        lpips_mean_rgb /= len(im_path_list)
        lpips_rgb['mean'] = lpips_mean_rgb
        log_str = 'Dataset: {:>8s}, Kernel: {:d}, PSNRY: {:5.2f}, SSIMY: {:6.4f}, LPIPS: {:6.4f}'
        logger.info(log_str.format(data, ind_kernel+1, psnr_mean_y, ssim_mean_y, lpips_mean_rgb))
        if args.save_dir and data != 'DIV2K100':
            pkl_name = '{:s}_sf{:d}_kernel{:d}_nl{:s}.pkl'.format(data, args.sf, ind_kernel+1, nl_str)
            pkl_path = Path(args.save_dir) / pkl_name
            if pkl_path.exists(): pkl_path.unlink()
            with open(str(pkl_path), 'wb') as file:
                pickle.dump({'im_sr_res':im_sr_res,
                             'im_lr_res':im_lr_res,
                             'psnr_y':psnr_y,
                             'ssim_y':ssim_y,
                             'lpips_rgb':lpips_rgb}, file)

        psnr_mean_kernel_y += psnr_mean_y
        ssim_mean_kernel_y += ssim_mean_y
        lpips_mean_kernel_rgb += lpips_mean_rgb

    psnr_mean_kernel_y /= num_kernel
    ssim_mean_kernel_y /= num_kernel
    lpips_mean_kernel_rgb /= num_kernel

    log_str = 'Dataset: {:>8s}, PSNRY: {:5.2f}, SSIMY: {:6.4f}, LPIPS: {:6.4f}'
    logger.info('\n'+'='*80)
    logger.info(log_str.format(data, psnr_mean_kernel_y, ssim_mean_kernel_y, lpips_mean_kernel_rgb))
    logger.info('='*80+'\n')

