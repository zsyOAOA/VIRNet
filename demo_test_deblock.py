
import sys
sys.path.append('./')

import torch
import cv2
import numpy as np
from pathlib import Path
from scipy.io import savemat
from networks.VIRNet import VIRNetBlock
from skimage import img_as_float32, img_as_ubyte
from utils import rgb2ycbcr, calculate_psnr, calculate_ssim, PadUNet

# load the pretrained model
print('Loading the Model')
net = VIRNetBlock(3, wf=64, dep_U=3, dep_S=5).cuda()
net.load_state_dict(torch.load(str(Path('./model_zoo/model_deblock.pt'))))
net.eval()

data = 'LIVE1'
img_lists = sorted(list(Path('./test_data/LIVE1').glob('*.bmp')))
for quality in [10, 20, 30, 40]:
    psnr_mean = ssim_mean = 0
    for im_path_iter in img_lists:
        im_name = im_path_iter.stem
        original_im_iter_BGR = cv2.imread(str(im_path_iter), cv2.IMREAD_COLOR)

        flag = cv2.imwrite('test.jpg', original_im_iter_BGR, [cv2.IMWRITE_JPEG_QUALITY, quality])
        assert flag
        compress_im_iter_RGB = img_as_float32(cv2.imread('test.jpg', cv2.IMREAD_COLOR)[:, :, ::-1])
        original_im_iter_RGB = original_im_iter_BGR[:, :, ::-1]

        inputs = torch.from_numpy(compress_im_iter_RGB.transpose((2,0,1))[np.newaxis,]).cuda()
        padunet = PadUNet(inputs, 3)
        with torch.set_grad_enabled(False):
            inputs = padunet.pad()
            outputs = net(inputs, 'test')
            outputs = padunet.pad_inverse(outputs[:, :3,]).squeeze()

        deblock_im_iter = img_as_ubyte(np.clip(outputs.cpu().numpy().transpose((1,2,0)), 0.0, 1.0))
        deblock_im_iter_y = rgb2ycbcr(deblock_im_iter, only_y=True)
        original_im_iter_y = rgb2ycbcr(original_im_iter_RGB, only_y=True)
        psnr_iter = calculate_psnr(deblock_im_iter_y, original_im_iter_y)
        psnr_mean += psnr_iter
        ssim_iter = calculate_ssim(deblock_im_iter_y, original_im_iter_y)
        ssim_mean += ssim_iter
        # print('Dataset: {:s}, Image: {:s}, PSNR: {:5.2f}, SSIM: {:6.4f}'.format(data, im_name,
                                                                              # psnr_iter, ssim_iter))
        Path('test.jpg').unlink()

    ssim_mean /= len(img_lists)
    psnr_mean /= len(img_lists)
    print('Dataset: {:>8s}, QF:{:d} Mean PSNR: {:5.2f}, Mean SSIM: {:6.4f}'.format(data, quality,
                                                                           psnr_mean, ssim_mean))


