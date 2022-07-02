# VIRNet
# Deep Variational Network Toward Blind Image Restoration [arXiv](https://arxiv.org/abs/2008.10796)
Note that this work is an extended version of VDN \([paper](https://papers.nips.cc/paper/2019/file/6395ebd0f4b478145ecfbaf939454fa4-Paper.pdf), [code](https://github.com/zsyOAOA/VDNet)\) that publised on the NeurIPS 2019. In the extended version, we further imporve our method both from model construction and algorithm design, and make it capable of handling blind image super-resolution.

---
Blind image restoration (IR) is a common yet challenging problem in computer vision. Classical model-based methods and recent deep learning (DL)-based methods represent two different methodologies for this problem, each with their own merits and drawbacks. In this paper, we propose a novel blind image restoration method, aiming to integrate both the advantages of them. Specifically, we construct a general Bayesian generative model for the blind IR, which explicitly depicts the degradation process. In this proposed model, a pixel-wise non-i.i.d. Gaussian distribution is employed to fit the image noise. It is with more flexibility than the simple i.i.d. Gaussian or Laplacian distributions as adopted in most of conventional methods, so as to handle more complicated noise types contained in the image degradation. To solve the model, we design a variational inference algorithm where all the expected posteriori distributions are parameterized as deep neural networks to increase their model capability. Notably, such an inference algorithm induces a unified framework to jointly deal with the tasks of degradation estimation and image restoration. Further, the degradation information estimated in the former task is utilized to guide the latter IR process. Experiments on two typical blind IR tasks, namely image denoising and super-resolution, demonstrate that the proposed method achieves superior performance over current state-of-the-arts.

><img src="./figures/Framework.png" align="middle" width="800">
---

# Requirements and Dependencies
* Ubuntu 16.04, cuda 10.0
* Python 3.7.5, Pytorch 1.3.1
* More detail (See environment.yml)

# Image Denoising
## Non-IID Gaussian Noise Removal
### Training

1. Download the source images from [Waterloo](https://ece.uwaterloo.ca/~k29ma/exploration/), [CBSD432](https://drive.google.com/folderview?id=0B-_yeZDtQSnobXIzeHV5SjY5NzA&usp=sharing) and [CImageNet400](https://drive.google.com/folderview?id=0B-_yeZDtQSnobXIzeHV5SjY5NzA&usp=sharing) as groundtruth. According to your own data path, modify this [config](configs/denoising_simulation_niid.json) file.

2. Training 

```
    python train_denoising_syn.py --gpu_id 0 --save_dir path_for_log
```

### Testing
```
    python demo_test_syn.py
```

## Real-world Noise Removal
### Training:
1. Download the training datasets [SIDD](ftp://sidd_user:sidd_2018@130.63.97.225/SIDD_Medium_Srgb.zip) and validation datasets \([noisy](ftp://sidd_user:sidd_2018@130.63.97.225/SIDD_Blocks/ValidationNoisyBlocksSrgb.mat), [groundtruth](ftp://sidd_user:sidd_2018@130.63.97.225/SIDD_Blocks/ValidationGtBlocksSrgb.mat)\).

2. Crop the training datasets into small image patches using this [script](datasets/prepare_data/Denoising/SIDD/im2patch_train.py), and modify the [config](configs/denoising_real.json) file.

3. Training (Supporting distributed training):
```
    python train_denoising_real.py --gpu_id 01 --save_dir path_for_log
```

### Testing:
```
    python demo_test_denoising_real.py
```

# Image Super-resolution
## Training:
1. Download the high-resolution images of DIV2K and Flick2K from this [link](https://cvnote.ddlee.cc/2019/09/22/image-super-resolution-datasets), and crop them into small image patches.
```
    python datasets/prepare_data/SISR/im2patch_train.py --DIV2K_dir DIV2K_HR_path --DIV2K_dir Flick2K_HR_path
```

2. Modify the [config](configs/sisr_x4.json) file according to your own settings.

3. Training:

```
    python train_sisr.py --gpu_id 0 --save_dir path_for_log
```
```
    python -m torch.distributed.launch --nproc_per_node=4 train_sisr.py --gpu_id 0123 --save_dir path_for_log
```
## Testing:
```
    python demo_test_sisr.py --sf 4 --noise_level 2.55
```

# Citation
```
    @article{yue2020variational,
      title={Variational Image Restoration Network},
      author={Yue, Zongsheng and Yong, Hongwei and Zhao, Qian and Zhang, Lei and Meng, Deyu},
      journal={arXiv preprint arXiv:2008.10796},
      year={2020}
    }
```
