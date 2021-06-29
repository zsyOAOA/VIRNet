# VIRNet
# Variational Image Restoration Network [arXiv](https://arxiv.org/abs/2008.10796)
Note that this work is an extended version of VDN \([paper](https://papers.nips.cc/paper/2019/file/6395ebd0f4b478145ecfbaf939454fa4-Paper.pdf), [code](https://github.com/zsyOAOA/VDNet)\) that publised on the NeurIPS 2019.

# Requirements and Dependencies
* Ubuntu 16.04, cuda 10.0
* Python 3.7.5, Pytorch 1.3.1
* More detail (See environment.yml)

# Image Denoising
## Non-IID Gaussian Noise Removal
### Training

1. Download the source images from [Waterloo](https://ece.uwaterloo.ca/~k29ma/exploration/), [CBSD432](https://drive.google.com/folderview?id=0B-_yeZDtQSnobXIzeHV5SjY5NzA&usp=sharing) and [CImageNet400](https://drive.google.com/folderview?id=0B-_yeZDtQSnobXIzeHV5SjY5NzA&usp=sharing) as groundtruth. According to your own data path, modify this [config](configs/denoising_simulation_niid.json) file.

2. Prepare the testing datasets:
```
    python datasets/prepare_data/Denoising/simulation/noise_generate_niid.py
```
3. Begin training:
```
    python train_denoising_simulation.py 
```

### Testing
```
    python demo_test_simulation.py
```

## Real-world Noise Removal
### Training:
1. Download the training datasets \([SIDD](ftp://sidd_user:sidd_2018@130.63.97.225/SIDD_Medium_Srgb.zip), [RENOIR](http://ani.stat.fsu.edu/~abarbu/Renoir.html) [Ploy](https://github.com/csjunxu/PolyU-Real-World-Noisy-Images-Dataset)\) and validation datasets \([noisy](ftp://sidd_user:sidd_2018@130.63.97.225/SIDD_Blocks/ValidationNoisyBlocksSrgb.mat), [groundtruth](ftp://sidd_user:sidd_2018@130.63.97.225/SIDD_Blocks/ValidationGtBlocksSrgb.mat)\).

2. Write the training datasets (SIDD and RENOIR) and validation datasets (SIDD) into hdf5 format using the code
in this [folder](datasets/prepare_data), and modify the [config](configs/denoising_real.json) file.

3. Begin training:
```
    python train_denoising_real.py 
```

### Testing:
```
    python demo_test_denoising_real.py
```

# Image Super-resolution
## Training:
1. Download the high-resolution images of [DIV2K](http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip), and write them into hdf5 format.
```
    python datasets/prepare_data/SISR/big2small_DIV2K_train.py --data_dir your_DIV2K_HR_path
```

2. Modify the [config](configs/sisr_general.json) file according to your own settings.

3. Begin training:
```
    python train_SISR.py 
```
## Testing:
```
    python demo_test_sisr.py
```

# JPEG Image Deblocking
## Training:
1. Download the high-resolution images of [DIV2K](http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip), and write them into hdf5 format.
```
    python datasets/prepare_data/Deblocking/generate_training.py --data_dir your_DIV2K_HR_path --save_dir your_path_save
```

2. Modify the [config](configs/deblocking.json) file according to your own settings.

3. Begin training:
```
    python train_deblock.py 
```
## Testing:
```
    python demo_test_deblock.py
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
