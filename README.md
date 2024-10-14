# Deep Variational Network Toward Blind Image Restoration (TPAMI, 2024) 
[Zongsheng Yue](https://zsyoaoa.github.io/), [Hongwei Yong](https://sites.google.com/view/yonghongwei-homepage), [Qian Zhao](https://gr.xjtu.edu.cn/en/web/timmy.zhaoqian/home), [Lei Zhang](https://www4.comp.polyu.edu.hk/~cslzhang/), [Deyu Meng](https://gr.xjtu.edu.cn/en/web/dymeng/1), [Kwan-Yee K. Wong](https://i.cs.hku.hk/~kykwong/) 

[Paper](https://arxiv.org/abs/2008.10796) 

Note that this work is an extended version of our work VDN ([paper](https://papers.nips.cc/paper/2019/file/6395ebd0f4b478145ecfbaf939454fa4-Paper.pdf), [code](https://github.com/zsyOAOA/VDNet)) that publised on the NeurIPS 2019. In the extended version, we further imporve our method both from model construction and algorithm design, and make it capable of handling blind image super-resolution.

:star: If our work is helpful to your research, please help star this repo. Thanks! :hugs: 

---
Blind image restoration (IR) is a common yet challenging problem in computer vision. Classical model-based methods and recent deep learning (DL)-based methods represent two different methodologies for this problem, each with their own merits and drawbacks. In this paper, we propose a novel blind image restoration method, aiming to integrate both the advantages of them. Specifically, we construct a general Bayesian generative model for the blind IR, which explicitly depicts the degradation process. In this proposed model, a pixel-wise non-i.i.d. Gaussian distribution is employed to fit the image noise. It is with more flexibility than the simple i.i.d. Gaussian or Laplacian distributions as adopted in most of conventional methods, so as to handle more complicated noise types contained in the image degradation. To solve the model, we design a variational inference algorithm where all the expected posteriori distributions are parameterized as deep neural networks to increase their model capability. Notably, such an inference algorithm induces a unified framework to jointly deal with the tasks of degradation estimation and image restoration. Further, the degradation information estimated in the former task is utilized to guide the latter IR process. Experiments on two typical blind IR tasks, namely image denoising and super-resolution, demonstrate that the proposed method achieves superior performance over current state-of-the-arts.

><img src="./figures/Framework.png" align="middle" width="800">
---

## Requirements
* Python 3.8, Pytorch 1.13.0
* More detail (See [environment.yml](environment.yml))
A suitable [conda](https://conda.io/) environment named `virnet` can be created and activated with:

```
conda create -n virnet python=3.8 -y
conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.6 -c pytorch -c nvidia
pip install -r requirements.txt
conda activate virnet
```

## :snowflake: Testing Demo
Befor testing, please first download the checkpoint from this [link](https://github.com/zsyOAOA/VIRNet/releases/tag/v1.0) and put them in the foloder "model_zoo".
1. General testing demo.
```
    python scripts/testing_demo.py --task task_name --in_path: input_path --out_path: output_path --sf sr_scale
```
+ --task: task name, "denoising-syn", "denoising-real", "sisr"
+ --in_path: input path of the low-quality images, image path or folder  
+ --out_path: output folder
+ --sf: scale factor for image super-resolution, 2, 3, or 4

2. Reproduce the results in Table 1 of our paper.
```
    python scripts/denoising_virnet_syn.py --save_dir output_path --noise_type niid
```
3. Reproduce the results in Table 2 of our paper.
```
    python scripts/denoising_virnet_syn.py --save_dir output_path --noise_type iid
```
4. Reproduce the results on SIDD dataset in Table 4 of our paper.
```
    python scripts/denoising_virnet_real_sidd.py --save_dir output_path --sidd_dir sidd_data_path
```
5. Reproduce the results on DND dataset in Table 4 of our paper.
```
    python scripts/denoising_virnet_real_dnd.py --save_dir output_path --dnd_dir dnd_data_path
```
6. Reproduce the results of super-resolution in Table 5 of our paper.
```
    python scripts/sisr_virnet_syn.py --save_dir output_path --sf 4 --nlevel 0.1
```
+ --nlevel: noise level, 0.1, 2.55 or 7.65

## :sunny: Training Pipeline:
### :pear: Image Denoising on Synthetic Data:
1. Download the source images from [Waterloo](https://kedema.org/project/exploration/index.html), [CBSD432](https://drive.google.com/folderview?id=0B-_yeZDtQSnobXIzeHV5SjY5NzA&usp=sharing), [Flick2K](http://cv.snu.ac.kr/research/EDSR/Flickr2K.tar) and [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) as groundtruth and fill the data path in the [config](configs/denoising_syn.json) file.

2. Begin to train: 

```
    CUDA_VISIBLE_DEVICES=gpu_id python train_denoising_syn.py --save_dir path_for_log
```

### :apple: Image Denoising on Real-world Data:
1. Download the training datasets [SIDD](ftp://sidd_user:sidd_2018@130.63.97.225/SIDD_Medium_Srgb.zip) and validation datasets [noisy](ftp://sidd_user:sidd_2018@130.63.97.225/SIDD_Blocks/ValidationNoisyBlocksSrgb.mat), [groundtruth](ftp://sidd_user:sidd_2018@130.63.97.225/SIDD_Blocks/ValidationGtBlocksSrgb.mat).

2. Crop the training datasets into small image patches using this [script](datasets/prepare_data/Denoising/SIDD/im2patch_train.py), and fill the data path in the [config](configs/denoising_real.json) file.

3. Begin to training:
```
    CUDA_VISIBLE_DEVICES=gpu_id python train_denoising_real.py --save_dir path_for_log
```

### :peach: Image Super-resolution
1. Download the high-resolution images of [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) and [Flick2K](http://cv.snu.ac.kr/research/EDSR/Flickr2K.tar), and crop them into small image patches using this [script](datasets/prepare_data/SISR/im2patch_train.py).

2. Fill data path in the [config](configs/sisr_x4.json) file.

3. Begin to train:

```
    CUDA_VISIBLE_DEVICES=gpu_id python train_SISR.py --save_dir path_for_log --config configs/sisr_x4.json
```

### :dolphin: Citation
```
    @article{yue2024variational,
      title={Deep Variational Network Toward Blind Image Restoration},
      author={Yue, Zongsheng and Yong, Hongwei and Zhao, Qian and Zhang, Lei and Meng, Deyu and Wong, Kwan-Yee K},
      journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
      year={2024}
    }
```

### :envelope: Contact
If you have any questions, please feel free to contact me via `zsyzam@gmail.com`.

