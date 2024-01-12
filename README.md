# PANet: Few-Shot Image Semantic Segmentation with Prototype Alignment

This repo contains code for our ICCV 2019 paper [PANet: Few-Shot Image Semantic Segmentation with Prototype Alignment](https://arxiv.org/abs/1908.06391).

### 依赖

* Python 3.6 +
* PyTorch 1.0.1
* torchvision 0.2.1
* NumPy, SciPy, PIL
* pycocotools
* sacred 0.7.5
* tqdm 4.32.2

### VOC数据集的数据准备

1. 下载 `SegmentationClassAug`, `SegmentationObjectAug`, `ScribbleAugAuto` from [here](https://drive.google.com/drive/folders/1N00R9m9qe2rKZChZ8N7Hib_HR2HGtXHp?usp=sharing) 从这里把它们放到 `VOCdevkit/VOC2012`.

2. 下载 `Segmentation` from [here](https://drive.google.com/drive/folders/1N00R9m9qe2rKZChZ8N7Hib_HR2HGtXHp?usp=sharing) 用它来代替 `VOCdevkit/VOC2012/ImageSets/Segmentation`.

路径：/media/fys/1e559c53-a2a9-468a-a638-74933f8e167c/lww/VOCdevkit/VOC2012

### 使用

1. 下载imagenet预训练的VGG16网络的权重 
`torchvision`: [https://download.pytorch.org/models/vgg16-397923af.pth](https://download.pytorch.org/models/vgg16-397923af.pth) 放在 `PANet/pretrained_model` 文件夹下.

2. 通过`config.py`更改配置，然后使用`python train.py`训练模型或使用`python test.py`测试模型。你可以使用`sacred`功能，例如:`python train.py with gpu_id=2`。

### Citation
Please consider citing our paper if the project helps your research. BibTeX reference is as follows.
```
@InProceedings{Wang_2019_ICCV,
author = {Wang, Kaixin and Liew, Jun Hao and Zou, Yingtian and Zhou, Daquan and Feng, Jiashi},
title = {PANet: Few-Shot Image Semantic Segmentation With Prototype Alignment},
booktitle = {The IEEE International Conference on Computer Vision (ICCV)},
month = {October},
year = {2019}
}
```
