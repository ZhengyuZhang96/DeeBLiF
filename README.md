![Python 3.8](https://img.shields.io/badge/python-3.8-green.svg)

# DeeBLiF

Zhengyu Zhang, Shishun Tian, Wenbin Zou, Luce Morin, and Lu Zhang.

Official PyTorch code for our ICIP2022 paper "DeeBLiF: Deep blind light field image quality assessment by extracting angular and spatial information". Please refer to our [paper](https://ieeexplore-ieee-org.rproxy.insa-rennes.fr/document/9897951) for details.

**Note: We first convert the dataset into h5 files in MATLAB and then train/test the model in PYTHON.**

### Requirements
- PyTorch 1.7.1
- python 3.8

### Installation
Download this repository:
```
    $ git clone https://github.com/ZhengyuZhang96/DeeBLiF.git
```

### Generate Dataset in MATLAB
Convert the dataset into h5 files and put them in the root path ( ./DeeBLiF/Win5_160x160/... ):
```
    $ Generateh5_for_Win5_Dataset.m
```
or you can directly download the [generated h5 files](https://pan.baidu.com/s/1eEJWBegtkCyjqd-CIi96aw) (code: INSA).

### Usage
Train the model from scratch:
```
    $ python Train.py
```
Reproduce the performance in the paper with our [pre-trained models](https://pan.baidu.com/s/1eEJWBegtkCyjqd-CIi96aw) (code: INSA) ( ./DeeBLiF/PreTrainedModels/... ):
```
    $ python Test.py
```

### Citation
If you find this work helpful, please consider citing:
```
@inproceedings{zhang2022deeblif,
  title        = {Deeblif: Deep Blind Light Field Image Quality Assessment by Extracting Angular and Spatial Information},
  author       = {Zhang, Zhengyu and Tian, Shishun and Zou, Wenbin and Morin, Luce and Zhang, Lu},
  booktitle    = {2022 IEEE International Conference on Image Processing (ICIP)},
  pages        = {2266--2270},
  year         = {2022},
  organization = {IEEE}
}
```

## Contact
Welcome to raise issues or email to [zhengyu.zhang@insa-rennes.fr](zhengyu.zhang@insa-rennes.fr) for any question regarding this work.
