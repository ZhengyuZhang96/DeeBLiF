![Python 3.8](https://img.shields.io/badge/python-3.8-green.svg)

# DeeBLiF

Zhengyu Zhang, Shishun Tian, Wenbin Zou, Luce Morin, and Lu Zhang.

Official PyTorch code for our ICIP2022 paper "DeeBLiF: Deep blind light field image quality assessment by extracting angular and spatial information". Please refer to our [paper](https://ieeexplore.ieee.org/abstract/document/9897951) for details.

**Note: We first convert the dataset into h5 files in MATLAB and then train/test the model in PYTHON.**

**Hope our work is helpful to you :)**

### Requirements
- PyTorch 1.7.1
- python 3.8

### Installation
Download this repository:
```
    $ git clone https://github.com/ZhengyuZhang96/DeeBLiF.git
```

### Generate Dataset in MATLAB 
Take the Win5-LID dataset for instance, download the .m file on [Google drive](https://drive.google.com/drive/folders/1EDCKqoLUAx-cuf21ROTLKnQWvxN1H7L_?usp=sharing) or [Baidu drive](https://pan.baidu.com/s/1teWdRY4_XGiUC07h8DZDHw) (code: INSA), convert the dataset into h5 files, and then put them into './DeeBLiF/Datasets/Win5_160x160/...':
```
    $ ./DeeBLiF/Datasets/Generateh5_for_Win5_Dataset.m
```
or you can directly download the generated h5 files on [Google drive](https://drive.google.com/drive/folders/1EDCKqoLUAx-cuf21ROTLKnQWvxN1H7L_?usp=sharing) or [Baidu drive](https://pan.baidu.com/s/1teWdRY4_XGiUC07h8DZDHw) (code: INSA).

### Train
Train the model from scratch:
```
    $ python Train.py  --trainset_dir ./Datasets/Win5_160x160/
```

### Test overall performance
Reproduce the performance in the paper: download our pre-trained models on [Google drive](https://drive.google.com/drive/folders/1EDCKqoLUAx-cuf21ROTLKnQWvxN1H7L_?usp=sharing) or [Baidu drive](https://pan.baidu.com/s/1teWdRY4_XGiUC07h8DZDHw) (code: INSA) and put them into './DeeBLiF/PreTrainedModels/Win5/...'.
```
    $ python Test.py
```

### Test individual distortion type performance
Test the performance of individual distortion type by the following script. 
```
    $ python Test_Dist.py
```

### Results
Our paper only provides the experimental results of the overall performance on the Win5-LID dataset, here we additionally provide the individual distortion type performance of the Win5-LID dataset, and the individual distortion type performance and overall performance of the NBU-LF1.0 and SHU datasets. Alternatively, you can reproduce these performances using the h5 results we provide in './DeeBLiF/Results/...'.

**Win5-LID dataset:**
| **Distortion types** | **PLCC** | **SROCC** | **KROCC** | **RMSE** |
|  :---------: | :----------: | :----------: | :----------: | :----------: |
|    HEVC  |  0.9389  |  0.9103  |  0.7988  |  0.3406  |
|    JPEG2000  |  0.9254  |  0.8686  |  0.7508  |  0.3257  |
|    LN  |  0.9021  |  0.7914  |  0.6548  |  0.2964  |
|    NN  |  0.9207  |  0.8628  |  0.7382  |  0.2701  |
|    Overall  |  0.8427  |  0.8186  |  0.6502  |  0.5160  |

**NBU-LF1.0 dataset:**
| **Distortion types** | **PLCC** | **SROCC** | **KROCC** | **RMSE** |
|  :---------: | :----------: | :----------: | :----------: | :----------: |
|    NN  |  0.9610  |  0.9168  |  0.8100  |  0.1843  |
|    BI  |  0.9499  |  0.8986  |  0.7918  |  0.2736  |
|    EPICNN  |  0.9395  |  0.8027  |  0.6899  |  0.2283  |
|    Zhang  |  0.6659  |  0.5832  |  0.5003  |  0.4365  |
|    VDSR  |  0.9487  |  0.9062  |  0.8042  |  0.2614  |
|    Overall  |  0.8583  |  0.8229  |  0.6515  |  0.4588  |

**SHU dataset:**
| **Distortion types** | **PLCC** | **SROCC** | **KROCC** | **RMSE** |
|  :---------: | :----------: | :----------: | :----------: | :----------: |
|    GAUSS  |  0.9556  |  0.9507  |  0.8609  |  0.2238  |
|    JPEG2000  |  0.9031  |  0.8980  |  0.7962  |  0.1620  |
|    JPEG  |  0.9804  |  0.9567  |  0.8641  |  0.2040  |
|    Motion Blur  |  0.9676  |  0.9474  |  0.8516  |  0.2099  |
|    White Noise  |  0.9553  |  0.9527  |  0.8709  |  0.2832  |
|    Overall  |  0.9548  |  0.9419  |  0.8149  |  0.3185  |

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

### Correction
In our paper, we claim that we use "K-fold corss-validation" strategy to conducet the experiments. However, it should actually be "Leave-two-fold-out corss-validation". We sincerely apologize for any confusion or inconvenience caused by this wrong expression.


## Contact
Welcome to raise issues or email to [zhengyu.zhang@insa-rennes.fr](zhengyu.zhang@insa-rennes.fr) for any question regarding this work.
