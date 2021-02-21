## Description
This is a PyTorch Re-Implementation of CRAFT: Character Region Awareness for Text Detection.

| Model | Recall | Precision | F-score | 
| - | - | - | - |
| Original | 93.1 | 97.4 | 95.2 |
| Re-Implement | 91.98 | 92.32 | 92.15 |

## Prerequisites
Only tested on
* Anaconda3
* Python 3.7.1
* PyTorch 1.0.1
* opencv-python 4.0.0.21
* easydict 1.9

## Installation
### 1. Clone the repo

```
git clone https://github.com/SakuraRiven/CRAFT.git
cd CRAFT
```

### 2. Data & Pre-Trained Model
* [ICDAR2013](https://rrc.cvc.uab.es/?ch=2&com=evaluation&task=1&f=1&e=2), [ICDAR2017](https://rrc.cvc.uab.es/?ch=8&com=evaluation&task=1), [SynthText](http://www.robots.ox.ac.uk/~vgg/data/scenetext/)

* [VGG16](https://drive.google.com/open?id=1WzDnOuU_dELMDSaecs1e9uc0aqRBHQX4)

* [Pre-trained model](https://drive.google.com/open?id=1RROAUqBQsydRhGpmUwTT1zu3yWNoVWsO)

* [Finetune model](https://drive.google.com/open?id=1aYa7vv3jOx3TJpyz-CpMbe5pq8ec8dtI)

Make a new folder ```pths``` and put the download pths into ```pths```
  
```
mkdir pths
cd pths
mkdir backbone pretrain ft
cd ..
mv vgg16_bn-6c64b313.pth pths/backbone/
mv model_iter_50000.pth pths/pretrain/
mv model_iter_31600.pth pths/ft/
```

Here is an example:
```
.
├── CRAFT
│   ├── evaluate
│   ├── pths
│   └── sync_batchnorm
└── data
    ├── ICDAR2013
    ├── ICDAR2017
    └── SynthText
```
## Train
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py
```
## Finetune
```
CUDA_VISIBLE_DEVICES=0,1,2 python finetune.py
```
## Detect
```
CUDA_VISIBLE_DEVICES=0 python detect.py
```
## Evaluate
```
CUDA_VISIBLE_DEVICES=0 python eval.py
```
