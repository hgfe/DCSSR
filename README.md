# DCSSR

Pytorch implementation of "Disparity-Constrained Stereo Endoscopic Image Super-Resolution"

## Da Vinci Dataset

[Download](https://drive.google.com/drive/folders/1ov7tX916DrPRqrsaaKInR-jLtQHgVPAe?usp=sharing)

The da Vinci Dataset contains 6300 pairs of stereo laparoscopic images, divided into 5 folds. The size of the frame is 512 $\times$ 512.

## Training

### Requirements

pytorch    
numpy    
matplotlib    
scikit-image    
torchvision    
pandas   
tqdm    
PIL  

### Prepare training data 

1. Download the [da Vinci Dataset](https://drive.google.com/drive/folders/1ov7tX916DrPRqrsaaKInR-jLtQHgVPAe?usp=sharing) and put into `data/train/`.
2. Run `data/train/generate_trainset.m` to generate training patches.
3. Example training set has been uploaded to `data/train/Davinci_patches_fold1`.

### Training

Here is an example to train DCSSR.
```
# Training
python train.py --scale_factor 2 --batch_size 8 --fold 1234 --device cuda:0 --trainset_name Davinci
```

## Testing

### Prepare test data

Download the [da Vinci Dataset](https://drive.google.com/drive/folders/1ov7tX916DrPRqrsaaKInR-jLtQHgVPAe?usp=sharing) and put the testset (such as Davinci_fold1_test) into `data/test/`. Examples (10 pairs) have been uploaded to `data/test/Davinci_fold5_test`.

### Pretrained models

An example of pretrained model has been uploaded to this repo in `model/x2/'. All the pretrained models are provided [here](https://drive.google.com/drive/folders/1NoHq-AyZszmTM7x7Oznj_i9EtCqETBAI?usp=sharing).

### Test

Here is an example to test DCSSR.
```
# Test
python test.py --scale_factor 2 --fold 1234 --device cuda:0 --trainset_name Davinci --dataset Davinci_fold5_test
```

## Results of 5-fold Cross Validation

Please refer to [RESULT.md](https://github.com/hgfe/DCSSR/blob/main/RESULT.md) in this repo.

### da Vinci Dataset (x2)

PSNR
|  Fold   | 1 | 2 | 3 | 4 | 5 | avg |
|:----------:|:----:|:----:|:----:|:----:|:----:|:----:|
|bicubic  |37.7731±0.5440|36.9459±0.8474|37.0630±0.5632|37.7490±1.2220|38.2665±1.1444|37.5595±0.8642|
|SRCNN    |41.6352±0.5763|40.9319±0.9536|40.9069±0.5482|41.7738±1.1885|41.6496±1.2996|41.3795±0.9132|
|VDSR     |42.2607±0.5331|41.7304±0.8328|41.7068±0.5147|42.4065±1.0917|42.0082±1.3634|42.0225±0.8671|
|DRRN     |42.0806±0.5521|41.7680±0.8843|41.4216±0.5245|42.2336±1.1579|42.1917±1.3461|41.9391±0.8930|
|StereoSR |42.0639±0.5477|41.7154±0.8739|41.4031±0.5344|42.3086±1.1105|42.2135±1.3671|41.9409±0.8867|
|PASSR    |42.3387±0.5337|41.9255±0.8311|41.7217±0.5447|42.1344±1.0566|42.0639±1.2710|42.0368±0.8474|
|DCSSR    |42.3822±0.5206|41.9016±0.8178|41.9057±0.5393|42.4578±1.0976|42.2875±1.3292|42.1870±0.8609|
|a1       |42.3172±0.5424|41.7715±0.8287|41.8643±0.5396|42.3508±1.0962|42.1697±1.3731|42.0947±0.8760|
|a2       |42.3390±0.5249|41.8738±0.8215|41.9521±0.5400|42.3714±1.0981|42.2462±1.3595|42.1565±0.8688|
|a3       |42.2963±0.5258|41.8531±0.8065|41.7328±0.5470|42.3913±1.1074|42.2094±1.3685|42.0966±0.8710|
|a4       |42.3514±0.5546|41.8982±0.8884|41.7974±0.5387|42.3230±1.0860|42.2447±1.3167|42.1229±0.8769|
|a5       |42.3048±0.5383|41.8133±0.8504|41.7509±0.5390|42.3796±1.1005|42.2360±1.3340|42.0969±0.8724|
 

## Acknowledgement

The codes are based on [PASSRnet](https://github.com/The-Learning-And-Vision-Atelier-LAVA/PASSRnet). Please also follow their licenses. Thanks for their awesome works.
