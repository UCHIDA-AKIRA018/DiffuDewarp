# DiffuDewarp
![overview](https://github.com/UCHIDA-AKIRA018/DiffuDewarp/blob/main/imgs/overview.png)
**Measuring distortion strength with Dewarping Diffusion Models in Anomaly Detection**

Akira Uchida, Satoshi Ikehata, Yuichi Yoshida, Ikuro Sato
## Get Started
### Environment
```
pip install -r requirements.txt
```
git clone [soft_splatting](https://github.com/sniklaus/softmax-splatting) and place it under DiffuDewarp.

### Datasets
MvTec-AD: Download the dataset from [here](https://www.mvtec.com/company/research/datasets/mvtec-ad/). Images of these foregrounds can be downloaded from [DiffusionAD](https://github.com/HuiZhang0812/DiffusionAD?tab=readme-ov-file).

AnoClip: Download the dataset from [here](https://drive.google.com/file/d/1L1T-kbSFYdPxp_bdjga239C8q3VoPcrW/view?usp=sharing) and place it with the same structure as other MVTec categories.

### Train
```
python train.py
```
### Test
```
python eval.py
```

