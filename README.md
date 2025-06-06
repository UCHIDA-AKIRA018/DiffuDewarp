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

AnoClip: Download the dataset from [here](https://drive.google.com/drive/folders/1WZQrSfEOH0xigkGprWRFvbfFQP0cTboj?usp=drive_link) and place it with the same structure as other MVTec categories.

Model: Download the learned model from [here](https://drive.google.com/drive/folders/1WZQrSfEOH0xigkGprWRFvbfFQP0cTboj?usp=drive_link)

### Train
```
python train.py
```
### Test
```
python eval.py
```

