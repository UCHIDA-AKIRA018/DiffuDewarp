from random import seed
import torch
import os
import json
import matplotlib.pyplot as plt
import numpy as np
from torch import optim
from torch.utils.data import DataLoader
from models.Recon_subnetwork import UNetModel, update_ema_params
from models.Seg_subnetwork import SegmentationSubNetwork
from tqdm import tqdm
import torch.nn as nn
from data.dataset_beta_thresh import MVTecTrainDataset,MVTecTestDataset
from math import exp
import torch.nn.functional as F
from models.DDPM import GaussianDiffusionModel, get_beta_schedule
from scipy.ndimage import gaussian_filter
from skimage.measure import label, regionprops
from sklearn.metrics import roc_auc_score,auc,average_precision_score
import pandas as pd
from collections import defaultdict
from torch.utils.tensorboard import SummaryWriter
import logging
import sys
from torchvision.transforms.functional import gaussian_blur

# def weights_init(m):
#         classname = m.__class__.__name__
#         if classname.find('Conv') != -1:
#             m.weight.data.normal_(0.0, 0.02)
#         elif classname.find('BatchNorm') != -1:
#             m.weight.data.normal_(1.0, 0.02)
#             m.bias.data.fill_(0)    

def defaultdict_from_json(jsonDict):
    func = lambda: defaultdict(str)
    dd = func()
    dd.update(jsonDict)
    return dd

class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=4, logits=False, reduce=True): 
        super(BinaryFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss

def train(training_dataset_loader, testing_dataset_loader, args, data_len,sub_class,class_type,device ):
    writer = SummaryWriter(log_dir="tensorboard/" + args["arg_num"] + "_" + args["subclass"])

    in_channels = args["channels"]
    if args["mode"] == "DiffusionAD":
        out_channels = 3
    elif args["mode"] == "DiffuDewarp":
        out_channels = 5

    unet_model = UNetModel(args['img_size'][0], args['base_channels'], channel_mults=args['channel_mults'], dropout=args[
                "dropout"], n_heads=args["num_heads"], n_head_channels=args["num_head_channels"],
            in_channels=in_channels, out_channels=out_channels
            ).to(device)


    betas = get_beta_schedule(args['T'], args['beta_schedule'])

    ddpm_sample =  GaussianDiffusionModel(
            args['img_size'], betas, loss_weight=args['loss_weight'], mode=args["mode"], img_channels=in_channels
            )

    seg_model=SegmentationSubNetwork(in_channels=6, out_channels=1).to(device)

    optimizer_ddpm = optim.Adam( unet_model.parameters(), lr=args['diffusion_lr'],weight_decay=args['weight_decay'])
    optimizer_seg = optim.Adam(seg_model.parameters(),lr=args['seg_lr'],weight_decay=args['weight_decay'])

    loss_focal = BinaryFocalLoss().to(device)
    loss_smL1= nn.SmoothL1Loss().to(device)
    
    scheduler_seg =optim.lr_scheduler.CosineAnnealingLR(optimizer_seg, T_max=10, eta_min=0, last_epoch=- 1, verbose=False)

    # 保存読み込み
    dir_path =f'{args["output_path"]}/model/diff-params-ARGS={args["arg_num"]}/{sub_class}'
    model_files = [file for file in os.listdir(dir_path) if file.startswith("params-") and file.endswith(".pt") and file != 'params-best.pt' and file != 'params-last.pt']
    start_epoch = 0
    best_image_auroc=0.0
    best_pixel_auroc=0.0
    best_epoch=0
    if model_files:
        # 最後に保存されたモデル
        largest_model_filename = max(model_files, key=lambda x: int(x.split("-")[1].split(".")[0]))
        largest_model_path = os.path.join(dir_path, largest_model_filename)
        loaded_last_model = torch.load(largest_model_path, map_location=device)
        unet_model.load_state_dict(loaded_last_model["unet_model_state_dict"])
        seg_model.load_state_dict(loaded_last_model["seg_model_state_dict"])
        start_epoch = loaded_last_model['n_epoch'] + 1
        optimizer_ddpm.load_state_dict(loaded_last_model["optimizer_ddpm_state_dict"])
        optimizer_seg.load_state_dict(loaded_last_model["optimizer_seg_state_dict"])
        scheduler_seg.load_state_dict(loaded_last_model["scheduler_seg_state_dict"])
        del loaded_last_model

        # 最も結果が良かったモデル
        if sub_class != 'clip':
            best_model_path = os.path.join(dir_path, "params-best.pt")
            loaded_best_model = torch.load(best_model_path, map_location=device)
            best_image_auroc= loaded_best_model['best_image_auroc']
            best_pixel_auroc= loaded_best_model['best_pixel_auroc']
            best_epoch = loaded_best_model['n_epoch']
            del loaded_best_model

            temp_image_auroc,temp_pixel_auroc= eval(testing_dataset_loader,args,unet_model,seg_model,data_len,sub_class,device)
    
    # if sub_class != 'clip':
    #     temp_image_auroc,temp_pixel_auroc= eval(testing_dataset_loader,args,unet_model,seg_model,data_len,sub_class,device)

    tqdm_epoch = range(start_epoch, args['EPOCHS'])
    for epoch in tqdm_epoch:
        unet_model.train()
        seg_model.train()
        train_loss = 0.0
        train_focal_loss=0.0
        train_smL1_loss = 0.0
        train_noise_loss = 0.0
        tbar = tqdm(training_dataset_loader)
        for i, sample in enumerate(tbar):
            image = sample['image'].to(device)
            aug_image=sample['augmented_image'].to(device)
            anomaly_mask = sample["anomaly_mask"].to(device)
            # goodなら0 合成なら1
            anomaly_label = sample["has_anomaly"].to(device).squeeze()

            another_image = sample["anomaly_img_augmented"].to(device)
            perlin_mask = sample["perlin_mask"].to(device)
            perlin_mask_sec = sample["perlin_mask_sec"].to(device)
            object_mask = sample["object_mask"].to(device)

            if args['mode']=="DiffusionAD":
                noise_loss, pred_x0, _, _, _ = ddpm_sample.norm_guided_one_step_denoising(unet_model, aug_image, anomaly_label,args)
                pred_mask = seg_model(torch.cat((aug_image, pred_x0), dim=1)) 
                focal_loss = loss_focal(pred_mask,anomaly_mask)
                smL1_loss = loss_smL1(pred_mask, anomaly_mask)
                loss = noise_loss + 5*focal_loss + smL1_loss
            elif args['mode']=="DiffuDewarp":
                noise_loss, aug_image, anomaly_mask, pred_x0 = ddpm_sample.new_warping_denoising_batch(unet_model, image, args, perlin_mask=perlin_mask, perlin_mask_sec=perlin_mask_sec, object_mask=object_mask, another_image=another_image)
                loss = noise_loss

            optimizer_ddpm.zero_grad()
            optimizer_seg.zero_grad()
            loss.backward()

            optimizer_ddpm.step()
            optimizer_seg.step()
            scheduler_seg.step()

            train_loss += loss.item()
            tbar.set_description('Epoch:%d, Train loss: %.3f' % (epoch, train_loss))

        if (epoch+1) % 50==0 and epoch > 0:
            if sub_class != 'clip':
                temp_image_auroc,temp_pixel_auroc= eval(testing_dataset_loader,args,unet_model,seg_model,data_len,sub_class,device)
                writer.add_scalar("eval: image_auroc", temp_image_auroc, epoch)     
                writer.add_scalar("eval: pixel_auroc", temp_pixel_auroc, epoch) 
                if(temp_image_auroc+temp_pixel_auroc>=best_image_auroc+best_pixel_auroc):
                    if temp_image_auroc>=best_image_auroc:
                        save(unet_model,seg_model, args=args,final='best',epoch=epoch,sub_class=sub_class,optimizer_ddpm=optimizer_ddpm,optimizer_seg=optimizer_seg,scheduler_seg=scheduler_seg,best_image_auroc=best_image_auroc,best_pixel_auroc=best_pixel_auroc)
                        best_image_auroc = temp_image_auroc
                        best_pixel_auroc = temp_pixel_auroc
                        best_epoch = epoch
                
        if (epoch+1) % 100==0 and epoch > 0:
            save(unet_model,seg_model, args=args,final=epoch,epoch=epoch,sub_class=sub_class,optimizer_ddpm=optimizer_ddpm,optimizer_seg=optimizer_seg,scheduler_seg=scheduler_seg)
    
    save(unet_model,seg_model, args=args,final='last',epoch=args['EPOCHS'],sub_class=sub_class,optimizer_ddpm=optimizer_ddpm,optimizer_seg=optimizer_seg,scheduler_seg=scheduler_seg)


    temp = {"classname":[sub_class],"Image-AUROC": [best_image_auroc],"Pixel-AUROC":[best_pixel_auroc],"epoch":best_epoch}
    df_class = pd.DataFrame(temp)
    df_class.to_csv(f"{args['output_path']}/metrics/ARGS={args['arg_num']}/{class_type}_image_pixel_auroc_train.csv", mode='a',header=False,index=False)
   
    

def eval(testing_dataset_loader,args,unet_model,seg_model,data_len,sub_class,device):
    unet_model.eval()
    seg_model.eval()
    os.makedirs(f'{args["output_path"]}/metrics/ARGS={args["arg_num"]}/{sub_class}/', exist_ok=True)
    in_channels = args["channels"]
    betas = get_beta_schedule(args['T'], args['beta_schedule'])

    ddpm_sample =  GaussianDiffusionModel(
            args['img_size'], betas, loss_weight=args['loss_weight'], mode=args["mode"], img_channels=in_channels
            )
    
    logging.info(f"data_len: {data_len}",)
    total_image_pred = np.array([])
    total_image_gt =np.array([])
    total_pixel_gt=np.array([])
    total_pixel_pred = np.array([])
    total_pixel_pred_flow = np.array([])
    tbar = tqdm(testing_dataset_loader)
    for i, sample in enumerate(tbar):
        image = sample["image"].to(device)
        target=sample['has_anomaly'].to(device)
        gt_mask = sample["mask"].to(device)
        # only clip
        object_mask = sample['object_mask'].to(device)

        if args['mode']=="DiffusionAD":
            normal_t_tensor = torch.tensor([args["eval_normal_t"]], device=image.device).repeat(image.shape[0])
            noiser_t_tensor = torch.tensor([args["eval_noisier_t"]], device=image.device).repeat(image.shape[0])
            loss, pred_x_0_condition, _, _, _, _, _ = ddpm_sample.norm_guided_one_step_denoising_eval(unet_model, image, normal_t_tensor,noiser_t_tensor,args)
            pred_mask = seg_model(torch.cat((image, pred_x_0_condition), dim=1)) 
        elif args['mode']=="DiffuDewarp":
            t_tensor = torch.tensor([args["T"]], device=image.device).repeat(image.shape[0])
            # TODO warping_denoising_ite_eval
            pred_x_0, _, _, flow_all, _, _ = ddpm_sample.warping_denoising_ite_eval(unet_model, image, t_tensor, args, object_mask)
            
            pred_mask = (image - pred_x_0).square()
            pred_mask =  gaussian_blur(pred_mask, (41, 41), 5) # PatchCore
            
            pred_mask_flow = torch.norm(flow_all, dim=1).unsqueeze(0).repeat(1,3,1,1)
            out_mask_flow = gaussian_blur(pred_mask_flow, (41, 41), 5)

        out_mask = pred_mask

        if args['mode']=="DiffusionAD":
            topk_out_mask = torch.flatten(out_mask[0], start_dim=1)
            flatten_gt_mask =gt_mask[0][[0],:,:].flatten().detach().cpu().numpy().astype(int)
        elif args['mode']=="DiffuDewarp":
            flow_weight = 1/2
            if args['anomaly_color']:
                image_weight = 1
            else:
                image_weight = 0
            topk_out_mask = torch.flatten(image_weight * out_mask[0] + flow_weight * out_mask_flow[0], start_dim=1)
            flatten_pred_mask_flow=out_mask_flow[0].flatten().detach().cpu().numpy()
            total_pixel_pred_flow=np.append(total_pixel_pred_flow,flatten_pred_mask_flow)
            flatten_gt_mask =gt_mask[0].flatten().detach().cpu().numpy().astype(int)
            
        topk_out_mask = torch.topk(topk_out_mask, 50, dim=1, largest=True)[0]
        image_score = torch.mean(topk_out_mask)
        
        total_image_pred=np.append(total_image_pred,image_score.detach().cpu().numpy())
        total_image_gt=np.append(total_image_gt,target[0].detach().cpu().numpy())

        flatten_pred_mask=out_mask[0].flatten().detach().cpu().numpy()
        total_pixel_pred=np.append(total_pixel_pred,flatten_pred_mask)
        total_pixel_gt=np.append(total_pixel_gt,flatten_gt_mask)
        
    logging.info(sub_class)
    auroc_image = round(roc_auc_score(total_image_gt,total_image_pred),3)*100
    logging.info(f"Image AUC-ROC: {auroc_image}")
    
    if args['mode']=="DiffusionAD":
        auroc_pixel =  round(roc_auc_score(total_pixel_gt, total_pixel_pred),3)*100
    elif args['mode']=="DiffuDewarp":
        auroc_pixel = round(roc_auc_score(total_pixel_gt, image_weight * total_pixel_pred + flow_weight*total_pixel_pred_flow),3)*100

    logging.info(f"Pixel AUC-ROC: {auroc_pixel}")
   
    return auroc_image,auroc_pixel


def save(unet_model,seg_model, args,final,epoch,sub_class,optimizer_ddpm,optimizer_seg,scheduler_seg,best_image_auroc=None,best_pixel_auroc=None):
    dir_path =f'{args["output_path"]}/model/diff-params-ARGS={args["arg_num"]}/{sub_class}'
    model_files = [file for file in os.listdir(dir_path) if file.startswith("params-") and file.endswith(".pt") and file != 'params-best.pt']
    
    if final=='last':
        torch.save(
            {
                'n_epoch':              epoch,
                'unet_model_state_dict': unet_model.state_dict(),
                'seg_model_state_dict':  seg_model.state_dict(),
                'optimizer_ddpm_state_dict': optimizer_ddpm.state_dict(),
                'optimizer_seg_state_dict': optimizer_seg.state_dict(),
                'scheduler_seg_state_dict': scheduler_seg.state_dict(),
                'best_image_auroc': best_image_auroc,
                'best_pixel_auroc': best_pixel_auroc,
                "args":                 args
                }, f'{args["output_path"]}/model/diff-params-ARGS={args["arg_num"]}/{sub_class}/params-{final}.pt'
            )
    
    else:
        torch.save(
                {
                    'n_epoch':              epoch,
                    'unet_model_state_dict': unet_model.state_dict(),
                    'seg_model_state_dict':  seg_model.state_dict(),
                    'optimizer_ddpm_state_dict': optimizer_ddpm.state_dict(),
                    'optimizer_seg_state_dict': optimizer_seg.state_dict(),
                    'scheduler_seg_state_dict': scheduler_seg.state_dict(),
                    'best_image_auroc': best_image_auroc,
                    'best_pixel_auroc': best_pixel_auroc,
                    "args":                 args
                    }, f'{args["output_path"]}/model/diff-params-ARGS={args["arg_num"]}/{sub_class}/params-{final}.pt'
                )

    # いらないファイル削除
    # threshold = epoch - 300
    # for filename in model_files:
    #     # "params-" と ".pt" の間の数字を取得
    #     num_str = filename.split("-")[1].split(".")[0]
    #     try:
    #         num = int(num_str)
    #     except ValueError:
    #         continue
        
    #     if num < threshold and (num+1) % 500 !=0:
    #         os.remove(f'{dir_path}/{filename}')
    
    

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # read file from argument
    if len(sys.argv[1:]) > 0:
        files = sys.argv[1:]
    else:
        raise ValueError("Missing file argument")
    # read file from argument
    file = files[0]
    file = f"args{file}.json"
    # load the json args
    with open(f'./args/{file}', 'r') as f:
        args = json.load(f)
    args['arg_num'] = file[4:-5]
    args = defaultdict_from_json(args)

    if 'anomaly_color' not in args:
        args['anomaly_color'] = False

    mvtec_classes = ['carpet', 'grid', 'leather', 'tile', 'wood', 'bottle', 'cable', 'capsule', 'hazelnut', 'metal_nut', 'pill', 'screw',
            'toothbrush', 'transistor', 'zipper']
    
    if args['subclass'] == "all":
        current_classes = mvtec_classes
    else:
        current_classes = [args['subclass']]

    class_type='MVTec'
    for sub_class in current_classes:    
        logging.info(f"class: {sub_class}")
        subclass_path = os.path.join(args["mvtec_root_path"],sub_class)
        if sub_class != 'crip':
            training_dataset = MVTecTrainDataset(
                subclass_path,sub_class,img_size=args["img_size"],args=args
                )
            testing_dataset = MVTecTestDataset(
                subclass_path,sub_class,img_size=args["img_size"],
                )
        else:
            training_dataset = MVTecTrainDataset(
                subclass_path,sub_class,img_size=[1300,1300],args=args
                )
            testing_dataset = MVTecTestDataset(
                subclass_path,sub_class,img_size=[1300,1300],
                )
        
        logging.info(args)     
        logging.info(file)     

        data_len = len(testing_dataset) 
        training_dataset_loader = DataLoader(training_dataset, batch_size=args['Batch_Size'],shuffle=True,num_workers=8,pin_memory=True,drop_last=True)
        test_loader = DataLoader(testing_dataset, batch_size=1,shuffle=False, num_workers=4)

        # make arg specific directories
        for i in [f'{args["output_path"]}/model/diff-params-ARGS={args["arg_num"]}/{sub_class}',
                f'{args["output_path"]}/diffusion-training-images/ARGS={args["arg_num"]}/{sub_class}',
                 f'{args["output_path"]}/metrics/ARGS={args["arg_num"]}/{sub_class}']:
            try:
                os.makedirs(i)
            except OSError:
                pass

    
        train(training_dataset_loader, test_loader, args, data_len,sub_class,class_type,device )

if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s:%(name)s - %(message)s",
        filename="./log/train.log"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(device)

    seed(42)
    main()
