import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import os
import cv2
import sys
import time 
import pickle
import sklearn
import argparse
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from scipy.ndimage import gaussian_filter


import dataloaders
import networks.resnet as resnet
import preprocess_feature_first.pretrain_vgg as pretrain_vgg
from config import ROOT

from ei import patch
patch(select=True)

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default="bottle")
parser.add_argument('--kmeans', type=int, default=128)
parser.add_argument('--type', type=str, default="all")
parser.add_argument('--index', type=int, default=30)
parser.add_argument('--image_size', type=int, default=1024)
parser.add_argument('--patch_size', type=int, default=64)
parser.add_argument('--dim_reduction', type=str, default='PCA')
args = parser.parse_args()


scratch_model = nn.Sequential(
    resnet.resnet50(pretrained=False, num_classes=args.kmeans)
)
scratch_model = nn.DataParallel(scratch_model).cuda()

### DataSet for all defect type
test_path = "{}/dataset/{}/test_resize/all/".format(ROOT, args.data)
test_dataset = dataloaders.MvtecLoader(test_path)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

test_good_path = "{}/dataset/{}/test_resize/good/".format(ROOT, args.data)
test_good_dataset = dataloaders.MvtecLoader(test_good_path)
test_good_loader = DataLoader(test_good_dataset, batch_size=1, shuffle=False)

mask_path = "{}/dataset/{}/ground_truth_resize/all/".format(ROOT, args.data)
mask_dataset = dataloaders.MaskLoader(mask_path)
mask_loader = DataLoader(mask_dataset, batch_size=1, shuffle=False)

## Models
pretrain_model = nn.DataParallel(pretrain_vgg.model).cuda()

## Clusters
kmeans_path = "{}/preprocessData/kmeans/{}/{}/vgg19_{}_100_128.pickle".format(ROOT, args.dim_reduction, args.data, args.kmeans)
kmeans = pickle.load(open(kmeans_path, "rb"))

pca_path = "{}/preprocessData/{}/{}/vgg19_{}_100_128.pickle".format(ROOT, args.dim_reduction, args.data, args.kmeans)
pca = pickle.load(open(pca_path, "rb"))

## Label
test_label_name = "{}/preprocessData/label/vgg19/{}/{}/test/all_{}_100.pth".format(ROOT, args.dim_reduction, args.data, args.kmeans)
test_label = torch.tensor(torch.load(test_label_name))

test_good_label_name = "{}/preprocessData/label/vgg19/{}/{}/test/good_{}_100.pth".format(ROOT, args.dim_reduction, args.data, args.kmeans)
test_good_label = torch.tensor(torch.load(test_good_label_name))

## Others
pil_to_tensor = transforms.ToTensor()

test_data = args.data
test_type = args.type




def norm(features):
    if features.max() == 0:
        return features
    else:
        return features / features.max()

def eval_feature(epoch, model, test_loader, test_label):
    global pretrain_model
    global kmeans

    model.eval()
    pretrain_model.eval()

    with torch.no_grad():
        img_feature = []
        total_gt = []
        total_idx = []
 
        start = time.time()

        for (idx, img) in test_loader:
            img = img.cuda()
            idx = idx[0].item()

            print(f'eval phase: img idx={idx}')

            value_feature = []
            value_label = []
            label_idx = []
            label_gt = []

            xs = []
            ys = []
            crop_list = []
            
            patches = []

            crop_output = pretrain_model(img)
            chunk_num = int(args.image_size / args.patch_size)

            for i in range(chunk_num):
                for j in range(chunk_num):
                    out_ = crop_output[0, :, i, j]
                    patches.append(out_.detach().cpu().numpy())

                    mask = torch.ones(1, 1, 1024, 1024)
                    mask[:, :, i*args.patch_size:i*args.patch_size+args.patch_size, j*args.patch_size:j*args.patch_size+args.patch_size] = 0
                    mask = mask.cuda()
                    x = img * mask
                    x = torch.cat((x, mask), 1)
                    label = test_label[idx][i*chunk_num+j].cuda()
                   
                    xs.append(x)
                    ys.append(label)

                if (len(xs) == chunk_num):
                    np_patches = np.array(patches)
                    np_patches = np_patches.reshape(-1, np_patches.shape[-1])

                    new_outs = pca.transform(np_patches)
                    for i in range(new_outs.shape[0]):
                        f = new_outs[i].reshape(1, -1)
                        f = pil_to_tensor(f).cuda()
                        f = torch.squeeze(f)

                        crop_list.append(f)

                    x = torch.cat(xs, 0)
                    y = torch.stack(ys).squeeze().cuda()
                    xs.clear()
                    ys.clear()
                    patches.clear()

                    output = model(x)
                    y_ = output.argmax(-1).detach().cpu().numpy()
                
                    for k in range(chunk_num):
                        label_idx.append(y_[k])
                        label_gt.append(y[k].item())
                        output_center = kmeans.cluster_centers_[y_[k]]
                        output_center = np.reshape(output_center, (1, -1))
                        output_center = pil_to_tensor(output_center).cuda()
                        output_center = torch.squeeze(output_center)

                        isWrongLabel = int(y_[k] != y[k].item())
                        diff = isWrongLabel * nn.MSELoss()(output_center, crop_list[k])
                        value_feature.append(diff.item())
                    crop_list.clear()
                    
            total_gt.append(label_gt)
            total_idx.append(label_idx)
            img_feature.append(value_feature)
        print("total running time: ", time.time() - start)

    img_feature = np.array(img_feature).reshape((len(test_loader), -1))
    total_gt = np.array(total_gt).reshape((len(test_loader), -1))
    total_idx = np.array(total_idx).reshape((len(test_loader), -1))

    return img_feature, total_gt, total_idx


start = time.time()
""" load model """
global_index = args.index
scratch_model.load_state_dict(torch.load('{}/models/vgg19/{}/exp1_{}_{}.ckpt'.format(ROOT, args.data, args.kmeans, global_index)))

print("------- For defect type -------")
value_feature, total_gt, total_idx = eval_feature(global_index, scratch_model, test_loader, test_label)
print("------- For good type -------")
value_good_feature, total_good_gt, total_good_idx = eval_feature(global_index, scratch_model, test_good_loader, test_good_label)

label_pred = []
label_gt = []

chunk_num = int(args.image_size / args.patch_size)
""" for defect type """ 
for ((idx, img), (idx2, img2)) in zip(test_loader, mask_loader):
    img = img.cuda()
    idx = idx[0].item()


    error_map = np.zeros((1024, 1024))
    for index, scalar in enumerate(value_feature[idx]):
        mask = np.zeros((1024, 1024))
        x = index // chunk_num
        y = index % chunk_num
        mask[x*args.patch_size:x*args.patch_size+args.patch_size, y*args.patch_size:y*args.patch_size+args.patch_size] = 1
        
        error_map += mask * scalar
    error_map = gaussian_filter(error_map, sigma=1)

    img_ = np.squeeze(img.detach().cpu().numpy()).transpose((1,2,0))
    defect_gt = np.squeeze(img2.cpu().numpy()).transpose((1,2,0))
    ironman_grid = plt.GridSpec(1, 3)
    fig = plt.figure(figsize=(18,6), dpi=100)
    ax1 = fig.add_subplot(ironman_grid[0,0])
    im1 = ax1.imshow(error_map, cmap="Blues")
    ax2 = fig.add_subplot(ironman_grid[0,1])
    ax3 = fig.add_subplot(ironman_grid[0,2])
    im2 = ax2.imshow(img_)
    im3 = ax3.imshow(defect_gt)


    for i in range(chunk_num):
        for j in range(chunk_num):
            ax1.text((j+0.2)*args.patch_size, (i+0.6)*args.patch_size, total_idx[idx][i*chunk_num+j], fontsize=10)
            ax2.text((j+0.2)*args.patch_size, (i+0.6)*args.patch_size, total_gt[idx][i*chunk_num+j], fontsize=10)


    ## 可以在這邊算
    defect_gt = np.squeeze(img2.cpu().numpy()).transpose(1,2,0)
    true_mask = defect_gt[:, :, 0].astype('int32')
    label_pred.append(error_map)
    label_gt.append(true_mask)    
    # print(f'EP={global_index} defect_img_idx={idx}')

    errorMapPath = "./testing/{}/all/{}/".format(test_data, args.kmeans)
    if not os.path.isdir(errorMapPath):
        os.makedirs(errorMapPath)
        print("----- create folder for type:{} -----".format(test_type))
    
    errorMapName = "{}_{}.png".format(
        str(idx),
        str(global_index)
    )

    plt.savefig(errorMapPath+errorMapName, dpi=100)
    plt.close(fig)


""" for good type """
for (idx, img) in test_good_loader:
    img = img.cuda()
    idx = idx[0].item()

    error_map = np.zeros((1024, 1024))
    for index, scalar in enumerate(value_good_feature[idx]):
        mask = np.zeros((1024, 1024))
        x = index // chunk_num
        y = index % chunk_num
        mask[x*args.patch_size:x*args.patch_size+args.patch_size, y*args.patch_size:y*args.patch_size+args.patch_size] = 1
        error_map += mask * scalar
    error_map = gaussian_filter(error_map, sigma=1)

    img_ = np.squeeze(img.detach().cpu().numpy()).transpose((1,2,0))
    ironman_grid = plt.GridSpec(1, 2)
    fig = plt.figure(figsize=(12,6), dpi=100)
    ax1 = fig.add_subplot(ironman_grid[0,0])
    im1 = ax1.imshow(error_map, cmap="Blues")
    ax2 = fig.add_subplot(ironman_grid[0,1])
    im2 = ax2.imshow(img_)

    
    for i in range(chunk_num):
        for j in range(chunk_num):
            ax1.text((j+0.2)*args.patch_size, (i+0.6)*args.patch_size, total_idx[idx][i*chunk_num+j], fontsize=10)
            ax2.text((j+0.2)*args.patch_size, (i+0.6)*args.patch_size, total_gt[idx][i*chunk_num+j], fontsize=10)

    defect_gt = np.zeros((1024, 1024, 3))
    true_mask = defect_gt[:, :, 0].astype('int32')
    label_pred.append(error_map)
    label_gt.append(true_mask)    
    # print(f'EP={global_index} good_img_idx={idx}')

    errorMapPath = "./testing/{}/good/{}/".format(test_data, args.kmeans)
    if not os.path.isdir(errorMapPath):
        os.makedirs(errorMapPath)
        print("----- create folder for type:{} -----".format(test_type))
    
    errorMapName = "{}_{}.png".format(
        str(idx),
        str(global_index)
    )

    plt.savefig(errorMapPath+errorMapName, dpi=100)
    plt.close(fig)

label_pred = norm(np.array(label_pred))
print('spend w/o auroc: ', time.time() - start)
auc = roc_auc_score(np.array(label_gt).flatten(), label_pred.flatten())

print("Single Map AUC score for testing data {}: {}".format(args.data, auc))
print('spend with auroc: ', time.time() - start)