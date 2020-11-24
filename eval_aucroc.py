import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import os
import cv2
import sys
import pickle
import sklearn
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix



import resnet
import pretrain_vgg
import dataloaders
import argparse
import time 

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default="bottle")
parser.add_argument('--kmeans', type=int, default=128)
parser.add_argument('--type', type=str, default="all")
parser.add_argument('--index', type=int, default=30)
args = parser.parse_args()


scratch_model = nn.Sequential(
    resnet.resnet18(pretrained=False, num_classes=args.kmeans)
)
scratch_model = nn.DataParallel(scratch_model).cuda()

### DataSet for all defect type
test_path = "./dataset/{}/test_resize/all/".format(args.data)
test_dataset = dataloaders.MvtecLoader(test_path)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

test_good_path = "./dataset/{}/test_resize/good/".format(args.data)
test_good_dataset = dataloaders.MvtecLoader(test_good_path)
test_good_loader = DataLoader(test_good_dataset, batch_size=1, shuffle=False)

mask_path = "./dataset/{}/ground_truth_resize/all/".format(args.data)
mask_dataset = dataloaders.MaskLoader(mask_path)
mask_loader = DataLoader(mask_dataset, batch_size=1, shuffle=False)

## Models
pretrain_model = nn.DataParallel(pretrain_vgg.model).cuda()

## Clusters
kmeans_path = "./preprocessData/kmeans/{}/vgg19_{}_100_16.pickle".format(args.data, args.kmeans)
kmeans = pickle.load(open(kmeans_path, "rb"))

pca_path = "./preprocessData/PCA/{}/vgg19_{}_100_16.pickle".format(args.data, args.kmeans)
pca = pickle.load(open(pca_path, "rb"))

## Label
test_label_name = "./preprocessData/label/vgg19/{}/test/all_{}_100.pth".format(args.data, args.kmeans)
test_label = torch.tensor(torch.load(test_label_name))

test_good_label_name = "./preprocessData/label/vgg19/{}/test/good_{}_100.pth".format(args.data, args.kmeans)
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

def getOverlap(y_true, y_pred, threshold):
        y_pred[y_pred >= threshold] = 1
        y_pred[y_pred < threshold] = 0

        y_true.reshape((-1, 1024, 1024))
        y_pred.reshape((-1, 1024, 1024))

        y_pred = np.bitwise_and(y_pred, y_true)

        overlap_rate = 0

        for i in range( y_true.shape[0] ):
            overlap_rate += y_pred[i].sum() / y_true[i].sum()

        return overlap_rate / y_true.shape[0]

def pROC(y_true, y_pred):
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred)

    nearestIndex = np.argmin(abs(fpr - 0.3))

    thresholds = thresholds[:nearestIndex+1]
    fpr = norm(fpr[:nearestIndex+1])

    print(len(fpr))
    area = 0
    for index in range(1, nearestIndex+1):
        height = fpr[index] - fpr[index - 1]

        if height != 0:
            width1 = getOverlap(y_true, y_pred, thresholds[index])
            width2 = getOverlap(y_true, y_pred, thresholds[index - 1])

            area += (width1 + width2) * height / 2
        else:
            continue

    return area
    
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

            for i in range(16):
                for j in range(16):
                    crop_img = img[:, :, i*64:i*64+64, j*64:j*64+64].cuda()
                    crop_output = pretrain_model(crop_img)
                    """ flatten the dimension of H and W """
                    out_ = crop_output.flatten(1,2).flatten(1,2)
                    out = pca.transform(out_.detach().cpu().numpy())
                    out = pil_to_tensor(out).squeeze().cuda()
                    crop_list.append(out)

                    mask = torch.ones(1, 1, 1024, 1024)
                    mask[:, :, i*64:i*64+64, j*64:j*64+64] = 0
                    mask = mask.cuda()
                    x = img * mask
                    x = torch.cat((x, mask), 1)
                    label = test_label[idx][i*16+j].cuda()
                   
                    xs.append(x)
                    ys.append(label)

                if (len(xs) == 16):

                    x = torch.cat(xs, 0)
                    y = torch.stack(ys).squeeze().cuda()
                    xs.clear()
                    ys.clear()

                    output = model(x)
                    y_ = output.argmax(-1).detach().cpu().numpy()
                
                    for k in range(16):
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


""" load model """
global_index = args.index
scratch_model.load_state_dict(torch.load('./models/vgg19/{}/exp1_{}_{}.ckpt'.format(args.data, args.kmeans, global_index)))

print("------- For defect type -------")
value_feature, total_gt, total_idx = eval_feature(global_index, scratch_model, test_loader, test_label)
print("------- For good type -------")
value_good_feature, total_good_gt, total_good_idx = eval_feature(global_index, scratch_model, test_good_loader, test_good_label)

label_pred = []
label_gt = []

""" for defect type """ 
for ((idx, img), (idx2, img2)) in zip(test_loader, mask_loader):
    img = img.cuda()
    idx = idx[0].item()


    error_map = np.zeros((1024, 1024))
    for index, scalar in enumerate(value_feature[idx]):
        mask = cv2.imread('./dataset/big_mask/mask{}.png'.format(index), cv2.IMREAD_GRAYSCALE)
        mask = np.invert(mask)
        mask[mask==255]=1
        
        error_map += mask * scalar

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

    for i in range(16):
        for j in range(16):
            ax1.text((j+0.2)*64, (i+0.6)*64, total_idx[idx][i*16+j], fontsize=10)
            ax2.text((j+0.2)*64, (i+0.6)*64, total_gt[idx][i*16+j], fontsize=10)


    ## 可以在這邊算
    defect_gt = np.squeeze(img2.cpu().numpy()).transpose(1,2,0)
    true_mask = defect_gt[:, :, 0].astype('int32')
    label_pred.append(error_map)
    label_gt.append(true_mask)    
    print(f'EP={global_index} defect_img_idx={idx}')

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
        mask = cv2.imread('./dataset/big_mask/mask{}.png'.format(index), cv2.IMREAD_GRAYSCALE)
        mask = np.invert(mask)
        mask[mask==255]=1
        error_map += mask * scalar

    img_ = np.squeeze(img.detach().cpu().numpy()).transpose((1,2,0))
    ironman_grid = plt.GridSpec(1, 2)
    fig = plt.figure(figsize=(12,6), dpi=100)
    ax1 = fig.add_subplot(ironman_grid[0,0])
    im1 = ax1.imshow(error_map, cmap="Blues")
    ax2 = fig.add_subplot(ironman_grid[0,1])
    im2 = ax2.imshow(img_)

    
    for i in range(16):
        for j in range(16):
            ax1.text((j+0.2)*64, (i+0.6)*64, total_good_idx[idx][i*16+j], fontsize=10)
            ax2.text((j+0.2)*64, (i+0.6)*64, total_good_gt[idx][i*16+j], fontsize=10)

    defect_gt = np.zeros((1024, 1024, 3))
    true_mask = defect_gt[:, :, 0].astype('int32')
    label_pred.append(error_map)
    label_gt.append(true_mask)    
    print(f'EP={global_index} good_img_idx={idx}')

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
auc = roc_auc_score(np.array(label_gt).flatten(), label_pred.flatten())
auc_fpr30 = pROC(np.array(label_gt).flatten(), label_pred.flatten())
print("AUC score for testing data {}: {}".format(args.data, auc))
print("AUC FPR under 30 pecent score for testing data {}: {}".format(args.data, auc_fpr30))