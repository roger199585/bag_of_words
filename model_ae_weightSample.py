# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
import torchvision.models as models

import os
import sys
import cv2
import time
import pickle
import random
import argparse
import itertools
import numpy as np
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt

# customize
import dataloaders
import networks.resnet as resnet
import networks.autoencoder as autoencoder

from config import ROOT, RESULT_PATH

# evaluations
from sklearn.metrics import roc_auc_score


# from config import gamma

""" set parameters """
parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='bottle')
parser.add_argument('--kmeans', type=int, default=128)
parser.add_argument('--type', type=str, default='good')
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--epoch', type=int, default=10)
parser.add_argument('--train_batch', type=int, default=16)
parser.add_argument('--test_batch_size', type=int, default=64)
parser.add_argument('--with_mask', type=str, default='True')
parser.add_argument('--patch_size', type=int, default=64)
parser.add_argument('--image_size', type=int, default=1024)
parser.add_argument('--resolution', type=int, default=4)
args = parser.parse_args()

MAXAUCEPOCH = 0
ALLAUC = []

kmeans_path = f"{ ROOT }/preprocessData/kmeans/AE/{ args.data }/{ args.resolution}/AE_{ args.kmeans }.pickle"
left_i_path = f"{ ROOT }/preprocessData/coordinate/AE/{ args.data }/{ args.resolution }/left_i.pickle"
left_j_path = f"{ ROOT }/preprocessData/coordinate/AE/{ args.data }/{ args.resolution }/left_j.pickle"

kmeans = pickle.load(open(kmeans_path, "rb"))

""" image transform """
pil_to_tensor = transforms.ToTensor()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#-------------------------------------

scratch_model = nn.Sequential(
    resnet.resnet50(pretrained=False, num_classes=args.kmeans),
)

""" training """
train_path = f"{ ROOT }/dataset/{ args.data }/train_resize/good"
label_name = f"{ ROOT }/preprocessData/label/fullPatch/AE/{ args.data }/{ args.resolution }/kmeans_{ args.kmeans }.pth"
mask_path  = f"{ ROOT }/dataset/big_mask/"

""" Load part of pretrained model """
pretrain_model =  autoencoder.autoencoder(3, args.resolution)
pretrain_model.load_state_dict(torch.load(f"{ ROOT }/models/AE/{ args.data }_{ args.resolution }/40000.ckpt"))
pretrain_model = nn.DataParallel(pretrain_model).to(device)

print('training data: ', train_path)
print('training label: ', label_name)

""" testing """
if (args.type == 'good'):
    test_path           = f"{ ROOT }/dataset/{ args.data }/test_resize/good"
    test_label_name     = f"{ ROOT }/preprocessData/label/AE/{ args.data }/{ args.resolution }/test/good_{ args.kmeans }.pth"
    all_test_label_name = f"{ ROOT }/preprocessData/label/AE/{ args.data }/{ args.resolution }/test/all_{ args.kmeans}.pth"
else:
    test_path           = f"{ ROOT }/dataset/{ args.data }/test_resize/{ args.type }"
    defect_gt_path      = f"{ ROOT }/dataset/{ args.data }/ground_truth_resize/{ args.type }"
    test_label_name     = f"{ ROOT }/preprocessData/label/AE/{ args.data }/{ args.resolution }/test/{ args.type }_{ args.kmeans }.pth"


test_label = torch.tensor(torch.load(test_label_name))
all_test_label = torch.tensor(torch.load(all_test_label_name))
print(test_label.shape)
print(all_test_label.shape)

""" eval """
eval_path      = f"{ ROOT }/dataset/{ args.data }/test_resize/all"
eval_mask_path = f"{ ROOT }/dataset/{ args.data }/ground_truth_resize/all/"

print('testing data: ', test_path)
print('testing label: ', test_label_name)

eval_fea_count = 0

def myNorm(features):
    return features / features.max()

def eval_with_origin_feature(model, test_loader, test_data, global_index, good=False):
    global cluster_features
    global pretrain_model
    global kmeans

    model.eval()
    pretrain_model.eval()

    with torch.no_grad():
        img_feature = []
        
        start = time.time()

        for (idx, img) in test_loader:
            each_pixel_err_sum = np.zeros([1024, 1024])
            each_pixel_err_count = np.zeros([1024, 1024])

            # pixel_feature = []  
            img = img.to(device)
            idx = idx[0].item()
            
            print(f'eval phase: img idx={idx}')

            chunk_num = int(args.image_size / args.patch_size)
            """ slide window = 16 """
            map_num = int((args.image_size - args.patch_size) / chunk_num + 1)   ## = 61
            indices = list(itertools.product(range(map_num), range(map_num)))
            
            """ batch """
            batch_size = 16

            label_idx = []
            label_gt = []

            for batch_start_idx in range(0, len(indices), batch_size):
                xs = []
                ys = []
                crop_list = []

                batch_idxs = indices[batch_start_idx:batch_start_idx+batch_size]

                for i, j in batch_idxs:
                    crop_img = img[:, :, i*chunk_num:i*chunk_num+args.patch_size, j*chunk_num:j*chunk_num+args.patch_size].to(device)
                    _, latent_code = pretrain_model(crop_img)
                    """ flatten the dimension of H and W """
                    out = latent_code.flatten(1,2).flatten(1,2)
                    out_ = out.detach().cpu().numpy()
                    out_label = kmeans.predict(out_)
                    out_label = torch.from_numpy(out_label).to(device)
                    crop_list.append(out)

                    mask = torch.ones(1, 1, 1024, 1024)
                    mask[:, :, i*chunk_num:i*chunk_num+args.patch_size, j*chunk_num:j*chunk_num+args.patch_size] = 0
                    mask = mask.to(device)
                    x = img * mask if args.with_mask == 'True' else img
                    x = torch.cat((x, mask), 1)

                    xs.append(x)
                    ys.append(out_label)

                x = torch.cat(xs, 0)
                y = torch.stack(ys).squeeze().to(device)                        
                output = model(x)
                y_ = output.argmax(-1).detach().cpu().numpy()

                for n, (i, j) in enumerate(batch_idxs):
                    output_feature = np.expand_dims(cluster_features[y_[n]], axis=0)
                    output_feature = torch.from_numpy(output_feature).cuda()

                    isWrongLabel = int(y_[n] != y[n].item())
                    diff = isWrongLabel * nn.MSELoss()(output_feature, crop_list[n])
                    
                    each_pixel_err_sum[i*chunk_num:i*chunk_num+args.patch_size, j*chunk_num:j*chunk_num+args.patch_size] += diff.item()
                    each_pixel_err_count[i*chunk_num:i*chunk_num+args.patch_size, j*chunk_num:j*chunk_num+args.patch_size] += 1

            pixel_feature = each_pixel_err_sum / each_pixel_err_count

            img_feature.append(pixel_feature)

    print(np.array(img_feature).shape)
    img_feature = np.array(img_feature).reshape((len(test_loader), -1))
    return img_feature

def eval_feature_for_multiMap(model, test_loader, test_data, global_index, good=False):
    global pretrain_model
    global kmeans

    model.eval()
    pretrain_model.eval()

    with torch.no_grad():
        img_feature = []
        
        start = time.time()

        for (idx, img) in test_loader:
            each_pixel_err_sum = np.zeros([1024, 1024])
            each_pixel_err_count = np.zeros([1024, 1024])
 
            img = img.to(device)
            idx = idx[0].item()
            
            print(f'eval phase: img idx={idx}')

            chunk_num = int(args.image_size / args.patch_size)
            """ slide window = 16 """
            map_num = int((args.image_size - args.patch_size) / chunk_num + 1)   ## = 61
            indices = list(itertools.product(range(map_num), range(map_num)))
            
            """ batch """
            batch_size = 32

            label_idx = []
            label_gt = []

            for batch_start_idx in range(0, len(indices), batch_size):
                xs = []
                ys = []
                crop_list = []

                batch_idxs = indices[batch_start_idx:batch_start_idx+batch_size]

                for i, j in batch_idxs:
                    crop_img = img[:, :, i*chunk_num:i*chunk_num+args.patch_size, j*chunk_num:j*chunk_num+args.patch_size].to(device)
                    _, latent_code = pretrain_model(crop_img)
                    """ flatten the dimension of H and W """
                    out_ = latent_code.flatten(1,2).flatten(1,2)
                    out = out_.detach().cpu().numpy()
                    out_label = kmeans.predict(out)
                    out_label = torch.from_numpy(out_label).to(device)
                    out = pil_to_tensor(out).squeeze().to(device)
                    crop_list.append(out)

                    mask = torch.ones(1, 1, 1024, 1024)
                    mask[:, :, i*chunk_num:i*chunk_num+args.patch_size, j*chunk_num:j*chunk_num+args.patch_size] = 0
                    mask = mask.to(device)
                    x = img * mask if args.with_mask == 'True' else img
                    x = torch.cat((x, mask), 1)

                    xs.append(x)
                    ys.append(out_label)

                x = torch.cat(xs, 0)
                y = torch.stack(ys).squeeze().to(device)                        
                output = model(x)
                y_ = output.argmax(-1).detach().cpu().numpy()

                for n, (i, j) in enumerate(batch_idxs):
                    output_center = kmeans.cluster_centers_[y_[n]]
                    output_center = np.reshape(output_center, (1, -1))
                    output_center = pil_to_tensor(output_center).to(device)
                    output_center = torch.squeeze(output_center)

                    isWrongLabel = int(y_[n] != y[n].item())
                    diff = isWrongLabel * nn.MSELoss()(output_center, crop_list[n])
                    
                    each_pixel_err_sum[i*chunk_num:i*chunk_num+args.patch_size, j*chunk_num:j*chunk_num+args.patch_size] += diff.item()
                    each_pixel_err_count[i*chunk_num:i*chunk_num+args.patch_size, j*chunk_num:j*chunk_num+args.patch_size] += 1
            
            pixel_feature = each_pixel_err_sum / each_pixel_err_count
            img_feature.append(pixel_feature)

    print(np.array(img_feature).shape)
    img_feature = np.array(img_feature).reshape((len(test_loader), -1))
    return img_feature

def eval_feature(epoch, model, test_loader, __labels, isGood):
    global eval_fea_count
    global pretrain_model
    global kmeans

    model.eval()
    pretrain_model.eval()

    with torch.no_grad():
        img_feature = []

        for (idx, img) in test_loader:
            img = img.to(device)
            idx = idx[0].item()

            print(f'eval phase: img idx={idx}')

            value_feature = []
            value_label = []
            label_idx = []
            label_gt = []

            xs = []
            ys = []
            crop_list = []
            origin_feature_list = []

            chunk_num = int(args.image_size / args.patch_size)
            for i in range(chunk_num):
                for j in range(chunk_num):
                    crop_img = img[:, :, i*args.patch_size:i*args.patch_size+args.patch_size, j*args.patch_size:j*args.patch_size+args.patch_size].to(device)
                    _, latent_code = pretrain_model(crop_img)
                    """ flatten the dimension of H and W """
                    out_ = latent_code.flatten(1,2).flatten(1,2)
                    out = out_.detach().cpu().numpy()
                    out = pil_to_tensor(out).squeeze().to(device)
                    crop_list.append(out)
                    origin_feature_list.append(out_)

                    mask = torch.ones(1, 1, 1024, 1024)
                    mask[:, :, i*args.patch_size:i*args.patch_size+args.patch_size, j*args.patch_size:j*args.patch_size+args.patch_size] = 0
                    mask = mask.to(device)
                    x = img * mask if args.with_mask == 'True' else img
                    x = torch.cat((x, mask), 1)
                    label = __labels[idx][i*chunk_num+j].to(device)
                   
                    xs.append(x)
                    ys.append(label)

                if (len(xs) == args.test_batch_size):
                    x = torch.cat(xs, 0)
                    y = torch.stack(ys).squeeze().to(device)
                    xs.clear()
                    ys.clear()

                    output = model(x)
                    y_ = output.argmax(-1).detach().cpu().numpy()
                    acc = (output.argmax(-1).detach() == y).float().mean()  

                    for k in range(args.test_batch_size):
                        label_idx.append(y_[k])
                        label_gt.append(y[k].item())
                        output_center = kmeans.cluster_centers_[y_[k]]
                        output_center = np.reshape(output_center, (1, -1))
                        output_center = pil_to_tensor(output_center).to(device)
                        output_center = torch.squeeze(output_center)

                        isWrongLabel = int(y_[k] != y[k].item())

                        un_out = torch.unsqueeze(output[k], dim=0)
                        un_y = torch.unsqueeze(y[k], dim=0).long()
                        diff_label = nn.CrossEntropyLoss()(un_out, un_y)
                        diff = isWrongLabel * nn.MSELoss()(output_center, crop_list[k])
                        value_feature.append(diff.item())

                        output_feature = np.expand_dims(cluster_features[y_[k]], axis=0)
                        output_feature = torch.from_numpy(output_feature).cuda()

                        isWrongLabel = int(y_[k] != y[k].item())
                        origin_feature_diff = isWrongLabel * nn.MSELoss()(output_feature, origin_feature_list[k])
                    

                        if isGood:
                            writer.add_scalar('test_feature_loss', diff.item(), eval_fea_count)
                            writer.add_scalar('test_origin_feature_loss', origin_feature_diff.item(), eval_fea_count)
                            writer.add_scalar('test_label_loss', diff_label.item(), eval_fea_count)
                            writer.add_scalar('test_label_acc', acc.item(), eval_fea_count)
                            eval_fea_count += 1

                    crop_list.clear()
                    origin_feature_list.clear()

            img_feature.append(value_feature)
    print(np.array(img_feature).shape)
    print(len(test_loader))
    img_feature = np.array(img_feature).reshape((len(test_loader), -1))

    return img_feature

if __name__ == "__main__":
    """ Summary Writer """
    writer = SummaryWriter(log_dir=f"{ RESULT_PATH }/pretrainWithAE_mask_{ args.with_mask }_patch_{ args.patch_size }_{ args.data }_{ args.type }_kmeans_{ args.kmeans }_{ datetime.now() }")

    """ weight sampling with noise patch in training data """
    train_dataset = dataloaders.NoisePatchDataloader(train_path, label_name, left_i_path, left_j_path)
    samples_weights = torch.from_numpy(train_dataset.samples_weights)
    sampler = WeightedRandomSampler(samples_weights.type('torch.DoubleTensor'), len(samples_weights))
    train_loader = DataLoader(train_dataset, batch_size=args.train_batch, num_workers=1, sampler=sampler)

    # testing set (normal data)
    test_dataset = dataloaders.MvtecLoader(test_path)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    # testing set (defect data)
    eval_dataset = dataloaders.MvtecLoader(eval_path)
    eval_loader = DataLoader(eval_dataset, batch_size=1, shuffle=False)
    eval_mask_dataset = dataloaders.MaskLoader(eval_mask_path)
    eval_mask_loader = DataLoader(eval_mask_dataset, batch_size=1, shuffle=False)

    ## Cluster Center Features
    center_features_path = f"{ ROOT }/preprocessData/cluster_center/AE/{ args.data }/{ args.resolution }/{ args.kmeans }.pickle"
    cluster_features = pickle.load(open(center_features_path, "rb"))

    scratch_model = nn.DataParallel(scratch_model).to(device)
    epoch_num = 0

    """ training config """ 
    # criterion = nn.MSELoss()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(scratch_model.parameters(), lr = args.lr)
    
    iter_count = 1
    
    for epoch in range(args.epoch): 
        """ noise version 2 """
        print("------- For defect type -------")
        value_feature = eval_feature(epoch, scratch_model, eval_loader, all_test_label, isGood=False)
        print("------- For good type -------")
        value_good_feature = eval_feature(epoch, scratch_model, test_loader, test_label, isGood=True)

        label_pred = []
        label_gt = []

        """ for defect type """ 
        for ((idx, img), (idx2, img2)) in zip(eval_loader, eval_mask_loader):
            img = img.cuda()
            idx = idx[0].item()

            error_map = np.zeros((1024, 1024))
            for index, scalar in enumerate(value_feature[idx]):
                mask = cv2.imread(f"{ ROOT }/dataset/big_mask/mask{ index }.png", cv2.IMREAD_GRAYSCALE)
                mask = np.invert(mask)
                mask[mask==255]=1
                
                error_map += mask * scalar

            ## 可以在這邊算
            defect_gt = np.squeeze(img2.cpu().numpy()).transpose(1,2,0)
            true_mask = defect_gt[:, :, 0].astype('int32')
            label_pred.append(error_map)
            label_gt.append(true_mask)    
            print(f'EP={epoch} defect_img_idx={idx}')


        """ for good type """
        for (idx, img) in test_loader:
            img = img.cuda()
            idx = idx[0].item()

            error_map = np.zeros((1024, 1024))
            for index, scalar in enumerate(value_good_feature[idx]):
                mask = cv2.imread(f"{ ROOT }/dataset/big_mask/mask{ index }.png", cv2.IMREAD_GRAYSCALE)
                mask = np.invert(mask)
                mask[mask==255]=1
                error_map += mask * scalar

            defect_gt = np.zeros((1024, 1024, 3))
            true_mask = defect_gt[:, :, 0].astype('int32')
            label_pred.append(error_map)
            label_gt.append(true_mask)    
            print(f'EP={epoch} good_img_idx={idx}')

        label_pred = myNorm(np.array(label_pred))
        auc = roc_auc_score(np.array(label_gt).flatten(), label_pred.flatten())
        ALLAUC.append(auc)

        if auc >= ALLAUC[MAXAUCEPOCH]:
            MAXAUCEPOCH = epoch

        writer.add_scalar('roc_auc_score', auc, epoch)
        print("AUC score for testing data {}: {}".format(auc, args.data))
        
        for (idx, img, left_i, left_j, label, mask) in train_loader:
            scratch_model.train()
            idx = idx[0].item()
            
            img = img.to(device)
            mask = mask.to(device)

            x = img * mask if args.with_mask == 'True' else img
            x = torch.cat((x, mask), 1)
            label = label.squeeze().to(device, dtype=torch.long)

            output = scratch_model(x)
            
            loss = criterion(output, label)
            acc = (output.argmax(-1).detach() == label).float().mean()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            writer.add_scalar('loss', loss.item(), iter_count)
            writer.add_scalar('acc', acc.item(), iter_count)            
            
            print(f'Training EP={epoch+epoch_num} it={iter_count} loss={loss.item()}')
            
            iter_count += 1

            if iter_count % 1000 == 0:
                value_good_feature = eval_feature(epoch, scratch_model, test_loader, test_label, isGood=True)
        
        if not os.path.isdir(f"{ ROOT }/models/AE/{ args.data }/{ args.resolution}"):
            os.makedirs(f"{ ROOT }/models/AE/{ args.data }/{ args.resolution}")
        
        path = f"{ ROOT }/models/AE/{ args.data }/{ args.resolution }/exp_{ args.kmeans }_{ str(epoch+1+epoch_num) }.ckpt"
        torch.save(scratch_model.state_dict(), path)

    try:
        global_index = MAXAUCEPOCH
        scratch_model.load_state_dict(torch.load(f"{ ROOT }/models/AE/{ args.data }/{ args.resolution }/exp_{ args.kmeans }_{ str(epoch+1+epoch_num) }.ckpt"))
        
        """ 透過 pca 將為之後的 cluster center 去算 feature error 來畫圖"""
        print("----- defect -----")
        img_all_feature = eval_feature_for_multiMap(scratch_model, eval_loader, args.data, global_index, good=False)
        img_all_origin_feature = eval_with_origin_feature(scratch_model, eval_loader, args.data, global_index, good=False)
        print("----- good -----")
        img_good_feature = eval_feature_for_multiMap(scratch_model, test_loader, args.data, global_index, good=True)
        img_good_origin_feature = eval_with_orrigin_feature(scratch_model, test_loader, args.data, global_index, good=True)

        label_pred = []
        label_true = []

        """ for defect type """ 
        for ((idx, img), (idx2, img2)) in zip(eval_loader, eval_mask_loader):
            img = img.cuda()
            idx = idx[0].item()

            errorMap = img_all_feature[idx].reshape((1024, 1024))
            """  draw errorMap """
            img_ = np.squeeze(img.detach().cpu().numpy()).transpose((1,2,0))
            defect_gt = np.squeeze(img2.cpu().numpy()).transpose((1,2,0))
            ironman_grid = plt.GridSpec(1,3)
            fig = plt.figure(figsize=(18, 6), dpi=100)
            ax1 = fig.add_subplot(ironman_grid[0,1])
            ax1.set_axis_off()
            im1 = ax1.imshow(errorMap, cmap="Blues")
            ax2 = fig.add_subplot(ironman_grid[0,0])
            ax2.set_axis_off()
            ax3 = fig.add_subplot(ironman_grid[0,2])
            ax3.set_axis_off()
            im2 = ax2.imshow(img_)
            im3 = ax3.imshow(defect_gt)


            errorMapPath = f"{ ROOT }/Results/testing_multiMap/{ args.data }/all/{ args.kmeans }/ae_map/"
            if not os.path.isdir(errorMapPath):
                os.makedirs(errorMapPath)
                print(f"----- create folder for { args.data } | type: all -----")
            errorMapName = f"{ str(idx) }_{ str(global_index) }.png"

            plt.savefig(errorMapPath+errorMapName, dpi=100)
            plt.close(fig)

            """ for computing aucroc score """
            defect_gt = np.squeeze(img2.cpu().numpy()).transpose(1,2,0)
            true_mask = defect_gt[:, :, 0].astype('int32')
            label_pred.append(errorMap)
            label_true.append(true_mask)    
            print(f'EP={ global_index } defect_img_idx={ idx }')

        """ for good type """
        for (idx, img) in test_loader:
            img = img.cuda()
            idx = idx[0].item()
            
            errorMap = img_good_feature[idx].reshape((1024, 1024))
            
            """ draw errorMap """
            img_ = np.squeeze(img.detach().cpu().numpy()).transpose((1,2,0))
            ironman_grid = plt.GridSpec(1, 2)
            fig = plt.figure(figsize=(12,6), dpi=100)
            ax1 = fig.add_subplot(ironman_grid[0,0])
            ax1.set_axis_off()
            im1 = ax1.imshow(errorMap, cmap="Blues")
            ax2 = fig.add_subplot(ironman_grid[0,1])
            ax2.set_axis_off()
            im2 = ax2.imshow(img_)
            
            errorMapPath = f"{ ROOT }/Results/testing_multiMap/{ args.data }/good/{ args.kmeans }/ae_map/"
            if not os.path.isdir(errorMapPath):
                os.makedirs(errorMapPath)
                print(f"----- create folder for { args.data } | type: good -----")
            errorMapName = f"{ str(idx) }_{ str(global_index) }.png"

            plt.axis('off')
            plt.savefig(errorMapPath+errorMapName, dpi=100)
            plt.close(fig)

            """ for computing aucroc score """
            defect_gt = np.zeros((1024, 1024, 3))
            true_mask = defect_gt[:, :, 0].astype('int32')
            label_pred.append(errorMap)
            label_true.append(true_mask)    
            print(f'EP={ global_index } good_img_idx={ idx }')

        label_pred = myNorm(np.array(label_pred))
        auc = roc_auc_score(np.array(label_true).flatten(), label_pred.flatten())

        f = open("overlap_score.txt", "a")
        f.write("AUC score for testing data {} with pca feature: {}".format(args.data, auc))
        f.close()
        
        print("AUC score for testing data {} with pca feature: {}".format(args.data, auc))


        label_pred = []
        label_true = []

        """ for defect type """ 
        for ((idx, img), (idx2, img2)) in zip(eval_loader, eval_mask_loader):
            img = img.cuda()
            idx = idx[0].item()

            errorMap = img_all_origin_feature[idx].reshape((1024, 1024))
            """  draw errorMap """
            img_ = np.squeeze(img.detach().cpu().numpy()).transpose((1,2,0))
            defect_gt = np.squeeze(img2.cpu().numpy()).transpose((1,2,0))
            ironman_grid = plt.GridSpec(1,3)
            fig = plt.figure(figsize=(18, 6), dpi=100)
            ax1 = fig.add_subplot(ironman_grid[0,1])
            ax1.set_axis_off()
            im1 = ax1.imshow(errorMap, cmap="Blues")
            ax2 = fig.add_subplot(ironman_grid[0,0])
            ax2.set_axis_off()
            ax3 = fig.add_subplot(ironman_grid[0,2])
            ax3.set_axis_off()
            im2 = ax2.imshow(img_)
            im3 = ax3.imshow(defect_gt)


            errorMapPath = "testing_multiMap/{}/all/{}/origin_map/".format(test_data, args.kmeans)
            if not os.path.isdir(errorMapPath):
                os.makedirs(errorMapPath)
                print("----- create folder for {} | type: all -----".format(test_data))
            
            errorMapName = "{}_{}.png".format(
                str(idx),
                str(global_index)
            )

            plt.savefig(errorMapPath+errorMapName, dpi=100)
            plt.close(fig)

            """ for computing aucroc score """
            defect_gt = np.squeeze(img2.cpu().numpy()).transpose(1,2,0)
            true_mask = defect_gt[:, :, 0].astype('int32')
            label_pred.append(errorMap)
            label_true.append(true_mask)    
            print(f'EP={global_index} defect_img_idx={idx}')

        """ for good type """
        for (idx, img) in test_loader:
            img = img.cuda()
            idx = idx[0].item()
            
            errorMap = img_good_origin_feature[idx].reshape((1024, 1024))
            
            """ draw errorMap """
            img_ = np.squeeze(img.detach().cpu().numpy()).transpose((1,2,0))
            ironman_grid = plt.GridSpec(1, 2)
            fig = plt.figure(figsize=(12,6), dpi=100)
            ax1 = fig.add_subplot(ironman_grid[0,0])
            ax1.set_axis_off()
            im1 = ax1.imshow(errorMap, cmap="Blues")
            ax2 = fig.add_subplot(ironman_grid[0,1])
            ax2.set_axis_off()
            im2 = ax2.imshow(img_)
            
            errorMapPath = f"{ ROOT }/Results/testing_multiMap/{ args.data }/good/{ args.kmeans }/origin_map/"
            if not os.path.isdir(errorMapPath):
                os.makedirs(errorMapPath)
                print(f"----- create folder for { args.data } | type: good -----")
            errorMapName = f"{ str(idx) }_{ str(global_index) }.png"

            plt.axis('off')
            plt.savefig(errorMapPath+errorMapName, dpi=100)
            plt.close(fig)

            """ for computing aucroc score """
            defect_gt = np.zeros((1024, 1024, 3))
            true_mask = defect_gt[:, :, 0].astype('int32')
            label_pred.append(errorMap)
            label_true.append(true_mask)    
            print(f'EP={ global_index } good_img_idx={ idx }')

        label_pred = myNorm(np.array(label_pred))
        auc = roc_auc_score(np.array(label_true).flatten(), label_pred.flatten())

        f = open("overlap_score.txt", "a")
        f.write("AUC score for testing data {} with origin feature: {}".format(args.data, auc))
        f.close()
        
        print("AUC score for testing data {} with origin feature: {}".format(args.data, auc))
    except:
        print('Multi Map calculate error')