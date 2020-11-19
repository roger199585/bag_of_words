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
from PIL import Image
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt

# customize
import resnet
import dataloaders
import pretrain_vgg
import pretrain_resnet
from config import ROOT, RESULT_PATH
from visualize import errorMap
from utils.tools import one_hot, one_hot_forMap, draw_errorMap

# evaluations
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_auc_score, average_precision_score


# from config import gamma

""" set parameters """
parser = argparse.ArgumentParser()
parser.add_argument('--kmeans', type=int, default=16)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--epoch', type=int, default=10)
parser.add_argument('--data', type=str, default='bottle')
parser.add_argument('--type', type=str, default='good')
parser.add_argument('--batch', type=int, default=100)
parser.add_argument('--dim', type=int, default=16)
parser.add_argument('--model', type=str, default='vgg19')
parser.add_argument('--train_batch', type=int, default=16)
parser.add_argument('--test_batch_size', type=int, default=64)
parser.add_argument('--with_mask', type=str, default='True')
args = parser.parse_args()

MAXAUCEPOCH = 0
ALLAUC = []

kmeans_path = "{}/preprocessData/kmeans/{}/{}_{}_{}_{}.pickle".format(
    ROOT,
    args.data,
    str(args.model),
    str(args.kmeans),
    str(args.batch),
    str(args.dim)
)

pca_path = "{}/preprocessData/PCA/{}/{}_{}_{}_{}.pickle".format(
    ROOT,
    args.data, 
    str(args.model), 
    str(args.kmeans), 
    str(args.batch), 
    str(args.dim)
)

left_i_path = "{}/preprocessData/coordinate/vgg19/{}/left_i.pickle".format(ROOT, args.data)
left_j_path = "{}/preprocessData/coordinate/vgg19/{}/left_j.pickle".format(ROOT, args.data)

kmeans = pickle.load(open(kmeans_path, "rb"))
pca = pickle.load(open(pca_path, "rb"))


""" image transform """
pil_to_tensor = transforms.ToTensor()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#-------------------------------------

out = args.kmeans
scratch_model = nn.Sequential(
    resnet.resnet50(pretrained=False, num_classes=out),
)

""" training """
train_path = "{}/dataset/{}/train_resize/good".format(ROOT, args.data)
label_name = "{}/preprocessData/label/fullPatch/{}/{}/kmeans_{}_{}.pth".format(
    ROOT,
    str(args.model),
    args.data,
    str(out),
    str(args.batch)
)
mask_path = "{}/dataset/big_mask/".format(ROOT)

if args.model == 'vgg19':
    pretrain_model = nn.DataParallel(pretrain_vgg.model).to(device)

if args.model == 'resnet34':
    pretrain_model = nn.DataParallel(pretrain_resnet.model).to(device)

print('training data: ', train_path)
print('training label: ', label_name)

""" testing """
if (args.type == 'good'):
    test_path = "{}/dataset/{}/test_resize/good".format(ROOT, args.data)
    test_label_name = "{}/preprocessData/label/{}/{}/test/good_{}_{}.pth".format(
        ROOT,
        str(args.model),
        args.data,
        str(out),
        str(args.batch)
    )
    
    all_test_label_name = "{}/preprocessData/label/{}/{}/test/all_{}_{}.pth".format(
        ROOT,
        str(args.model),
        args.data,
        str(out),
        str(args.batch)
    )

else:
    test_path = "{}/dataset/{}/test_resize/{}".format(ROOT, args.data, args.type)
    test_label_name = "{}/preprocessData/label/{}/{}/test/{}_{}_{}.pth".format(
        ROOT,
        str(args.model),
        args.data,
        args.type,
        str(out),
        str(args.batch)
    )
    defect_gt_path = "{}/dataset/{}/ground_truth_resize/{}/".format(ROOT, args.data, args.type)


test_label = torch.tensor(torch.load(test_label_name))
print(test_label.shape)
all_test_label = torch.tensor(torch.load(all_test_label_name))
print(all_test_label.shape)

""" eval """
eval_path = "{}/dataset/{}/test_resize/all".format(ROOT, args.data)
eval_mask_path = "{}/dataset/{}/ground_truth_resize/all/".format(ROOT, args.data)

print('testing data: ', test_path)
print('testing label: ', test_label_name)

eval_fea_count = 0

def myNorm(features):
    return features / features.max()

def eval_with_origin_feature(model, test_loader, test_data, global_index, good=False):
    global cluster_features
    global pretrain_model
    global kmeans
    global pca

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

            """ slide window = 16 """
            map_num = int((1024 - 64) / 16 + 1)   ## = 61
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
                    crop_img = img[:, :, i*16:i*16+64, j*16:j*16+64].to(device)
                    crop_output = pretrain_model(crop_img)
                    """ flatten the dimension of H and W """
                    out = crop_output.flatten(1,2).flatten(1,2)
                    out_ = pca.transform(out.detach().cpu().numpy())
                    out_label = kmeans.predict(out_)
                    out_label = torch.from_numpy(out_label).to(device)
                    # out = pil_to_tensor(out).squeeze().to(device)
                    crop_list.append(out)

                    mask = torch.ones(1, 1, 1024, 1024)
                    mask[:, :, i*16:i*16+64, j*16:j*16+64] = 0
                    mask = mask.to(device)
                    x = img * mask
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
                    
                    each_pixel_err_sum[i*16:i*16+64, j*16:j*16+64] += diff.item()
                    each_pixel_err_count[i*16:i*16+64, j*16:j*16+64] += 1

            pixel_feature = each_pixel_err_sum / each_pixel_err_count

            img_feature.append(pixel_feature)

    print(np.array(img_feature).shape)
    img_feature = np.array(img_feature).reshape((len(test_loader), -1))
    return img_feature

def eval_feature_for_multiMap(model, test_loader, test_data, global_index, good=False):
    global pretrain_model
    global kmeans
    global pca

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

            """ slide window = 16 """
            map_num = int((1024 - 64) / 16 + 1)   ## = 61
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
                    crop_img = img[:, :, i*16:i*16+64, j*16:j*16+64].to(device)
                    crop_output = pretrain_model(crop_img)
                    """ flatten the dimension of H and W """
                    out_ = crop_output.flatten(1,2).flatten(1,2)
                    out = pca.transform(out_.detach().cpu().numpy())
                    out_label = kmeans.predict(out)
                    out_label = torch.from_numpy(out_label).to(device)
                    out = pil_to_tensor(out).squeeze().to(device)
                    crop_list.append(out)

                    mask = torch.ones(1, 1, 1024, 1024)
                    mask[:, :, i*16:i*16+64, j*16:j*16+64] = 0
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
                    
                    each_pixel_err_sum[i*16:i*16+64, j*16:j*16+64] += diff.item()
                    each_pixel_err_count[i*16:i*16+64, j*16:j*16+64] += 1
            
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

            for i in range(16):
                for j in range(16):
                    crop_img = img[:, :, i*64:i*64+64, j*64:j*64+64].to(device)
                    crop_output = pretrain_model(crop_img)
                    """ flatten the dimension of H and W """
                    out_ = crop_output.flatten(1,2).flatten(1,2)
                    out = pca.transform(out_.detach().cpu().numpy())
                    out = pil_to_tensor(out).squeeze().to(device)
                    crop_list.append(out)
                    origin_feature_list.append(out_)

                    mask = torch.ones(1, 1, 1024, 1024)
                    mask[:, :, i*64:i*64+64, j*64:j*64+64] = 0
                    mask = mask.to(device)
                    x = img * mask if args.with_mask == 'True' else img
                    x = torch.cat((x, mask), 1)
                    label = __labels[idx][i*16+j].to(device)
                   
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
    writer = SummaryWriter(log_dir="{}/fullvgggeature_mask_{}_{}_{}_{}_{}".format(RESULT_PATH, args.with_mask, args.data, args.type, args.kmeans, datetime.now()))

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
    center_features_path = "{}/preprocessData/cluster_center/{}/{}.pickle".format(ROOT, args.kmeans, args.data)
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
                mask = cv2.imread('{}/dataset/big_mask/mask{}.png'.format(ROOT, index), cv2.IMREAD_GRAYSCALE)
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
                mask = cv2.imread('{}/dataset/big_mask/mask{}.png'.format(ROOT, index), cv2.IMREAD_GRAYSCALE)
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
                value_good_feature, total_good_gt, total_good_idx = eval_feature(epoch, scratch_model, test_loader, test_label, isGood=True)
        
        if not os.path.isdir('{}/models/{}/{}'.format(ROOT, args.model, args.data)):
            os.makedirs('{}/models/{}/{}'.format(ROOT, args.model, args.data))
        
        path = "{}/models/{}/{}/exp1_{}_{}_smooth.ckpt".format(
            ROOT,
            args.model, 
            args.data, 
            str(out), 
            str(epoch+1+epoch_num)
        )
        torch.save(scratch_model.state_dict(), path)



    try:
        global_index = MAXAUCEPOCH
        scratch_model.load_state_dict(torch.load('{}/models/vgg19/{}/exp1_{}_{}.ckpt'.format(ROOT, args.data, args.kmeans, global_index)))
        
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


            errorMapPath = "testing_multiMap/{}/all/{}/pca_map/".format(test_data, args.kmeans)
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
            
            errorMapPath = "testing_multiMap/{}/good/{}/pca_map/".format(test_data, args.kmeans)
            if not os.path.isdir(errorMapPath):
                os.makedirs(errorMapPath)
                print("----- create folder for {} | type: good -----".format(test_data))

            errorMapName = "{}_{}.png".format(
                str(idx),
                str(global_index)
            )

            plt.axis('off')
            plt.savefig(errorMapPath+errorMapName, dpi=100)
            plt.close(fig)

            """ for computing aucroc score """
            defect_gt = np.zeros((1024, 1024, 3))
            true_mask = defect_gt[:, :, 0].astype('int32')
            label_pred.append(errorMap)
            label_true.append(true_mask)    
            print(f'EP={global_index} good_img_idx={idx}')

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
            
            errorMapPath = "testing_multiMap/{}/good/{}/origin_map/".format(test_data, args.kmeans)
            if not os.path.isdir(errorMapPath):
                os.makedirs(errorMapPath)
                print("----- create folder for {} | type: good -----".format(test_data))

            errorMapName = "{}_{}.png".format(
                str(idx),
                str(global_index)
            )

            plt.axis('off')
            plt.savefig(errorMapPath+errorMapName, dpi=100)
            plt.close(fig)

            """ for computing aucroc score """
            defect_gt = np.zeros((1024, 1024, 3))
            true_mask = defect_gt[:, :, 0].astype('int32')
            label_pred.append(errorMap)
            label_true.append(true_mask)    
            print(f'EP={global_index} good_img_idx={idx}')

        label_pred = myNorm(np.array(label_pred))
        auc = roc_auc_score(np.array(label_true).flatten(), label_pred.flatten())

        f = open("overlap_score.txt", "a")
        f.write("AUC score for testing data {} with origin feature: {}".format(args.data, auc))
        f.close()
        
        print("AUC score for testing data {} with origin feature: {}".format(args.data, auc))
    except:
        print('Multi Map calculate error')