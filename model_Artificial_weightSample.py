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
from preprocess_Artificial.artificial_feature import to_int, to_hist
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
parser.add_argument('--test_batch_size', type=int, default=32)
parser.add_argument('--with_mask', type=str, default='True')
parser.add_argument('--patch_size', type=int, default=64)
parser.add_argument('--image_size', type=int, default=1024)
args = parser.parse_args()

MAXAUCEPOCH = 0
ALLAUC = []

kmeans_path = f"{ ROOT }/preprocessData/kmeans/artificial/{ args.data }/artificial_{ args.kmeans }.pickle"
left_i_path = f"{ ROOT }/preprocessData/coordinate/artificial/{ args.data }/left_i.pickle"
left_j_path = f"{ ROOT }/preprocessData/coordinate/artificial/{ args.data }/left_j.pickle"

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
label_name = f"{ ROOT }/preprocessData/label/fullPatch/artificial/{ args.data }/kmeans_{ args.kmeans }.pth"
mask_path  = f"{ ROOT }/dataset/big_mask/"

print('training data: ', train_path)
print('training label: ', label_name)

""" testing """
if (args.type == 'good'):
    test_path           = f"{ ROOT }/dataset/{ args.data }/test_resize/good"
    test_label_name     = f"{ ROOT }/preprocessData/label/artificial/{ args.data }/test/good_{ args.kmeans }.pth"
    all_test_label_name = f"{ ROOT }/preprocessData/label/artificial/{ args.data }/test/all_{ args.kmeans}.pth"
else:
    test_path           = f"{ ROOT }/dataset/{ args.data }/test_resize/{ args.type }"
    defect_gt_path      = f"{ ROOT }/dataset/{ args.data }/ground_truth_resize/{ args.type }"
    test_label_name     = f"{ ROOT }/preprocessData/label/artificial/{ args.data }/test/{ args.type }_{ args.kmeans }.pth"


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

def eval_feature(epoch, model, test_loader, __labels, isGood):
    global eval_fea_count
    global kmeans

    model.eval()

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
                    out = to_hist(crop_img)
                    out_ = out.reshape(1, -1).to(device)
                    """ flatten the dimension of H and W """
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

                        output_feature = np.expand_dims(cluster_features[y_[k]], axis=0)
                        output_feature = torch.from_numpy(output_feature).cuda()

                        isWrongLabel = int(y_[k] != y[k].item())
                        origin_feature_diff = isWrongLabel * nn.MSELoss()(output_feature, origin_feature_list[k])
                        value_feature.append(origin_feature_diff.item())
                    

                        if isGood:
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
    writer = SummaryWriter(log_dir=f"{ RESULT_PATH }/Artificial_{args.data}_mask_{ args.with_mask }_patch_{ args.patch_size }_type_{ args.type }_kmeans_{ args.kmeans }_{ datetime.now() }")

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
    center_features_path = f"{ ROOT }/preprocessData/cluster_center/artificial/{ args.data }/{ args.kmeans }.pickle"
    cluster_features = pickle.load(open(center_features_path, "rb"))

    scratch_model = nn.DataParallel(scratch_model).to(device)
    epoch_num = 0

    """ training config """ 
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
        
        if not os.path.isdir(f"{ ROOT }/models/artificial/{ args.data }"):
            os.makedirs(f"{ ROOT }/models/artificial/{ args.data }")
        
        path = f"{ ROOT }/models/artificial/{ args.data }/exp_{ args.kmeans }_{ str(epoch+1+epoch_num) }.ckpt"
        torch.save(scratch_model.state_dict(), path)
