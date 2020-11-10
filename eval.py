import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms
import torchvision.models as models

import cv2
import sys
import time
import random
import pickle
import argparse
import numpy as np
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score｀

import resnet
import dataloaders
import pretrain_vgg
import model_weightSample
from utils.tools import draw_errorMap


from config import ROOT


""" set parameters """
parser = argparse.ArgumentParser()
parser.add_argument('--kmeans', type=int, default=16)
parser.add_argument('--data', type=str, default='bottle')
parser.add_argument('--model', type=str, default='vgg19')
parser.add_argument('--batch_size', type=int, default=16)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

## Models
pretrain_model = nn.DataParallel(pretrain_vgg.model).to(device)
scratch_model = nn.Sequential(
    resnet.resnet18(pretrained=False, num_classes=args.kmeans)
)
scratch_model = nn.DataParallel(scratch_model).cuda()

# out = args.kmeans
### Dataset for all defect type
eval_path = "{}/dataset/{}/test_resize/all".format(ROOT, args.data)
eval_dataset = dataloaders.MvtecLoader(eval_path)
eval_loader = DataLoader(eval_dataset, batch_size=1, shuffle=False)

eval_mask_path = "{}/dataset/{}/ground_truth_resize/all/".format(ROOT, args.data)
eval_mask_dataset = dataloaders.MaskLoader(eval_mask_path)
eval_mask_loader = DataLoader(eval_mask_dataset, batch_size=1, shuffle=False)

test_good_path = "{}/dataset/{}/test_resize/good/".format(ROOT, args.data)
test_good_dataset = dataloaders.MvtecLoader(test_good_path)
test_good_loader = DataLoader(test_good_dataset, batch_size=1, shuffle=False)

# ## Label
test_label_name = "{}/preprocessData/label/vgg19/{}/test/all_{}_100.pth".format(ROOT, args.data, args.kmeans)
test_label = torch.tensor(torch.load(test_label_name))

## Clusters
kmeans_path = "{}/preprocessData/kmeans/{}/vgg19_{}_100_16.pickle".format(ROOT, args.data, args.kmeans)
kmeans = pickle.load(open(kmeans_path, "rb"))
pca_path = "{}/preprocessData/PCA/{}/vgg19_{}_100_16.pickle".format(ROOT, args.data, args.kmeans)
pca = pickle.load(open(pca_path, "rb"))


## Other
pil_to_tensor = transforms.ToTensor()
eval_fea_count = 0

def myNorm(features):
    return features / features.max()


def eval_feature(epoch, model, test_loader, writer):
    global eval_fea_count
    global pretrain_model
    global kmeans

    model.eval()
    pretrain_model.eval()

    with torch.no_grad():
        img_feature = []
        total_gt = []
        total_idx = []

        for (idx, img) in test_loader:
            img = img.to(device)
            idx = idx[0].item()

            print(f'eval phase: img idx={idx}')

            value_feature = []
            value_label = []
            label_idx = []
            label_gt = []

            """ batch = 128 """
            xs = []
            ys = []
            crop_list = []

            for i in range(16):
                for j in range(16):
                    crop_img = img[:, :, i*64:i*64+64, j*64:j*64+64].to(device)
                    crop_output = pretrain_model(crop_img)
                    """ flatten the dimension of H and W """
                    out_ = crop_output.flatten(1,2).flatten(1,2)
                    out = pca.transform(out_.detach().cpu().numpy())
                    out = pil_to_tensor(out).squeeze().to(device)
                    crop_list.append(out)

                    mask = torch.ones(1, 1, 1024, 1024)
                    mask[:, :, i*64:i*64+64, j*64:j*64+64] = 0
                    mask = mask.to(device)
                    x = img * mask
                    x = torch.cat((x, mask), 1)
                    label = test_label[idx][i*16+j].to(device)
                   
                    xs.append(x)
                    ys.append(label)

                if (len(xs) == args.batch_size):
                    x = torch.cat(xs, 0)
                    y = torch.stack(ys).squeeze().to(device)
                    xs.clear()
                    ys.clear()

                    output = model(x)
                    y_ = output.argmax(-1).detach().cpu().numpy()

                    acc = (output.argmax(-1).detach() == y).float().mean()  

                    for k in range(16):
                        label_idx.append(y_[k])
                        label_gt.append(y[k].item())
                        output_center = kmeans.cluster_centers_[y_[k]]
                        output_center = np.reshape(output_center, (1, -1))
                        output_center = pil_to_tensor(output_center).to(device)
                        output_center = torch.squeeze(output_center)

                        isWrongLabel = int(y_[k] != y[k].item())
                        diff = isWrongLabel * nn.MSELoss()(output_center, crop_list[k])

                        un_out = torch.unsqueeze(output[k], dim=0)
                        un_y = torch.unsqueeze(y[k], dim=0).long()

                        diff_label = nn.CrossEntropyLoss()(un_out, un_y)
                        value_feature.append(diff.item())

                        writer.add_scalar('test_feature_loss', diff.item(), eval_fea_count)
                        writer.add_scalar('test_label_loss', diff_label.item(), eval_fea_count)
                        writer.add_scalar('test_label_acc', acc.item(), eval_fea_count)
                    crop_list.clear()
                    
                    eval_fea_count += 1

            total_gt.append(label_gt)
            total_idx.append(label_idx)
            img_feature.append(value_feature)
    print(np.array(img_feature).shape)
    print(len(test_loader))
    img_feature = np.array(img_feature).reshape((len(test_loader), -1))
    total_gt = np.array(total_gt).reshape((len(test_loader), -1))
    total_idx = np.array(total_idx).reshape((len(test_loader), -1))

    return myNorm(img_feature), total_gt, total_idx


if __name__ == "__main__":
    writer = SummaryWriter(log_dir="../full_tensorboard/{}_{}_{}_{}".format(args.data, 'all', args.kmeans, datetime.now()))

    for i in range(1, 30):
        scratch_model.load_state_dict(torch.load('{}/models/vgg19/{}/exp1_{}_{}.ckpt'.format(ROOT, args.data, args.kmeans, i)))            
        scratch_model = scratch_model.to(device)

        print("------- For defect type -------")
        value_feature, total_gt, total_idx = eval_feature(i, scratch_model, eval_loader, writer)
        print("------- For good type -------")
        value_good_feature, total_good_gt, total_good_idx = eval_feature(i, scratch_model, test_good_loader, writer)

        label_pred = []
        label_gt = []

        """ for defect type """ 
        for ((idx, img), (idx2, img2)) in zip(eval_loader, eval_mask_loader):
            img = img.cuda()
            idx = idx[0].item()

            error_map = np.zeros((1024, 1024))
            for index, scalar in enumerate(value_feature[idx]):
                mask = cv2.imread('dataset/big_mask/mask{}.png'.format(index), cv2.IMREAD_GRAYSCALE)
                mask = np.invert(mask)
                mask[mask==255]=1
                
                error_map += mask * scalar

            ## 可以在這邊算
            defect_gt = np.squeeze(img2.cpu().numpy()).transpose(1,2,0)
            true_mask = defect_gt[:, :, 0].astype('int32')
            label_pred.append(error_map)
            label_gt.append(true_mask)    
            print(f'EP={i} defect_img_idx={idx}')


        """ for good type """
        for (idx, img) in test_good_loader:
            img = img.cuda()
            idx = idx[0].item()

            error_map = np.zeros((1024, 1024))
            for index, scalar in enumerate(value_good_feature[idx]):
                mask = cv2.imread('dataset/big_mask/mask{}.png'.format(index), cv2.IMREAD_GRAYSCALE)
                mask = np.invert(mask)
                mask[mask==255]=1
                error_map += mask * scalar

            defect_gt = np.zeros((1024, 1024, 3))
            true_mask = defect_gt[:, :, 0].astype('int32')
            label_pred.append(error_map)
            label_gt.append(true_mask)    
            print(f'EP={i} good_img_idx={idx}')

        try:
            auc = roc_auc_score(np.array(label_gt).flatten(), np.array(label_pred).flatten())
            print("AUC score for testing data {}: {}".format(auc, args.data))
        except ValueError:
            pass
