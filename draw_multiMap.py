
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn import preprocessing
import torchvision.models as models
from torchvision import transforms
import os
from PIL import Image
from tqdm import tqdm
import pretrain_vgg
import resnet
import argparse
import pickle
import cv2
from visualize import errorMap
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import random
import sys
import dataloaders
from sklearn.metrics import roc_auc_score
import time
import itertools
from config import ROOT

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pil_to_tensor = transforms.ToTensor()

def norm(feature):
    return feature / feature.max()

if __name__ == "__main__":

    """ set parameters """
    parser = argparse.ArgumentParser()
    parser.add_argument('--kmeans', type=int, default=128)
    parser.add_argument('--data', type=str, default='bottle')
    parser.add_argument('--index', type=int, default=30)
    parser.add_argument('--resume', type=bool, default=True)
    args = parser.parse_args()

    global_index = args.index
    test_data = args.data

    ### DataSet for all defect type
    test_all_path = "{}/dataset/{}/test_resize/all/".format(ROOT, args.data)
    test_all_dataset = dataloaders.MvtecLoader(test_all_path)
    test_all_loader = DataLoader(test_all_dataset, batch_size=1, shuffle=False)

    test_good_path = "{}/dataset/{}/test_resize/good/".format(ROOT, args.data)
    test_good_dataset = dataloaders.MvtecLoader(test_good_path)
    test_good_loader = DataLoader(test_good_dataset, batch_size=1, shuffle=False)

    mask_path = "{}/dataset/{}/ground_truth_resize/all/".format(ROOT, args.data)
    mask_dataset = dataloaders.MaskLoader(mask_path)
    mask_loader = DataLoader(mask_dataset, batch_size=1, shuffle=False)

    
    print("----- defect -----")
    if args.resume and os.path.isfile('{}/Results/testing_multiMap/{}/all/128_img_all_feature_{}_Origin.pickle'.format(ROOT, args.data, args.index)):
        print("load from {}/Results/testing_multiMap/{}/all/128_img_all_feature_{}_Origin.pickle".format(ROOT, args.data, args.index))
        img_all_feature = pickle.load(open('{}/Results/testing_multiMap/{}/all/128_img_all_feature_{}_Origin.pickle'.format(ROOT, args.data, args.index), 'rb'))
    else:
        img_all_feature = eval_feature(pretrain_model, scratch_model, test_all_loader, kmeans, pca, args.data, global_index, good=False)

    print("----- good -----")
    if args.resume and os.path.isfile('{}/Results/testing_multiMap/{}/good/128_img_good_feature_{}_Origin.pickle'.format(ROOT, args.data, args.index)):
        print("load from {}/Results/testing_multiMap/{}/good/128_img_good_feature_{}_Origin.pickle".format(ROOT, args.data, args.index))
        img_good_feature = pickle.load(open('{}/Results/testing_multiMap/{}/good/128_img_good_feature_{}_Origin.pickle'.format(ROOT, args.data, args.index), 'rb'))
    else:
        img_good_feature = eval_feature(pretrain_model, scratch_model, test_good_loader, kmeans, pca, args.data, global_index, good=True)
    

    label_pred = []
    label_true = []

    """ for defect type """ 
    for ((idx, img), (idx2, img2)) in zip(test_all_loader, mask_loader):
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


        errorMapPath = "{}/Results/testing_multiMap/{}/all/{}/map/".format(ROOT, test_data, args.kmeans)
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
    for (idx, img) in test_good_loader:
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
        
        errorMapPath = "{}/Results/testing_multiMap/{}/good/{}/map/".format(ROOT, test_data, args.kmeans)
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

    label_pred = norm(np.array(label_pred))
    auc = roc_auc_score(np.array(label_true).flatten(), label_pred.flatten())
    print("AUC score for testing data {}: {}".format(args.data, auc))

