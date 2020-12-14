# import pickle

# from config import ROOT

# DATA = 'capsule'
# ## Cluster Center Features
# center_features_path = "{}/preprocessData/cluster_center/128/{}.pickle".format(ROOT, DATA)
# cluster_features = pickle.load(open(center_features_path, "rb"))

# ## 經常誤判的兩群
# cluster_features[26]
# cluster_features[36]


"""
    Author: Yong Yu Chen
    Collaborator: Corn

    Update: 2020/12/2
    History: 
        2020/12/2 -> code refactor

    Description: This file is to assign label to each patch, includes the patch in training set and testing set
"""

""" STD Library """
import os
import sys
import pickle
import argparse
import functools
import numpy as np
from PIL import Image
from tqdm import tqdm

""" sklearn Library """
from sklearn.decomposition import PCA

""" Pytorch Library """
import torch
import torchvision
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

""" Custom Library """
import pretrain_vgg
from config import ROOT

@functools.lru_cache(300)
def cached_load_image(path):
    return Image.open(path).convert('RGB')

def feature_compare():
    model = pretrain_vgg.model

    pred_path = f"{ ROOT }/preprocessData/kmeans_img/{ args.data }/128/idx_{ args.pred }.png"
    gt_path   = f"{ ROOT }/preprocessData/kmeans_img/{ args.data }/128/idx_{ args.gt }.png"
    
    pca_path    = f"{ ROOT }/preprocessData/PCA/{ args.data }/vgg19_128_100_128.pickle"
    pca = pickle.load(open(pca_path, "rb"))
    _max = 0
    for i in range(128):
        for j in range(i+1, 128):
            pred_path = f"{ ROOT }/preprocessData/kmeans_img/{ args.data }/128/idx_{ i }.png"
            gt_path = f"{ ROOT }/preprocessData/kmeans_img/{ args.data }/128/idx_{ j }.png"

            transform = transforms.Compose([
                transforms.ToTensor(),
            ])

            gt_img = cached_load_image(gt_path)
            gt_img = transform(gt_img)
            gt_img = gt_img.unsqueeze(dim=0)

            pred_img = cached_load_image(pred_path)
            pred_img = transform(pred_img)
            pred_img = pred_img.unsqueeze(dim=0)


            model.eval()
            gt_feature = model(gt_img)
            gt_feature = gt_feature.flatten(1,2).flatten(1,2)
            gt_feature = pca.transform( gt_feature.detach().cpu().numpy() )

            pred_feature = model(pred_img)
            pred_feature = pred_feature.flatten(1,2).flatten(1,2)
            pred_feature = pca.transform( pred_feature.detach().cpu().numpy() )

            dif = np.abs(gt_feature - pred_feature).mean()
            if dif > _max:
                _max = dif
            print(f"{i}, {j} => {dif}" )
    print(f"Max distance => {_max}")
    
    pred_path = f"{ ROOT }/preprocessData/kmeans_img/{ args.data }/128/idx_{ args.pred }.png"
    gt_path   = f"{ ROOT }/preprocessData/kmeans_img/{ args.data }/128/idx_{ args.gt }.png"

    gt_img = cached_load_image(gt_path)
    gt_img = transform(gt_img)
    gt_img = gt_img.unsqueeze(dim=0)

    pred_img = cached_load_image(pred_path)
    pred_img = transform(pred_img)
    pred_img = pred_img.unsqueeze(dim=0)

    model.eval()
    gt_feature = model(gt_img)
    gt_feature = gt_feature.flatten(1,2).flatten(1,2)
    gt_feature = pca.transform( gt_feature.detach().cpu().numpy() )

    pred_feature = model(pred_img)
    pred_feature = pred_feature.flatten(1,2).flatten(1,2)
    pred_feature = pca.transform( pred_feature.detach().cpu().numpy() )

    dif = np.abs(gt_feature - pred_feature).mean()
    print(f"Patch { args.pred } v.s Patch { args.gt } => { dif }")

if __name__ == "__main__":
    """ set parameters """ 
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='bottle')
    parser.add_argument('--pred', type=str, default='0')
    parser.add_argument('--gt', type=str, default='0')
    
    args = parser.parse_args()

    feature_compare()

