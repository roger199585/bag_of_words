import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
from torchvision import transforms
import os
from PIL import Image
from tqdm import tqdm
import pickle
import argparse
import matplotlib.pyplot as plt
import random
import cv2
import math 
import functools
from config import ROOT
import scipy.ndimage as ndimage
import sys

from ei import patch
patch(select=True)

transform_set = [ 
#    transforms.RandomPerspective(distortion_scale=0.5, p=0.5, interpolation=2),
    transforms.RandomGrayscale(p=0.5),
#    transforms.GaussianBlur(7, 3),
]

train_path = './normalize_value/bottle/'
train_mean = pickle.load(open(train_path + 'mean.pickle', 'rb'))
train_std = pickle.load(open(train_path + 'std.pickle', 'rb'))
train_mean = np.array(train_mean)
train_std = np.array(train_std)

data_transforms = {
    'ft-train': transforms.Compose([
        transforms.Resize(64),
        transforms.ToTensor()
    ]),
    'ft-val': transforms.Compose([
        transforms.Resize(64),
        transforms.ToTensor()
    ]),
    'aug': transforms.Compose([
        transforms.RandomChoice(transform_set),
        transforms.ToTensor(),
    ]),
    'train': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(train_mean, train_std)
    ]),
    'train2': transforms.Compose([
        transforms.ToTensor()
    ])
}

@functools.lru_cache(300)
def cached_load_image(path):
    return Image.open(path).convert('RGB')

def get_partial(img, i, j):
    if (i < 2 and j < 2):
        img = img[:, i*64:(i+5)*64, j*64:(j+5)*64]
    elif (i < 2 and j > 12):
        img = img[:, i*64:(i+5)*64, (j-5)*64:j*64]
    elif (i > 12 and j < 2):
        img = img[:, (i-5)*64:i*64, j*64:(j+5)*64]
    elif (i > 12 and j > 12):
        img = img[:, (i-5)*64:i*64, (j-5)*64:j*64]
    elif (i < 2 and 2 <= j <= 12):
        img = img[:, i*64:(i+5)*64, (j-2)*64:(j+3)*64]
    elif (i > 12 and 2 <= j <= 12):
        img = img[:, (i-5)*64:i*64, (j-2)*64:(j+3)*64]
    elif (j < 2 and 2 <= i <= 12):
        img = img[:, (i-2)*64:(i+3)*64, j*64:(j+5)*64]
    elif (j > 12 and 2 <= i <= 12):
        img = img[:, (i-2)*64:(i+3)*64, (j-5)*64:j*64]

    else:
        img = img[:, (i-2)*64:(i+3)*64, (j-2)*64:(j+3)*64]
        
    # print("i: {},j: {} | img size: {}".format(i,j,img.size()))

    return img

class MvtecLoader(Dataset):
    def __init__(self, dir, transforms_type='train'):  
        self.dir = dir
        self.list = os.listdir(self.dir)
        self.list.sort(key= lambda x: int(x[:-4])) # 移除 .png 的檔案名稱
        self.trans = transforms_type

    def __len__(self):
        return len(self.list)

    def __getitem__(self, index):
        img_path = self.dir + '/' + self.list[index]
        img = cached_load_image(img_path)
        norm_img = data_transforms['train'](img)
        img = data_transforms['train2'](img)
        return index, img, norm_img

class MaskLoader(Dataset):
    def __init__(self, dir):
        self.dir = dir
        self.list = os.listdir(self.dir)
        self.list.sort(key= lambda x: int(x[:3]))

    def __len__(self):
        return len(self.list)

    def __getitem__(self, index):
        img_path = self.dir + '/' + self.list[index]
        img = cached_load_image(img_path)
        # img = Image.open(img_path).convert('RGB')
        img = data_transforms['train2'](img)
        return index, img
        
class FullPatchLoader(Dataset):
    def __init__(self, img, mask, label):
        self.img_path = img
        self.mask_path = mask
        self.img_list = os.listdir(self.img_path)
        self.mask_list = os.listdir(self.mask_path)
        self.img_list.sort(key= lambda x: int(x[:-4]))
        self.mask_list.sort(key= lambda x: int(x[4:-4]))
        self.label_list = torch.load(label)

        """ count the num of each label class for weight sampling """ 
        label_count = torch.tensor(self.label_list)
        self.class_sample_count = np.array([len(np.where(label_count==t)[0]) for t in np.unique(label_count)])
        self.weight = 1. / self.class_sample_count
        # print(self.class_sample_count, len(self.class_sample_count))
        self.samples_weights = np.array([self.weight[t] for t in label_count])
    
    def __len__(self):
        return len(self.label_list)

    def __getitem__(self, index):
        """ idx for img and mask """
        img_idx = index // 256
        mask_idx = index % 256
        
        img_ = self.img_path + '/' + self.img_list[img_idx]
        img__ = Image.open(img_).convert('RGB')
        mask_ = self.mask_path + '/' + self.mask_list[mask_idx]
        mask__ = Image.open(mask_)
        label = self.label_list[index]

        img = data_transforms['train'](img__)
        mask = data_transforms['train'](mask__)

        return index, img, mask, label
       
class NoisePatchDataloader(Dataset):
    def __init__(self, img, label, left_i_path, left_j_path):
        self.img_path = img
        self.img_list = os.listdir(self.img_path)
        self.img_list = [ x for x in self.img_list if x.endswith('.png') ]
        self.img_list.sort(key=lambda x: int(x[:-4]))

        self.label_list = torch.load(label)
        self.left_i_list = pickle.load(open(left_i_path, 'rb'))
        self.left_j_list = pickle.load(open(left_j_path, 'rb'))

        """ weight sampling """
        label_count = torch.tensor(self.label_list)
        self.class_sample_count = np.array([len(np.where(label_count==t)[0]) for t in np.unique(label_count)])
        self.weight = 1. / self.class_sample_count
        self.samples_weights = np.array([self.weight[t] for t in label_count])

    def __len__(self):
        return len(self.label_list)

    def __getitem__(self, index):
        img_idx = index // 256
        img = self.img_path + "/" + self.img_list[img_idx]
        img = cached_load_image(img)
        norm_img = data_transforms['train'](img)
        img = data_transforms['train2'](img)
        
        """ for mask position """
        left_i = self.left_i_list[index]
        left_j = self.left_j_list[index]
        mask = torch.ones(1, 256, 256)
        mask_idx = index % 256

        i = mask_idx // 16 
        j = mask_idx % 16
        mask[:, i*16+left_i:i*16+16+left_i, j*16+left_j:j*16+16+left_j] = 0

        label = self.label_list[index]

        return index, img, norm_img, left_i, left_j, label, mask

# 這是給先抽取 feature 在去切 patch 用的
class NoisePatchDataloader2(Dataset):
    def __init__(self, img, label):
        self.img_path = img
        self.img_list = os.listdir(self.img_path)
        self.img_list = [ x for x in self.img_list if x.endswith('.png') ]
        self.img_list.sort(key=lambda x: int(x[:-4]))

        self.label_list = torch.load(label)

        """ weight sampling """
        label_count = torch.tensor(self.label_list)
        self.class_sample_count = np.array([len(np.where(label_count==t)[0]) for t in np.unique(label_count)])
        self.weight = 1. / self.class_sample_count
        self.samples_weights = np.array([self.weight[t] for t in label_count])

    def __len__(self):
        return len(self.label_list)

    def __getitem__(self, index):
        img_idx = index // 49
        img = self.img_path + "/" + self.img_list[img_idx]
        img = cached_load_image(img)
        img = data_transforms['train'](img)
        
        """ for mask position """
        mask = torch.ones(1, 224, 224)
        mask_idx = index % 49

        i = mask_idx // 7
        j = mask_idx % 7
        mask[:, i*32:i*32+32, j*32:j*32+32] = 0

        label = self.label_list[index]

        return index, img, label, mask


def fullPatchLabel(label_path, model, data, kmeans, batch):

    index_label = torch.load(label_path)
    label_list = []
    
    saveLabelPath = "{}/preprocessData/label/fullPatch/{}/{}/".format(
        ROOT,
        args.model,
        args.data
    )

    saveLabelName = "kmeans_{}_{}.pth".format(
        str(kmeans), 
        str(batch)
    )

    if not os.path.isdir(saveLabelPath):
        os.makedirs(saveLabelPath)
    
    chunk_num = int(args.image_size / args.patch_size)
    count = np.zeros(args.kmeans)
    for idx in range(len(index_label)):
        for i in range(chunk_num):
            for j in range(chunk_num):
                label = index_label[idx][i*chunk_num+j]
                label_list.append(label)

                count[label] += 1
    
    print(count)

    print("save label: ", saveLabelPath+saveLabelName)
    torch.save(label_list, saveLabelPath+saveLabelName)



    
if __name__ == "__main__":
    
    """ parameters """
    parser = argparse.ArgumentParser()
    parser.add_argument('--kmeans', type=int, default=128)
    parser.add_argument('--data', type=str, default='bottle')
    parser.add_argument('--batch', type=int, default=100)
    parser.add_argument('--model', type=str, default='vgg19')
    parser.add_argument('--patch_size', type=int, default=16)
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--dim_reduction', type=str, default='PCA')
    args = parser.parse_args()
    out = args.kmeans

    train_path = "{}/dataset/{}/train_resize".format(ROOT, args.data.split("_")[0])
    label_path = "{}/preprocessData/label/{}/{}/{}/train/{}_{}.pth".format(
        ROOT,
        args.model,
        str(args.dim_reduction),
        args.data,
        str(out),
        str(args.batch)
    )
    fullPatchLabel(label_path, model=args.model, data=args.data, kmeans=args.kmeans, batch=args.batch)


    
