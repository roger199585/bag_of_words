"""
    Author: Yong Yu Chen
    Collaborator: Corn

    Update: 2020/12/3
    History: 
        2020/12/3 -> code refactor

    Description: 使用 pretrain 的 vgg19 將每個 patch 轉換成 feature 並存下來
"""

""" STD Library """
import os
import sys
import pickle
import random
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
sys.path.append('../')

""" Pytorch Libaray """
import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

""" Custom Library """
import dataloaders
from config import ROOT


cfgs = {
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    # 'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512],
}

class VGG(nn.Module):
    def __init__(self, features, num_classes=1000):
        super(VGG, self).__init__()
        self.features = features
        # avgpool 是為了強制將我們的 feature 轉換成 512x1x1
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        return x

def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

def _vgg(arch, cfg, batch_norm, pretrained, progress, **kwargs):
    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)
    return model

def vgg19(pretrained=False, progress=True, **kwargs):
    return _vgg('vgg19', 'E', False, pretrained, progress, **kwargs)

""" Load part of pretrained model """
pre_model = models.vgg19(pretrained=True)
pre_model = pre_model.state_dict()

model = vgg19()
model_dict = model.state_dict()

pre_model = {k: v for k, v in pre_model.items() if k in model_dict}
model_dict.update(pre_model) 
model.load_state_dict(pre_model)


""" Save chunks of training datas to fit the corresponding kmeans """

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    """ Set parameters """ 
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='bottle')
    parser.add_argument('--patch_size', type=int, default=64)
    parser.add_argument('--image_size', type=int, default=1024)
    args = parser.parse_args()

    print('data: ', args.data)
    print('patch size: ', args.patch_size)
    print('image size: ', args.image_size)

    """" """
    patch_list = []
    patch_i = []
    patch_j = []

    model = model.to(device)
    """ Load dataset """
    train_dataset = dataloaders.MvtecLoader( f"{ ROOT }/dataset/{ args.data }/train_resize/good/" )
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)

    for idx, img in tqdm(train_loader):
        model.train()
        for i in range(int(args.image_size / args.patch_size)):
            for j in range(int(args.image_size / args.patch_size)):
                noise_i = random.randint(-1 * int(args.patch_size / 2) , int(args.patch_size / 2))
                noise_j = random.randint(-1 * int(args.patch_size / 2) , int(args.patch_size / 2))

                if (i * args.patch_size + args.patch_size + noise_i > args.image_size or i * args.patch_size + noise_i < 0):
                    noise_i = 0
                if (j * args.patch_size + args.patch_size + noise_j > args.image_size or j * args.patch_size + noise_j < 0):
                    noise_j = 0
                
                patch_i.append(noise_i)
                patch_j.append(noise_j)

                img_ = img[:, :, i * args.patch_size+noise_i:i*args.patch_size+noise_i+args.patch_size, j*args.patch_size+noise_j:j*args.patch_size+noise_j+args.patch_size].to(device)
               
                output = model.forward(img_)
                """ flatten the dimension of H and W """
                out_ = output.flatten(1,2).flatten(1,2).squeeze()
                patch_list.append(out_.detach().cpu().numpy())

    save_chunk = f"{ ROOT }/preprocessData/chunks/vgg19/"
    if not os.path.isdir(save_chunk):
        os.makedirs(save_chunk)
    save_coor = f"{ ROOT }/preprocessData/coordinate/vgg19/{ args.data }/"
    if not os.path.isdir(save_coor):
        os.makedirs(save_coor)

    save_i = 'left_i.pickle'
    save_j = 'left_j.pickle'

    with open( f"{ save_chunk }chunks_{ args.data }_train.pickle", 'wb') as write:
        print(np.array(patch_list).shape)
        pickle.dump(patch_list, write)

    with open(save_coor+save_i, 'wb') as write:
        print(np.array(patch_i).shape)
        pickle.dump(patch_i, write)

    with open(save_coor+save_j, 'wb') as write:
        print(np.array(patch_j).shape)
        pickle.dump(patch_j, write)
