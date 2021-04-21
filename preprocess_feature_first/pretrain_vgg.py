"""
    Author: Corn

    Update: 2021/3/7
    History: 
        2021/3/7 -> 更改成先取 feature 之後再去切 patch

    Description: 使用 pretrain 的 vgg19 將每個 patch 轉換成 feature 並存下來
"""

""" STD Library """
import os
import sys
import pickle
import random
import argparse
import numpy as np
from tqdm import tqdm
sys.path.append('../')

""" Pytorch Libaray """
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader

""" Custom Library """
sys.path.append("./")
import dataloaders
from config import ROOT


cfgs = {
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}

class VGG(nn.Module):
    def __init__(self, features, num_classes=1000):
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AvgPool2d(2, stride=2)

    def forward(self, x):
        x = self.features(x)
        # x = self.avgpool(x)
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
    # return _vgg('vgg19', 'E', True, pretrained, progress, **kwargs)

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
    parser.add_argument('--image_size', type=int, default=1024)
    parser.add_argument('--patch_size', type=int, default=64)
    parser.add_argument('--fine_tune_epoch', type=int, default=100)
    args = parser.parse_args()

    chunk_num = (int)(args.image_size / args.patch_size)

    print('data: ', args.data)

    """" """
    patch_list = []
    patch_i = []
    patch_j = []

    model = model.to(device)
    if args.fine_tune_epoch != 0:
        model.load_state_dict(torch.load(f"/train-data2/corn/fine-tune-models/{ args.data.split('_')[0] }/{ args.fine_tune_epoch }.ckpt"))

    """ Load dataset """
    train_dataset = dataloaders.MvtecLoader( f"{ ROOT }/dataset/{ args.data.split('_')[0] }/train_resize/good/" )
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)

    test_dataset = dataloaders.MvtecLoader( f"{ ROOT }/dataset/{ args.data.split('_')[0] }/test_resize/all/" )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    for idx, img in tqdm(train_loader):
        model.eval()
        img = img.to(device)
        feature = model(img)
        for i in range(chunk_num):
            for j in range(chunk_num):
                patch_list.append(feature[0, :, i, j].detach().cpu().numpy())
    

    save_chunk = f"{ ROOT }/preprocessData/chunks/vgg19/"
    if not os.path.isdir(save_chunk):
        os.makedirs(save_chunk)

    with open( f"{ save_chunk }chunks_{ args.data }_train.pickle", 'wb') as write:
        print(np.array(patch_list).shape)
        pickle.dump(patch_list, write)

