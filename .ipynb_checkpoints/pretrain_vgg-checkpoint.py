# PyTorch
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
from torchvision import transforms

# standar library
import os
import sys
import pickle
import random
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.append('../')
# custimize library
import dataloaders
from config import ROOT


cfgs = {
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M'],
}

class VGG(nn.Module):
    def __init__(self, features, num_classes=1000):
        super(VGG, self).__init__()
        self.features = features
        #self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

    def forward(self, x):
        x = self.features(x)
        #x = self.avgpool(x)
        #x = torch.flatten(x, 1)
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

""" load part of pretrained model """
pre_model = models.vgg19(pretrained=True)

pre_model = pre_model.state_dict()
model = vgg19()
model_dict = model.state_dict()

pre_model = {k: v for k, v in pre_model.items() if k in model_dict}
model_dict.update(pre_model) 
model.load_state_dict(pre_model)


""" save chunks of training datas to fit the corresponding kmeans """

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='bottle')
    args = parser.parse_args()

    print('data: ', args.data)

    """ load dataset """
    train_dataset = dataloaders.MvtecLoader(ROOT + '/dataset/' + args.data + '/train_resize/good/')
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    model = model.to(device)
    model.train()
    patch_list = []
    patch_i = []
    patch_j = []

    for idx, img in tqdm(train_loader):
        for i in range(16):
            for j in range(16):
                noise_i = random.randint(-32, 32)
                noise_j = random.randint(-32, 32)

                if (i*64+64+noise_i > 1024 or i*64+noise_i < 0):
                    noise_i = 0
                if (j*64+64+noise_j > 1024 or j*64+noise_j < 0):
                    noise_j = 0
                
                patch_i.append(noise_i)
                patch_j.append(noise_j)

                img_ = img[:, :, i*64+noise_i:i*64+noise_i+64, j*64+noise_j:j*64+noise_j+64].to(device)
               
                output = model.forward(img_)
                """ flatten the dimension of H and W """
                out_ = output.flatten(1,2).flatten(1,2).squeeze()
                patch_list.append(out_.detach().cpu().numpy())

    save_chunk = ROOT + '/preprocessData/chunks/vgg19/'
    if not os.path.isdir(save_chunk):
        os.makedirs(save_chunk)
    save_coor = ROOT + '/preprocessData/coordinate/vgg19/{}/'.format(args.data)
    if not os.path.isdir(save_coor):
        os.makedirs(save_coor)

    save_i = 'left_i.pickle'
    save_j = 'left_j.pickle'

    with open(save_chunk+'chunks_' + args.data +'_train.pickle', 'wb') as write:
        print(np.array(patch_list).shape)
        pickle.dump(patch_list, write)

    with open(save_coor+save_i, 'wb') as write:
        print(np.array(patch_i).shape)
        pickle.dump(patch_i, write)

    with open(save_coor+save_j, 'wb') as write:
        print(np.array(patch_j).shape)
        pickle.dump(patch_j, write)
