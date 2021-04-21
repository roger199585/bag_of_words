"""
    Author: Corn

    Update: 2021/4/16
    History: 
        2021/4/16 -> 驗證兩個同位置的 patch 在 feature base 上面是否距離有被拉進

    Description: 現在正在實驗如果將 vgg19 進行 fine-tune 之後，能不能把同位置但是 feature 差異大的這個現象降低
"""

""" STD Library """
import sys
import pickle
import argparse
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
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
    parser.add_argument('--pos_x', type=int, default=0)
    parser.add_argument('--pos_y', type=int, default=0)
    parser.add_argument('--fine_tune_epoch', type=int, default=0)
    args = parser.parse_args()

    # print('data: ', args.data)

    pca = pickle.load(open(f"preprocessData/PCA/{ args.data }/vgg19_128_100_128.pickle", 'rb'))

    model = model.to(device)
    if args.fine_tune_epoch != 0:
        model.load_state_dict(torch.load(f"/train-data2/corn/fine-tune-models/{ args.data.split('_')[0] }/{ args.fine_tune_epoch }.ckpt"))

    """ Load dataset """
    train_dataset = dataloaders.MvtecLoader( f"{ ROOT }/dataset/{ args.data.split('_')[0] }/train_resize/good/" )
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)

    test_dataset = dataloaders.MvtecLoader( f"{ ROOT }/dataset/{ args.data.split('_')[0] }/test_resize/good/" )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    features = []
    for idx, img in test_loader:
        model.eval()
        img = img.to(device)
        feature = model(img)
        
        features.append(feature[0, :, args.pos_x, args.pos_y].detach().cpu().numpy())
    
    features = np.array(features) # 特定位置經過 pre-trained model 所得到的特徵
    features = pca.transform(features) # 用處理好的 pca 將 feature 降維至 128D

    diff = []
    diff_max = []
    for i in range(features.shape[0]):
        for j in range(features.shape[0]):
            diff.append( np.abs(features[i] - features[j]).mean() )
            diff_max.append( np.abs(features[i] - features[j]).max() )
    
    print(np.array(diff).mean(), np.array(diff).max())
    plt.imshow(np.array(diff).reshape(features.shape[0], features.shape[0]), cmap='binary', interpolation='nearest')
    plt.savefig(f"pos_{ args.pos_x }_{ args.pos_y }_tune_{ args.fine_tune_epoch }_comp.png")
