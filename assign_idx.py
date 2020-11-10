from sklearn.cluster import MiniBatchKMeans
from scipy.spatial import distance
import numpy as np
import pretrain_vgg
import pretrain_resnet
import random
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
from torchvision import transforms
import os
from PIL import Image
from tqdm import tqdm
from sklearn.cluster import MiniBatchKMeans
#import ipydbg
import argparse
import pickle
from sklearn.decomposition import PCA
import sys
import dataloaders
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from config import ROOT

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

""" set parameters """ 
parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='bottle')
parser.add_argument('--type', type=str, default='train')
parser.add_argument('--kmeans', type=int, default=16, help='number of kmeans clusters')
parser.add_argument('--batch', type=int, default=100)
parser.add_argument('--dim', type=int, default=16)
parser.add_argument('--model', type=str, default='vgg19')
args = parser.parse_args()

if args.model == 'vgg19':
    model = pretrain_vgg.model
elif args.model == 'resnet34':
    model = pretrain_resnet.model

kmeans_path = "{}/preprocessData/kmeans/{}/{}_{}_{}_{}.pickle".format(
    ROOT,
    args.data,
    args.model, 
    str(args.kmeans),
    str(args.batch),
    str(args.dim)
)

pca_path = "{}/preprocessData/PCA/{}/{}_{}_{}_{}.pickle".format(
    ROOT,
    args.data,
    args.model,
    str(args.kmeans),
    str(args.batch),
    str(args.dim)
)

left_i_path = "{}/preprocessData/coordinate/vgg19/{}/left_i.pickle".format(ROOT, args.data)
left_j_path = "{}/preprocessData/coordinate/vgg19/{}/left_j.pickle".format(ROOT, args.data)

kmeans = pickle.load(open(kmeans_path, "rb"))
# gmm = pickle.load(open(gmm_path, "rb"))
pca = pickle.load(open(pca_path, "rb"))
left_i = pickle.load(open(left_i_path, "rb"))
left_j = pickle.load(open(left_j_path, "rb"))

if args.type == 'train':
    path = "{}/dataset/{}/train_resize/good/".format(ROOT, args.data)
    save_path = "{}/preprocessData/label/{}/{}/train/{}_{}.pth".format(
        ROOT,
        args.model,
        args.data,
        str(args.kmeans),
        str(args.batch)
    )
    if not os.path.isdir('{}/preprocessData/label/{}/{}/train/'.format(ROOT, args.model, args.data)):
        os.makedirs('{}/preprocessData/label/{}/{}/train/'.format(ROOT, args.model, args.data))

elif args.type == 'test':
    path = "{}/dataset/{}/test_resize/good/".format(ROOT, args.data)
    save_path = "{}/preprocessData/label/{}/{}/test/good_{}_{}.pth".format(
        ROOT,
        args.model,
        args.data,
        str(args.kmeans),
        str(args.batch)
    )
    
    if not os.path.isdir('{}/preprocessData/label/{}/{}/test/'.format(ROOT, args.model, args.data)):
        os.makedirs('{}/preprocessData/label/{}/{}/test/'.format(ROOT, args.model, args.data))
        
elif args.type == 'all':
    path = "{}/dataset/{}/test_resize/all/".format(ROOT, args.data)
    save_path = "{}/preprocessData/label/{}/{}/test/all_{}_{}.pth".format(
        ROOT,
        args.model,
        args.data,
        str(args.kmeans),
        str(args.batch)
    )
    
    if not os.path.isdir('{}/preprocessData/label/{}/{}/test/'.format(ROOT, args.model, args.data)):
        os.makedirs('{}/preprocessData/label/{}/{}/test/'.format(ROOT, args.model, args.data))


def save_img(img, save_name):
    img_ = np.squeeze(img.detach().cpu().numpy()).transpose((1,2,0))
    im = Image.fromarray((img_ * 255).astype(np.uint8))
    im.save(save_name)

    
train_dataset = dataloaders.MvtecLoader(path)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
model = model.to(device)
model.eval()
img_index_list = []

""" kmeans version """
for idx, img in tqdm(train_loader):
    
    idx = idx[0].item()
    per_list = []
    for i in range(16):
        for j in range(16):
            
            if (args.type == 'train'):
                index = idx*256+i*16+j
                img_ = img[ :, :, i*64+left_i[index]:i*64+64+left_i[index], j*64+left_j[index]:j*64+64+left_j[index] ].to(device)
                # img_ = img[:, :, left_i[index]:left_i[index]+64, left_j[index]:left_j[index]+64].to(device)
            elif (args.type == 'test'):
                img_ = img[:, :, i*64:i*64+64, j*64:j*64+64].to(device)
            elif (args.type == 'all'):
                img_ = img[:, :, i*64:i*64+64, j*64:j*64+64].to(device)

            output = model.forward(img_)
            """ flatten the dimension of H and W """
            out_ = output.flatten(1,2).flatten(1,2)
            out = pca.transform(out_.detach().cpu().numpy())
            img_idx = kmeans.predict(out)
            if (args.type == 'train'):
                if not os.path.isdir('{}/preprocessData/kmeans_img/{}/{}/'.format(ROOT, args.data, str(args.kmeans))):
                    os.makedirs('{}/preprocessData/kmeans_img/{}/{}/'.format(ROOT, args.data, str(args.kmeans)))
                
                save_img(img_, '{}/preprocessData/kmeans_img/{}/{}/idx_{}.png'.format(
                        ROOT,
                        args.data,
                        str(args.kmeans),
                        str(img_idx[0])
                ))
            per_list.append(img_idx)
    img_index_list.append(per_list)

print(len(img_index_list), len(img_index_list[0]))

torch.save(img_index_list, save_path)

