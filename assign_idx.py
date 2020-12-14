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
import dataloaders
import pretrain_vgg
import pretrain_resnet
from config import ROOT

def save_img(img, save_name):
    if not os.path.isdir( f'{ ROOT }/preprocessData/kmeans_img/{ args.data }/{ str(args.kmeans) }/'):
        os.makedirs(f'{ ROOT }/preprocessData/kmeans_img/{ args.data }/{ str(args.kmeans) }/')

    img = np.squeeze( img.detach().cpu().numpy()).transpose((1,2,0) )
    img = Image.fromarray( (img * 255).astype(np.uint8) )
    img.save(save_name)

if __name__ == "__main__":
    """ Set parameters """ 
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='bottle')
    parser.add_argument('--type', type=str, default='train')
    parser.add_argument('--kmeans', type=int, default=16, help='number of kmeans clusters')
    parser.add_argument('--batch', type=int, default=100)
    parser.add_argument('--dim', type=int, default=16)
    parser.add_argument('--model', type=str, default='vgg19')
    args = parser.parse_args()

    """ Load preprocess datas """
    kmeans_path = f"{ ROOT }/preprocessData/kmeans/{ args.data }/{ args.model }_{ str(args.kmeans) }_{ str(args.batch) }_{ str(args.dim) }.pickle"
    pca_path    = f"{ ROOT }/preprocessData/PCA/{ args.data }/{ args.model }_{ str(args.kmeans) }_{ str(args.batch) }_{ str(args.dim) }.pickle"
    left_i_path = f"{ ROOT }/preprocessData/coordinate/{ args.model }/{ args.data }/left_i.pickle"
    left_j_path = f"{ ROOT }/preprocessData/coordinate/{ args.model }/{ args.data }/left_j.pickle"

    kmeans = pickle.load(open(kmeans_path, "rb"))
    pca    = pickle.load(open(pca_path, "rb"))
    left_i = pickle.load(open(left_i_path, "rb"))
    left_j = pickle.load(open(left_j_path, "rb"))

    """ Check folder if not exists auto create"""
    if args.type == 'train':
        path      = f"{ ROOT }/dataset/{ args.data}/train_resize/good/"
        save_path = f"{ ROOT }/preprocessData/label/{ args.model }/{ args.data }/train/{ str(args.kmeans) }_{ str(args.batch) }.pth"

        if not os.path.isdir( f"{ ROOT }/preprocessData/label/{ args.model }/{ args.data }/train/" ):
            os.makedirs( f"{ ROOT }/preprocessData/label/{ args.model }/{ args.data }/train/" )

    elif args.type == 'test':
        path      = f"{ ROOT }/dataset/{ args.data }/test_resize/good/"
        save_path = f"{ ROOT }/preprocessData/label/{ args.model }/{ args.data }/test/good_{ str(args.kmeans) }_{ str(args.batch) }.pth"
        
        if not os.path.isdir( f"{ ROOT }/preprocessData/label/{ args.model }/{ args.data }/test/"):
            os.makedirs( f"{ ROOT }/preprocessData/label/{ args.model }/{ args.data }/test/")
            
    elif args.type == 'all':
        path      = f"{ ROOT }/dataset/{ args.data }/test_resize/all/"
        save_path = f"{ ROOT }/preprocessData/label/{ args.model }/{ args.data }/test/all_{ str(args.kmeans) }_{ str(args.batch) }.pth"
        
        if not os.path.isdir(f"{ ROOT }/preprocessData/label/{ args.model }/{ args.data }/test/"):
            os.makedirs(f"{ ROOT }/preprocessData/label/{ args.model }/{ args.data }/test/")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = pretrain_vgg.model if args.model == 'vgg19' else pretrain_resnet if args.model == 'resnet34' else None
    model = model.to(device)
    model.eval()

    train_dataset = dataloaders.MvtecLoader(path)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)

    img_index_list = []

    """ kmeans version """
    for idx, img in tqdm(train_loader):
        img = img.to(device)
        idx = idx[0].item()

        patch_index_list = []

        for i in range(16):
            for j in range(16):
                """ Crop the image """
                if (args.type == 'train'):
                    index = idx*256+i*16+j
                    patch = img[ :, :, i*64+left_i[index]:i*64+64+left_i[index], j*64+left_j[index]:j*64+64+left_j[index] ].to(device)
                else:
                    patch = img[:, :, i*64:i*64+64, j*64:j*64+64].to(device)

                output = model.forward( patch )

                """ flatten the dimension of H and W """
                out = output.flatten(1,2).flatten(1,2)
                out = pca.transform( out.detach().cpu().numpy() )
                patch_idx = kmeans.predict(out)

                patch_index_list.append(patch_idx)

            if (args.type == 'train'):                
                save_img(img, f'{ ROOT }/preprocessData/kmeans_img/{ args.data }/{ str(args.kmeans) }/idx_{ str(idx) }.png')
            
        img_index_list.append(patch_index_list)
    torch.save(img_index_list, save_path)

    print(len(img_index_list), len(img_index_list[0]))


