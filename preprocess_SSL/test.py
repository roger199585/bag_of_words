""" STD Library """
import os
import sys
import time
import pickle
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm

""" Pytorch Library """
import torch
import torchvision
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

""" Custom Library """
import dataloaders
from config import ROOT
from preprocess_SSL.SSL import model as ssl_model
# import preprocess.pretrain_vgg as pretrain_vgg

if __name__ == "__main__":
    """ Set parameters """ 
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='bottle')
    parser.add_argument('--kmeans', type=int, default=128)
    parser.add_argument('--batch', type=int, default=100)
    parser.add_argument('--dim', type=int, default=128)
    parser.add_argument('--patch_size', type=int, default=16)
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--dim_reduction', type=str, default='PCA')
    args = parser.parse_args()

    """ Load preprocess datas """
    dim_reduction_path = f"{ ROOT }/preprocessData/{ args.dim_reduction }/{ args.data }/ssl_{ str(args.kmeans) }_{ str(args.batch) }_{ str(args.dim) }.pickle"
    kmeans_path        = f"{ ROOT }/preprocessData/kmeans/{ args.dim_reduction }/{ args.data }/ssl_{ str(args.kmeans) }_{ str(args.batch) }_{ str(args.dim) }.pickle"
    left_i_path        = f"{ ROOT }/preprocessData/coordinate/ssl/{ args.dim_reduction }/{ args.data }/left_i.pickle"
    left_j_path        = f"{ ROOT }/preprocessData/coordinate/ssl/{ args.dim_reduction }/{ args.data }/left_j.pickle"

    dim_reduction      = pickle.load(open(dim_reduction_path, "rb"))
    kmeans             = pickle.load(open(kmeans_path, "rb"))
    left_i             = pickle.load(open(left_i_path, "rb"))
    left_j             = pickle.load(open(left_j_path, "rb"))

    """ Check folder if not exists auto create"""
    path      = f"{ ROOT }/dataset/{ args.data }/train_resize/good/"
    save_path = f"{ ROOT }/preprocessData/label/ssl/{ args.dim_reduction }/{ args.data }/train/{ str(args.kmeans) }_{ str(args.batch) }.pth"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = f"{ ROOT }/preprocess_SSL/SSL/KNN/exp3/{ args.data }/2048_2000.pth"

    torch.manual_seed(0)
    np.random.seed(0)
    
    pretrain_model = ssl_model.Model()
    pretrain_model = nn.DataParallel(pretrain_model).cuda()
    pretrain_model.load_state_dict(torch.load(model_path))
    # pretrain_model = pretrain_vgg.model
    # pretrain_model = pretrain_model.to(device)
    # pretrain_model.eval()

    train_dataset = dataloaders.MvtecLoader(path)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)

    img_index_list = []

    """ kmeans version """

    image_list = []
    for idx, img in tqdm(train_loader):
        img = img.to(device)
        idx = idx[0].item()

        patch_index_list = []

        chunk_num = int(args.image_size / args.patch_size)
        
        patch_list = []
        for i in range(chunk_num):
            for j in range(chunk_num):
                """ Crop the image """
                index = idx*chunk_num*chunk_num+i*chunk_num+j
                patch = img[ :, :, i*args.patch_size+left_i[index]:i*args.patch_size+args.patch_size+left_i[index], j*args.patch_size+left_j[index]:j*args.patch_size+args.patch_size+left_j[index] ].to(device)

                pretrain_model.eval()
                output, _ = pretrain_model.forward( patch )
                # output = pretrain_model.forward( patch )

                print(aaa)
                """ flatten the dimension of H and W """
                out = output[0, :, :, :].flatten(0,1).flatten(0,1)

                patch_list.append(out.detach().cpu().numpy())
        image_list.append(patch_list)