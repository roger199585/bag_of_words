"""
    Author: Yong Yu Chen
    Collaborator: Corn

    Update: 2020/12/2
    History: 
        2020/12/2 -> code refactor
        2021/1/7 -> 找到使用其他降維方式速度過慢問題，並且修正他

    Description: This file is to assign label to each patch, includes the patch in training set and testing set
"""

""" STD Library """
import os
import sys
import time
import pickle
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm

""" sklearn Library """
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, MDS
from sklearn.cluster import DBSCAN

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
    if not os.path.isdir( f'{ ROOT }/preprocessData/kmeans_img/{ args.dim_reduction }/{ args.data }/{ str(args.kmeans) }/'):
        os.makedirs(f'{ ROOT }/preprocessData/kmeans_img/{ args.dim_reduction }/{ args.data }/{ str(args.kmeans) }/')

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
    parser.add_argument('--patch_size', type=int, default=64)
    parser.add_argument('--image_size', type=int, default=1024)
    parser.add_argument('--fine_tune_epoch', type=int, default=0)
    parser.add_argument('--model', type=str, default='vgg19')
    parser.add_argument('--dim_reduction', type=str, default='PCA')
    args = parser.parse_args()

    """ Load preprocess datas """
    dim_reduction_path = f"{ ROOT }/preprocessData/{ args.dim_reduction }/{ args.data }/{ args.model }_{ str(args.kmeans) }_{ str(args.batch) }_{ str(args.dim) }.pickle"
    kmeans_path        = f"{ ROOT }/preprocessData/kmeans/{ args.dim_reduction }/{ args.data }/{ args.model }_{ str(args.kmeans) }_{ str(args.batch) }_{ str(args.dim) }.pickle"
    left_i_path        = f"{ ROOT }/preprocessData/coordinate/{ args.model }/{ args.dim_reduction }/{ args.data }/left_i.pickle"
    left_j_path        = f"{ ROOT }/preprocessData/coordinate/{ args.model }/{ args.dim_reduction }/{ args.data }/left_j.pickle"

    dim_reduction      = pickle.load(open(dim_reduction_path, "rb"))
    kmeans             = pickle.load(open(kmeans_path, "rb"))
    left_i             = pickle.load(open(left_i_path, "rb"))
    left_j             = pickle.load(open(left_j_path, "rb"))

    """ Check folder if not exists auto create"""
    if args.type == 'train':
        path      = f"{ ROOT }/dataset/{ args.data }/train_resize/good/"
        save_path = f"{ ROOT }/preprocessData/label/{ args.model }/{ args.dim_reduction }/{ args.data }/train/{ str(args.kmeans) }_{ str(args.batch) }.pth"

        if not os.path.isdir( f"{ ROOT }/preprocessData/label/{ args.model }/{ args.dim_reduction }/{ args.data }/train/" ):
            os.makedirs( f"{ ROOT }/preprocessData/label/{ args.model }/{ args.dim_reduction }/{ args.data }/train/" )

    elif args.type == 'test':
        path      = f"{ ROOT }/dataset/{ args.data }/test_resize/good/"
        save_path = f"{ ROOT }/preprocessData/label/{ args.model }/{ args.dim_reduction }/{ args.data }/test/good_{ str(args.kmeans) }_{ str(args.batch) }.pth"
        
        if not os.path.isdir( f"{ ROOT }/preprocessData/label/{ args.model }/{ args.dim_reduction }/{ args.data }/test/"):
            os.makedirs( f"{ ROOT }/preprocessData/label/{ args.model }/{ args.dim_reduction }/{ args.data }/test/")
            
    elif args.type == 'all':
        path      = f"{ ROOT }/dataset/{ args.data }/test_resize/all/"
        save_path = f"{ ROOT }/preprocessData/label/{ args.model }/{ args.dim_reduction }/{ args.data }/test/all_{ str(args.kmeans) }_{ str(args.batch) }.pth"
        
        if not os.path.isdir(f"{ ROOT }/preprocessData/label/{ args.model }/{ args.dim_reduction }/{ args.data }/test/"):
            os.makedirs(f"{ ROOT }/preprocessData/label/{ args.model }/{ args.dim_reduction }/{ args.data }/test/")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = pretrain_vgg.model if args.model == 'vgg19' else pretrain_resnet.model if args.model == 'resnet34' else None
    model = model.to(device)
    # if args.fine_tune_epoch != 0:
    #     model.load_state_dict(torch.load(f"/train-data2/corn/fine-tune-models/{ args.data.split('_')[0] }/{ args.fine_tune_epoch }.ckpt"))
    model.eval()

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
                if (args.type == 'train'):
                    index = idx*chunk_num*chunk_num+i*chunk_num+j
                    patch = img[ :, :, i*args.patch_size+left_i[index]:i*args.patch_size+args.patch_size+left_i[index], j*args.patch_size+left_j[index]:j*args.patch_size+args.patch_size+left_j[index] ].to(device)
                else:
                    patch = img[:, :, i*args.patch_size:i*args.patch_size+args.patch_size, j*args.patch_size:j*args.patch_size+args.patch_size].to(device)

                output = model.forward( patch )

                """ flatten the dimension of H and W """
                out = output.flatten(1,2).flatten(1,2)

                patch_list.append(out.detach().cpu().numpy())
        image_list.append(patch_list)
    
    # 處理降維  
    image_list = np.array(image_list)
    image_list = image_list.reshape(-1, image_list.shape[-1])

    new_outs = dim_reduction.transform( image_list )
    print(new_outs.shape)
    image_saved = []
    chunk_num = int(args.image_size / args.patch_size)
    for i in range(new_outs.shape[0]):
        patch_idx = kmeans.predict(new_outs[i].reshape(1, -1))
        patch_index_list.append(patch_idx)

        if args.type == 'train' and patch_idx not in image_saved:
            image_saved.append(patch_idx)
            if not os.path.isdir(f"{ ROOT }/preprocessData/kmeans_img/{ args.dim_reduction }/{ args.data }/{ str(args.kmeans) }/"):
                print('create', f"{ ROOT }/preprocessData/kmeans_img/{ args.dim_reduction }/{ args.data }/{ str(args.kmeans) }/")
                os.makedirs(f"{ ROOT }/preprocessData/kmeans_img/{ args.dim_reduction }/{ args.data }/{ str(args.kmeans) }/")

            img_idx = int(i / (chunk_num * chunk_num))
            x = int((i % 256) / 16)
            y = int((i % 256) % 16)
            image = Image.open(f"{ ROOT }/dataset/{ args.data }/train_resize/good/{ str( img_idx ).zfill(3) }.png").convert('RGB')
            patch = img[:, :, x*args.patch_size:x*args.patch_size+args.patch_size, y*args.patch_size:y*args.patch_size+args.patch_size]

            
            save_img(patch, f"{ ROOT }/preprocessData/kmeans_img/{ args.dim_reduction }/{ args.data }/{ str(args.kmeans) }/idx_{ str(patch_idx.item()) }.png")

        if len(patch_index_list) == 256:
            img_index_list.append(patch_index_list)
            patch_index_list = []
        
        # if (args.type == 'train'):
        #     if not os.path.isdir(f"{ ROOT }/preprocessData/kmeans_img/{ args.dim_reduction }/{ args.data }/{ str(args.kmeans) }/"):
        #         print('create', f"{ ROOT }/preprocessData/kmeans_img/{ args.dim_reduction }/{ args.data }/{ str(args.kmeans) }/")
        #         os.makedirs(f"{ ROOT }/preprocessData/kmeans_img/{ args.dim_reduction }/{ args.data }/{ str(args.kmeans) }/")           
        #     save_img(patch, f"{ ROOT }/preprocessData/kmeans_img/{ args.dim_reduction }/{ args.data }/{ str(args.kmeans) }/idx_{ str(patch_idx.item()) }.png")

    torch.save(img_index_list, save_path)
    print(len(img_index_list), len(img_index_list[0]))
