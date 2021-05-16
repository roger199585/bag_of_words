"""
    Author: Corn

    Update: 2021/5/3

    Description: 使用預先訓練好的 SSL model 去做我們的異常偵測的前處理
"""

""" Pytorch Library """
import torch
from torch.utils.data import Dataset, DataLoader

""" STL Library """
import os
import random
import pickle
import argparse
import numpy as np
from tqdm import tqdm

import torch.nn as nn

""" Customize model """
import dataloaders
from config import ROOT
from preprocess_SSL.SSL import model as ssl_model

# import preprocess.pretrain_vgg as pretrain_vgg



""" Save chunks of training datas to fit the corresponding kmeans """
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    """ Set parameters """ 
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='bottle')
    parser.add_argument('--patch_size', type=int, default=16)
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--dim_reduction', type=str, default='PCA')
    args = parser.parse_args()

    print('data: ', args.data)
    print('patch size: ', args.patch_size)
    print('image size: ', args.image_size)

    model_path = f"{ ROOT }/preprocess_SSL/SSL/KNN/exp3/{ args.data }/2048_2000.pth"

    pretrain_model = ssl_model.Model()
    pretrain_model = nn.DataParallel(pretrain_model).cuda()
    pretrain_model.load_state_dict(torch.load(model_path))
    # pretrain_model = pretrain_vgg.model
    # pretrain_model = pretrain_model.to(device)
    # pretrain_model.eval()
    
    patch_list = []
    patch_i = []
    patch_j = []

        
    """ Load dataset """
    train_dataset = dataloaders.MvtecLoader( f"{ ROOT }/dataset/{ args.data }/train_resize/good/" )
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)

    test_dataset = dataloaders.MvtecLoader( f"{ ROOT }/dataset/{ args.data }/test_resize/all/" )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    for idx, img in tqdm(train_loader):
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
                pretrain_model.eval()
                output, _ = pretrain_model.forward(img_)
                # output = pretrain_model.forward(img_)
                """ flatten the dimension of H and W """
                out_ = output[0, :, :, :].flatten(0,1).flatten(0,1)
                patch_list.append(out_.detach().cpu().numpy())

    save_chunk = f"{ ROOT }/preprocessData/chunks/ssl/"
    save_coor  = f"{ ROOT }/preprocessData/coordinate/ssl/{ args.dim_reduction }/{ args.data }/"
    
    if not os.path.isdir(save_chunk):
        os.makedirs(save_chunk)
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
