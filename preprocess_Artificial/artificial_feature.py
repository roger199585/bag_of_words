"""
    Author: Corn

    Update: 2020/1/20
    History: 
        2021/1/20 -> Create artifical feature for images

    Description: Create artifical feature for our dataset
"""

""" STD Library """
import os
import sys
import pickle
import random
import argparse
import colorsys
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
sys.path.append('../')

""" Pytorch Library """
import torch
from torch.utils.data import DataLoader

""" Custom Library """
sys.path.append("../")
import dataloaders
from config import ROOT

from ei import patch
patch(select=True)

def to_256_colr(rgb_tensor):
    rgb_tensor = rgb_tensor * 255
    rgb_tensor = rgb_tensor.squeeze()

    new_tensor = torch.round((0.299 * rgb_tensor[0, :, :] + 0.587 * rgb_tensor[1, :, :] + 0.114 * rgb_tensor[2, :, :])) / 255
    
    return torch.histc(new_tensor, bins=256, min=0, max=1)


""" Save chunks of training datas to fit the corresponding kmeans """
if __name__ == "__main__":
    """ Set parameters """ 
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='bottle', help='category of dataset, EX: bottle, cable ...')
    parser.add_argument('--patch_size', type=int, default=64, help='Size of the patch you cut, default is 64')
    parser.add_argument('--image_size', type=int, default=1024, help='Size of your origin image')
    args = parser.parse_args()

    print('data: ', args.data)
    print('patch size: ', args.patch_size)
    print('image size: ', args.image_size)

    patch_list = []
    patch_i = []
    patch_j = []

    """ Load dataset """
    train_dataset = dataloaders.MvtecLoader( f"{ ROOT }/dataset/{ args.data }/train_resize/good/" )
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)

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

                patch = img[:, :, i * args.patch_size+noise_i:i*args.patch_size+noise_i+args.patch_size, j*args.patch_size+noise_j:j*args.patch_size+noise_j+args.patch_size]
                # hex_patch = to_int(patch)
                hex_patch = to_256_colr(patch)
                patch_list.append(hex_patch.detach().cpu().numpy())

    save_chunk = f"{ ROOT }/preprocessData/chunks/artificial/{ args.data }"
    if not os.path.isdir(save_chunk):
        os.makedirs(save_chunk)
    save_coor = f"{ ROOT }/preprocessData/coordinate/artificial/{ args.data }"
    if not os.path.isdir(save_coor):
        os.makedirs(save_coor)

    save_i = 'left_i.pickle'
    save_j = 'left_j.pickle'

    with open( f"{ save_chunk }/chunks_{ args.data }_train.pickle", 'wb') as write:
        print(np.array(patch_list).shape)
        pickle.dump(patch_list, write)

    with open(f"{ save_coor }/{ save_i }", 'wb') as write:
        print(np.array(patch_i).shape)
        pickle.dump(patch_i, write)

    with open(f"{ save_coor }/{ save_j }", 'wb') as write:
        print(np.array(patch_j).shape)
        pickle.dump(patch_j, write)
