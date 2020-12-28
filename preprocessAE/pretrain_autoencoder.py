"""
    Author: Yong Yu Chen
    Collaborator: Corn

    Update: 2020/12/24
    History: 
        2020/12/24 -> Prprocess by pretrain autoencoder 

    Description: Prprocess by pretrained autoencoder 
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
sys.path.append("../")
import dataloaders
from config import ROOT
import networks.autoencoder as autoencoder

""" Save chunks of training datas to fit the corresponding kmeans """
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    """ Set parameters """ 
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='bottle', help='category of dataset, EX: bottle, cable ...')
    parser.add_argument('--patch_size', type=int, default=64, help='Size of the patch you cut, default is 64')
    parser.add_argument('--image_size', type=int, default=1024, help='Size of your origin image')
    parser.add_argument('--resolution', type=int, default=4, help='Dimension of the autoencoder\'s latent code resolution')
    args = parser.parse_args()

    print('data: ', args.data)
    print('patch size: ', args.patch_size)
    print('image size: ', args.image_size)
    
    """ Load part of pretrained model """
    model =  autoencoder.autoencoder(3, args.resolution)
    model.load_state_dict(torch.load(f"{ ROOT }/models/AE/{ args.data }_{ args.resolution }/40000.ckpt"))
    model = model.to(device)

    patch_list = []
    patch_i = []
    patch_j = []

    """ Load dataset """
    train_dataset = dataloaders.MvtecLoader( f"{ ROOT }/dataset/{ args.data }/train_resize/good/" )
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)

    for idx, img in tqdm(train_loader):
        model.eval()
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
                output, latent_code = model.forward(img_)
                """ flatten the dimension of H and W """
                # out_ = output.flatten(1,2).flatten(1,2).squeeze()
                out_ = latent_code.flatten(1,2).flatten(1,2).squeeze()
                patch_list.append(out_.detach().cpu().numpy())

    save_chunk = f"{ ROOT }/preprocessData/chunks/AE/{ args.data }/{ args.resolution }"
    if not os.path.isdir(save_chunk):
        os.makedirs(save_chunk)
    save_coor = f"{ ROOT }/preprocessData/coordinate/AE/{ args.data }/{ args.resolution }"
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
