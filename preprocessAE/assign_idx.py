"""
    Author: Corn

    Update: 2020/12/28
    History: 
        2020/12/28 -> generate ground truth index

    Description: 製造每個 patch 的 ground truth
"""

""" STD Library """
import os
import sys
import pickle
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm

""" Pytorch Library """
import torch
from torch.utils.data import DataLoader

""" Custom Library """
import dataloaders
import networks.autoencoder as autoencoder
from config import ROOT

def save_img(img, save_name):
    if not os.path.isdir( f'{ ROOT }/preprocessData/kmeans_img/AE/{ args.data }/{ args.resolution }/{ str(args.kmeans) }/'):
        os.makedirs(f'{ ROOT }/preprocessData/kmeans_img/AE/{ args.data }/{ args.resolution }/{ str(args.kmeans) }/')

    img = np.squeeze( img.detach().cpu().numpy()).transpose((1,2,0) )
    img = Image.fromarray( (img * 255).astype(np.uint8) )
    img.save(save_name)

if __name__ == "__main__":
    """ Set parameters """ 
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='bottle', help='category of dataset, EX: bottle, cable ...')
    parser.add_argument('--type', type=str, default='train', help='train, test or val')
    parser.add_argument('--kmeans', type=int, default=16, help='number of kmeans clusters')
    parser.add_argument('--patch_size', type=int, default=64, help='Size of the patch you cut, default is 64')
    parser.add_argument('--image_size', type=int, default=1024, help='Size of your origin image')
    parser.add_argument('--resolution', type=int, default=4, help='Dimension of the autoencoder\'s latent code resolution') # only 4 or 8
    args = parser.parse_args()

    """ Load preprocess datas """
    kmeans_path = f"{ ROOT }/preprocessData/kmeans/AE/{ args.data }/{ args.resolution }/AE_{ str(args.kmeans) }.pickle"
    left_i_path = f"{ ROOT }/preprocessData/coordinate/AE/{ args.data }/{ args.resolution }/left_i.pickle"
    left_j_path = f"{ ROOT }/preprocessData/coordinate/AE/{ args.data }/{ args.resolution }/left_j.pickle"

    kmeans = pickle.load(open(kmeans_path, "rb"))
    left_i = pickle.load(open(left_i_path, "rb"))
    left_j = pickle.load(open(left_j_path, "rb"))

    """ Check folder if not exists auto create"""
    if args.type == 'train':
        path      = f"{ ROOT }/dataset/{ args.data}/train_resize/good/"
        save_path = f"{ ROOT }/preprocessData/label/AE/{ args.data }/{ args.resolution }/train/{ str(args.kmeans) }.pth"

        if not os.path.isdir( f"{ ROOT }/preprocessData/label/AE/{ args.data }/{ args.resolution }/train/" ):
            os.makedirs( f"{ ROOT }/preprocessData/label/AE/{ args.data }/{ args.resolution }/train/" )
    elif args.type == 'test':
        path      = f"{ ROOT }/dataset/{ args.data }/test_resize/good/"
        save_path = f"{ ROOT }/preprocessData/label/AE/{ args.data }/{ args.resolution }/test/good_{ str(args.kmeans) }.pth"
        
        if not os.path.isdir( f"{ ROOT }/preprocessData/label/AE/{ args.data }/{ args.resolution }/test/"):
            os.makedirs( f"{ ROOT }/preprocessData/label/AE/{ args.data }/{ args.resolution }/test/")
    elif args.type == 'all':
        path      = f"{ ROOT }/dataset/{ args.data }/test_resize/all/"
        save_path = f"{ ROOT }/preprocessData/label/AE/{ args.data }/{ args.resolution }/test/all_{ str(args.kmeans) }.pth"
        
        if not os.path.isdir(f"{ ROOT }/preprocessData/label/AE/{ args.data }/{ args.resolution }/test/"):
            os.makedirs(f"{ ROOT }/preprocessData/label/AE/{ args.data }/{ args.resolution }/test/")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model =  autoencoder.autoencoder(3, args.resolution)
    model.load_state_dict(torch.load(f"{ ROOT }/models/AE/{ args.data }_{ args.resolution }/40000.ckpt"))
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

        chunk_num = int(args.image_size / args.patch_size)
        
        for i in range(chunk_num):
            for j in range(chunk_num):
                """ Crop the image """
                if (args.type == 'train'):
                    index = idx*chunk_num*chunk_num+i*chunk_num+j
                    patch = img[ :, :, i*args.patch_size+left_i[index]:i*args.patch_size+args.patch_size+left_i[index], j*args.patch_size+left_j[index]:j*args.patch_size+args.patch_size+left_j[index] ].to(device)
                else:
                    patch = img[:, :, i*args.patch_size:i*args.patch_size+args.patch_size, j*args.patch_size:j*args.patch_size+args.patch_size].to(device)
                _, latent_code = model.forward( patch )

                """ flatten the dimension of H and W """
                out = latent_code.flatten(1,2).flatten(1,2)
                out = out.detach().cpu().numpy()
                patch_idx = kmeans.predict(out)

                patch_index_list.append(patch_idx)

            if (args.type == 'train'):                
                save_img(patch, f'{ ROOT }/preprocessData/kmeans_img/AE/{ args.data }/{ args.resolution }/{ str(args.kmeans) }/idx_{ str(patch_idx.item()) }.png')
            
        img_index_list.append(patch_index_list)

    torch.save(img_index_list, save_path)

    if args.type == 'train':
        saveLabelPath = f"{ ROOT }/preprocessData/label/fullPatch/AE/{ args.data }/{ args.resolution }/kmeans_128.pth"
        if not os.path.isdir(f"{ ROOT }/preprocessData/label/fullPatch/AE/{ args.data }/{ args.resolution }/"):
            os.makedirs(f"{ ROOT }/preprocessData/label/fullPatch/AE/{ args.data }/{ args.resolution }/")
        torch.save(np.array(img_index_list).reshape(-1, 1), saveLabelPath)


    print(len(img_index_list), len(img_index_list[0]))
