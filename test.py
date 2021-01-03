from PIL import Image
import collections
# from skimage.measure import structural_similarity as ssim
import matplotlib.pyplot as plt
import numpy as np
import cv2
from numpy import asarray
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import torch
import pickle
from torchvision import transforms
import torch.nn as nn

import preprocess.pretrain_vgg as pretrain_vgg


# """ Compare two images """

# im = Image.open("map/bottle/bad_small/errorMap_0_0.png")
# pixels = im.getdata()
# pixels_ls = list(im.getdata())

# im2 = Image.open("map/bottle/bad_small/errorMap_0_1.png")
# pixels2 = im2.getdata()
# pixels2_ls = list(im2.getdata())



def compare_pixel():
    if collections.Counter(pixels) == collections.Counter(pixels2):
        print ("The lists pixels and pixels2 are the same") 
    else: 
        print ("The lists pixels and pixels2 are not the same") 


def difference():
    pixels_arr = asarray(pixels).reshape(1000, -1)
    pixels2_arr = asarray(pixels2).reshape(1000, -1)
    ironman_grid = plt.GridSpec(1, 1)
    fig = plt.figure(figsize=(5, 5), dpi=100)
    ax = fig.add_subplot(ironman_grid[:,:2])
    im = ax.imshow((pixels_arr-pixels2_arr).transpose(1,0), cmap="Oranges")
    plt.colorbar(im, extend='both')
    im.set_clim(0, 10) # 這邊設定差值的上下界


    pathToSave = 'test_diff.png'
    plt.savefig(pathToSave, dpi=100)
    plt.close(fig)


from tqdm import tqdm

""" Add ground truth label for image """ 

def add_label():
    train_path = './dataset/bottle/train_resize/'
    train_dataset = pretrain_vgg.MvtecLoader(train_path)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)

    # label_name = 'label/bottle_trainK_16.pth'
    label_name = 'label/vgg19/bottle/train/16_100.pth'
    index_label = torch.load(label_name)

    for idx, img in tqdm(train_loader):
        idx = idx[0].item()
        ironman_grid = plt.GridSpec(1, 1)
        fig = plt.figure(figsize=(5,5), dpi=100)
        ax = fig.add_subplot(ironman_grid[:,:])

        im = ax.imshow(img.squeeze().cpu().detach().numpy().transpose(1,2,0), cmap="Blues")
        for i in range(32):
            for j in range(32):
                label = index_label[idx][i*32+j]
                label = label[0]
                ax.text(j*8, (i+0.5)*8, label, fontsize=5)
        
        
        pathToSave = './corn_ground_truth/label10K_' + str(idx) + '.png'
        plt.savefig(pathToSave, dpi=100)
        plt.close(fig)
# add_label()
#---------------------------------------#
def compute_var_white():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    split_size = 8

    train_path = './dataset/bottle/train_resize/'
    train_dataset = pretrain_vgg.MvtecLoader(train_path)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)

    model = pretrain_vgg.model
    model = model.to(device)
    white = []

    for idx, img in tqdm(train_loader):
        idx = idx[0].item()

        for x in range(32):
            patch = img[:, :, x*split_size : x*split_size+split_size, 0:8].to(device)
            crop_output = model.forward(patch)
            crop_output = crop_output.squeeze()
            # print(crop_output.size())
            white.append(crop_output.cpu().detach().numpy())
            
            patch = img[:, :, x*split_size : x*split_size+split_size, 248:256].to(device)
            crop_output = model.forward(patch)
            crop_output = crop_output.squeeze()

            white.append(crop_output.cpu().detach().numpy())
        
        for y in range(32):
            patch = img[:, :, 0:8, y*split_size : y*split_size+split_size].to(device)
            crop_output = model.forward(patch)
            crop_output = crop_output.squeeze()
            white.append(crop_output.cpu().detach().numpy())
            
            patch = img[:, :, 248:256, y*split_size : y*split_size+split_size].to(device)
            crop_output = model.forward(patch)
            crop_output = crop_output.squeeze()
            white.append(crop_output.cpu().detach().numpy())
        

    print(f'size of white patch: {len(white)}')

    # print(white[0].shape)
    
    # print(f'variance of white: {np.var(white, axis=0)}')

    draw_hist(np.var(white, axis=0), 'var_white.png')            


pil_to_tensor = transforms.ToTensor()

def compute_dis_white():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    split_size = 8

    train_path = './dataset/bottle/train_resize/'
    train_dataset = pretrain_vgg.MvtecLoader(train_path)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)

    kmeans_path = 'kmeans_bottle_16.pickle'
    kmeans = pickle.load(open(kmeans_path, "rb"))

    model = pretrain_vgg.model
    model = model.to(device)
    white = []
    white_l1 = []

    output_center = kmeans.cluster_centers_[13]
    output_center = np.reshape(output_center, (1, -1))
    output_center = pil_to_tensor(output_center).to(device)
    output_center = torch.squeeze(output_center)

    
    for idx, img in tqdm(train_loader):
        idx = idx[0].item()

        for x in range(32):
            patch = img[:, :, x*split_size : x*split_size+split_size, 0:8].to(device)
            crop_output = model.forward(patch)
            crop_output = torch.squeeze(crop_output)
            diff = nn.MSELoss()(output_center, crop_output)
            white.append(diff.item() ** 0.5)
            diff2 = nn.L1Loss()(output_center, crop_output)
            white_l1.append(abs(diff2.item()))


            # out = crop_output.cpu().detach().numpy().reshape((-1, 256))
            # img_idx = kmeans.predict(out)
            # if img_idx[0] != 13:
            #     print("out")

            
            patch = img[:, :, x*split_size : x*split_size+split_size, 248:256].to(device)
            crop_output = model.forward(patch)
            crop_output = torch.squeeze(crop_output)
            diff = nn.MSELoss()(output_center, crop_output)
            white.append(diff.item() ** 0.5)
            diff2 = nn.L1Loss()(output_center, crop_output)
            white_l1.append(abs(diff2.item()))

            # out = crop_output.cpu().detach().numpy().reshape((-1, 256))
            # img_idx = kmeans.predict(out)
            # if img_idx[0] != 13:
            #     print("out")

        
        for y in range(32):
            patch = img[:, :, 0:8, y*split_size : y*split_size+split_size].to(device)
            crop_output = model.forward(patch)
            crop_output = torch.squeeze(crop_output)
            diff = nn.MSELoss()(output_center, crop_output)
            white.append(diff.item() ** 0.5)
            diff2 = nn.L1Loss()(output_center, crop_output)
            white_l1.append(abs(diff2.item()))

            # out = crop_output.cpu().detach().numpy().reshape((-1, 256))
            # img_idx = kmeans.predict(out)
            # if img_idx[0] != 13:
            #     print("out")

            
            patch = img[:, :, 248:256, y*split_size : y*split_size+split_size].to(device)
            crop_output = model.forward(patch)
            crop_output = torch.squeeze(crop_output)
            diff = nn.MSELoss()(output_center, crop_output)
            white.append(diff.item() ** 0.5)
            diff2 = nn.L1Loss()(output_center, crop_output)
            white_l1.append(abs(diff2.item()))

            # out = crop_output.cpu().detach().numpy().reshape((-1, 256))
            # img_idx = kmeans.predict(out)
            # if img_idx[0] != 13:
            #     print("out")
    
    print(f'size of white patch: {len(white)}')

    # print(white[0].shape)
    
    # print(f'mean of white: {np.mean(white)}, variance of white: {np.var(white)}')
    # print(f'mean of whiteL1: {np.mean(white_l1)}, variance of whiteL1: {np.var(white_l1)}')

    plt.hist(white)
    plt.savefig('diff_white.png')
    # draw_hist(white, 'diff_white.png')


def compute_kmeans():
    kmeans_path = 'kmeans_bottle_16.pickle'
    kmeans = pickle.load(open(kmeans_path, "rb"))

    for i in range(16):
        print(f'mean of kmeans: {np.mean(kmeans.cluster_centers_[i])}, var of kmeans: {np.var(kmeans.cluster_centers_[i])}')


def compute_var_black():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    split_size = 8

    train_path = './dataset/bottle/train_resize/'
    train_dataset = pretrain_vgg.MvtecLoader(train_path)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)

    model = pretrain_vgg.model
    model = model.to(device)
    white = []

    for idx, img in tqdm(train_loader):
        idx = idx[0].item()

        for y in range(8):
            for x in range(8):
                    patch = img[:, :, 96+x*split_size : 96+x*split_size+split_size, 96+y*split_size : 96+y*split_size+split_size].to(device)
                    crop_output = model.forward(patch)
                    crop_output = crop_output.squeeze()
                    # print(crop_output.size())
                    white.append(crop_output.cpu().detach().numpy())
                    
        
    
    print(f'size of white patch: {len(white)}')

    # print(white[0].shape)
    
    # print(f'variance of white: {np.var(white, axis=0)}')

    draw_hist(np.var(white, axis=0), './var_black.png')


def compute_dis_black():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    split_size = 8

    train_path = './dataset/bottle/train_resize/'
    train_dataset = pretrain_vgg.MvtecLoader(train_path)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)

    kmeans_path = 'kmeans_bottle_16.pickle'
    kmeans = pickle.load(open(kmeans_path, "rb"))

    model = pretrain_vgg.model
    model = model.to(device)
    white = []
    white_l1 = []

    output_center = kmeans.cluster_centers_[13]
    output_center = np.reshape(output_center, (1, -1))
    output_center = pil_to_tensor(output_center).to(device)
    output_center = torch.squeeze(output_center)

    
    for idx, img in tqdm(train_loader):
        idx = idx[0].item()
        for y in range(8):
            for x in range(8):
                patch = img[:, :, 96+x*split_size : 96+x*split_size+split_size, 96+y*split_size : 96+y*split_size+split_size].to(device)
                crop_output = model.forward(patch)
                crop_output = crop_output.squeeze()
                diff = nn.MSELoss()(output_center, crop_output)
                white.append(diff.item() ** 0.5)
                diff2 = nn.L1Loss()(output_center, crop_output)
                white_l1.append(abs(diff2.item()))
                # print(crop_output.size())


    print(f'size of white patch: {len(white)}')

    # print(white[0].shape)
    
    # print(f'mean of white: {np.mean(white)}, variance of white: {np.var(white)}')
    # print(f'mean of whiteL1: {np.mean(white_l1)}, variance of whiteL1: {np.var(white_l1)}')

    plt.hist(white)
    plt.savefig('diff_black.png')
    # draw_hist(white, 'diff_white.png')


def compute_dis_total():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    split_size = 8

    train_path = './dataset/bottle/train_resize/'
    train_dataset = pretrain_vgg.MvtecLoader(train_path)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
   
    label_name = 'label/bottle_trainK_16.pth'
    index_label = torch.tensor(torch.load(label_name))

    kmeans_path = 'kmeans/kmeans_bottle_16.pickle'
    kmeans = pickle.load(open(kmeans_path, "rb"))

    model = pretrain_vgg.model
    model = model.to(device)
    white = []
    white_l1 = []

    
    for idx, img in tqdm(train_loader):
        idx = idx[0].item()
        for y in range(32):
            for x in range(32):

                label = index_label[idx][y*32+x].to(device)
                label = label.item()
                output_center = kmeans.cluster_centers_[label]
                output_center = np.reshape(output_center, (1, -1))
                output_center = pil_to_tensor(output_center).to(device)
                output_center = torch.squeeze(output_center)

                patch = img[:, :, x*split_size : x*split_size+split_size, y*split_size : y*split_size+split_size].to(device)
                crop_output = model.forward(patch)
                crop_output = crop_output.squeeze()
                diff = nn.MSELoss()(output_center, crop_output)
                white.append(diff.item() ** 0.5)
                diff2 = nn.L1Loss()(output_center, crop_output)
                white_l1.append(abs(diff2.item()))
                # print(crop_output.size())

    print(f'size of white patch: {len(white)}')
    plt.hist(white)
    plt.savefig('diff_total_16_K.png')

    

def draw_hist(ls, path):
    x = []
    y = []
    for i, v in enumerate(ls):
        # print(i, v)
        x.append(i)
        y.append(v)
    plt.bar(x, y)
    plt.savefig(path)


from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.manifold import MDS
import sys

from config import ROOT

def kmeans_visualization():
    """
    這邊我多放了 mds 的降維
    但是因為 mds 本身計算量很大，並沒有辦法把我們所有資料都放進去，因此我只有放第一張圖片去作降維做視覺化當參考依據
    至於 tsne 的化調整了 perplexity 的參數
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    kmeans_path = f"{ ROOT }/preprocessData/kmeans/AE/bottle/4/AE_128.pickle"
    kmeans = pickle.load(open(kmeans_path, "rb"))

    feature_path = f"{ ROOT }/preprocessData/chunks/AE/bottle/4/chunks_bottle_train.pickle"
    feature = pickle.load(open(feature_path, "rb"))
    

    y = kmeans.predict(np.array(feature))
    pca = PCA(n_components=2)
    feature = pca.fit_transform(np.array(feature))

    print(feature.shape)

    plt.scatter(feature[:, 0], feature[:, 1], c=y, s=5, cmap=plt.cm.get_cmap('Spectral', 128))
    center =  kmeans.cluster_centers_
    center = pca.transform(center)
    print(center.shape)
    plt.scatter(center[:, 0], center[:, 1], c='black', s=10, alpha=0.5)
    # plt.colorbar()

    plt.savefig('./vis_bottle_ae_pca.png')

kmeans_visualization()

def PCA_loss():
    chunks_path = 'chunks/chunks_bottle_train.pickle'
    chunks = pickle.load(open(chunks_path, "rb"))

    feature_path = 'chunks/PCA/bottle_32_100_16.pickle'
    feature = pickle.load(open(feature_path, "rb"))
    
    pca_path = 'PCA/bottle/32_100_16.pickle'
    pca = pickle.load(open(pca_path, "rb"))

    feature_project = pca.inverse_transform(feature)
    
    loss = 0
    criterion = nn.CrossEntropyLoss()
    c_loss = 0
    for i in range(len(chunks[0])):
        chunk_arr = np.array(chunks[i])
        feature_arr = np.array(feature_project[i])
        loss += ((chunks[i] / feature_project[i]) - 1.0).mean()

    loss /= len(chunks[0])
    print(loss)

# PCA_loss()

def normalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import scale
def norm_test():

    data = np.array([[3., -1., 2., 613.],
                    [2., 0., 0., 232],
                    [0., 1., -1., 113],
                    [1., 2., -3., 489]])

    min_max = MinMaxScaler()
    # sc = scale()
    data_minmax = min_max.fit_transform(data)
    data_sc = scale(data)
    print(data_minmax)
    print(data_sc)
    for i in range(data.shape[1]):
        data[:,i] = normalizeData(data[:, i])
    print(data)

# norm_test()


def check_training_label():

    dataset = pretrain_vgg.MvtecLoader('./dataset/bottle/train_resize/')
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    label_name = 'label/vgg19/bottle/train/16_100.pth'
    label = torch.load(label_name)
    # print(len(label))
    # print(len(label[0]))
    for idx, img in tqdm(loader):
        idx = idx[0].item()
        img = np.squeeze(img.detach().cpu().numpy()).transpose((1,2,0)) 
        ironman_grid = plt.GridSpec(1, 1)
        fig = plt.figure(figsize=(6,6), dpi=100)
        ax1 = fig.add_subplot(ironman_grid[0,0])
        im1 = ax1.imshow(img, cmap="Blues")

        """ add label text to each patch """ 
        for i in range(32):
            for j in range(32):
                # print(label[idx][i*32+j])
                ax1.text((j+0.2)*8, (i+0.6)*8, label[idx][i*32+j].item(), fontsize=5)
        pathToSave = './trainLabelMap/' + str(idx) + '.png'
        plt.savefig(pathToSave, dpi=100)
        plt.close(fig) 


# check_training_label()

