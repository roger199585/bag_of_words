import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn import preprocessing
import torchvision.models as models
from torchvision import transforms
import os
from PIL import Image
from tqdm import tqdm
import pretrain_vgg
import resnet
import argparse
import pickle
import cv2
from visualize import errorMap
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import random
import sys
import dataloaders
import model_weightSample
from sklearn.metrics import roc_auc_score


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pil_to_tensor = transforms.ToTensor()

def draw_errorMap(pretrain_model, model, test_loader, mask_loader, kmeans, pca, test_data, test_type):

    with torch.no_grad():  
        auc_list = [] 
        for ((idx, img), (idx2, img2)) in zip(test_loader, mask_loader):
            each_pixel_list = [[[] for i in range(1024)] for j in range(1024)]
            pixel_feature = []  
            img = img.to(device)
            idx = idx[0].item()
            print(f'eval phase: img idx={idx}')

            """ slide window = 16 """
            map_num = int((1024 - 64) / 16 + 1)   ## = 61
            
            for i in range(map_num):
                print(f'map_num:{i}')
                xs = []
                ys = []

                crop_list = []
                loss = 0.0
                loss_alpha = 0.0

                for j in range(map_num):
                    crop_img = img[:, :, i*16:i*16+64, j*16:j*16+64].to(device)
                    crop_output = pretrain_model(crop_img)
                    """ flatten the dimension of H and W """
                    out_ = crop_output.flatten(1,2).flatten(1,2)
                    out = pca.transform(out_.detach().cpu().numpy())
                    out_label = kmeans.predict(out)
                    out = pil_to_tensor(out).squeeze().to(device)

                    mask = torch.ones(1, 1, 1024, 1024)
                    mask[:, :, i*16:i*16+64, j*16:j*16+64] = 0
                    mask = mask.to(device)
                    x = img * mask
                    x = torch.cat((x, mask), 1)
                                        
                    output = model(x)
                    y = output.argmax(-1).detach().cpu().numpy()

                    output_center = kmeans.cluster_centers_[y]
                    output_center = np.reshape(output_center, (1, -1))
                    output_center = pil_to_tensor(output_center).to(device)
                    output_center = torch.squeeze(output_center)

                    # print(y, out_label)
                    if y == out_label:
                        isWrongLabel = 0
                    else:
                        isWrongLabel = 1
                    
                    # print(isWrongLabel)
                    # sys.exit(0)
                    diff = isWrongLabel * nn.MSELoss()(output_center, out)
                    
                    for x in range(64):
                        for y in range(64):
                            each_pixel_list[i*16+x][j*16+y].append(diff.item())
            
            for m in range(1024):
                for n in range(1024):
                    pixel_feature.append(sum(each_pixel_list[m][n]) / len(each_pixel_list[m][n]))

            pixel_feature = np.array(pixel_feature).reshape((1024, -1))

            if (test_type == 'good'):
                img_ = np.squeeze(img.detach().cpu().numpy()).transpose((1,2,0))
                ironman_grid = plt.GridSpec(1, 2)
                fig = plt.figure(figsize=(12,6), dpi=100)
                ax1 = fig.add_subplot(ironman_grid[0,0])
                im1 = ax1.imshow(pixel_feature, cmap="Blues")
                ax2 = fig.add_subplot(ironman_grid[0,1])
                im2 = ax2.imshow(img_)
            else:
                img_ = np.squeeze(img.detach().cpu().numpy()).transpose((1,2,0))
                defect_gt = np.squeeze(img2.cpu().numpy()).transpose((1,2,0))
                ironman_grid = plt.GridSpec(1, 3)
                fig = plt.figure(figsize=(18,6), dpi=100)
                ax1 = fig.add_subplot(ironman_grid[0,0])
                im1 = ax1.imshow(pixel_feature, cmap="Blues")
                ax2 = fig.add_subplot(ironman_grid[0,1])
                ax3 = fig.add_subplot(ironman_grid[0,2])
                im2 = ax2.imshow(img_)
                im3 = ax3.imshow(defect_gt)

                true_mask = defect_gt[:, :, 0].astype('int32') 
                auc = roc_auc_score(true_mask.flatten(), pixel_feature.flatten())

                auc_list.append(auc)
                ax1.set_title(auc)

            # errorMapPath = "multi_errorMap/vgg19/{}/{}/".format(
            #     test_data,
            #     test_type
            # )
            errorMapPath = "testing_multiMap/"
            if not os.path.isdir(errorMapPath):
                os.makedirs(errorMapPath)
                print("----- create folder for type:{} -----".format(test_type))
            
            errorMapName = "{}.png".format(
                str(idx)
            )

            plt.savefig(errorMapPath+errorMapName, dpi=100)
            plt.close(fig)
            
            print(f'img_idx={idx}')
        
        auc_average = sum(auc_list) / len(auc_list)
        print("Average score: ", auc_average)


if __name__ == "__main__":

    """ set parameters """
    parser = argparse.ArgumentParser()
    parser.add_argument('--kmeans', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--data', type=str, default='bottle')
    parser.add_argument('--type', type=str, default='good')
    parser.add_argument('--batch', type=int, default=100)
    parser.add_argument('--dim', type=int, default=16)
    parser.add_argument('--model', type=str, default='vgg19')
    parser.add_argument('--train_batch', type=int, default=32)
    args = parser.parse_args()

    out = args.kmeans
    test_good_path = "./dataset/{}/test/good_resize".format(args.data)

    kmeans_path = "kmeans/{}/{}_{}_{}_{}.pickle".format(
        args.data,
        str(args.model),
        str(args.kmeans),
        str(args.batch),
        str(args.dim)
    )

    pca_path = "PCA/{}/{}_{}_{}_{}.pickle".format(
        args.data, 
        str(args.model), 
        str(args.kmeans), 
        str(args.batch), 
        str(args.dim)
    )

    kmeans = pickle.load(open(kmeans_path, "rb"))
    pca = pickle.load(open(pca_path, "rb"))

    if (args.type == 'good'):
        test_path = test_good_path
        test_label_name = "label/{}/{}/test/good_{}_{}.pth".format(
            str(args.model),
            args.data,
            str(out),
            str(args.batch)
        )
        defect_gt_path = "dataset/bottle/ground_truth/broken_small_resize/"


    else:
        test_path = "./dataset/{}/test/{}_resize".format(args.data, args.type)
        test_label_name = "label/{}/{}/test/{}_{}_{}.pth".format(
            str(args.model),
            args.data,
            args.type,
            str(out),
            str(args.batch)
        )
        defect_gt_path = "dataset/{}/ground_truth/{}_resize/".format(args.data, args.type)
        test_label = torch.tensor(torch.load(test_label_name))

    test_dataset = dataloaders.MvtecLoader(test_path)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    mask_dataset = dataloaders.MaskLoader(defect_gt_path)
    mask_loader = DataLoader(mask_dataset, batch_size=1, shuffle=False)

    model = model_weightSample.scratch_model
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load('models/{}/{}/exp1_128_30.ckpt'.format(
        args.model, 
        args.data
        )   
    ))
        
    model = model.to(device)

    pretrain_model = nn.DataParallel(pretrain_vgg.model).to(device)
    
    if (args.type == 'good'):
        draw_errorMap(pretrain_model, model, test_loader, test_loader, kmeans, pca, args.data, args.type)

    else:    
        draw_errorMap(pretrain_model, model, test_loader, mask_loader, kmeans, pca, args.data, args.type)
