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
from sklearn import metrics
# import model_weightSample
from sklearn.metrics import roc_auc_score
import time
import itertools

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pil_to_tensor = transforms.ToTensor()

def pROC(y_true, y_pred):
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred)

    nearestIndex = np.argmin(abs(fpr - 0.3))
    
    tpr = tpr[:nearestIndex]
    fpr = fpr[:nearestIndex]

    area = 0
    for index in range(1, nearestIndex):
        width1 = tpr[index] - fpr[index]
        width2 = tpr[index - 1] - fpr[index - 1]
        height = fpr[index] - fpr[index]

        area += (width1 + width2) * height / 2
    return area

def norm(feature):
    return feature / feature.max()

def eval_feature(pretrain_model, model, test_loader, kmeans, pca, test_data, global_index, good=False):

    global label_pred
    global label_true

    model.eval()
    pretrain_model.eval()

    with torch.no_grad():

        img_feature = []
        
        start = time.time()

        for (idx, img) in test_loader:
            each_pixel_err_sum = np.zeros([1024, 1024])
            each_pixel_err_count = np.zeros([1024, 1024])

            # pixel_feature = []  
            img = img.to(device)
            idx = idx[0].item()
            
            print(f'eval phase: img idx={idx}')

            """ slide window = 16 """
            map_num = int((1024 - 64) / 16 + 1)   ## = 61
            indices = list(itertools.product(range(map_num), range(map_num)))
            
            """ batch """
            batch_size = 32

            label_idx = []
            label_gt = []

            for batch_start_idx in range(0, len(indices), batch_size):
                xs = []
                ys = []
                crop_list = []

                batch_idxs = indices[batch_start_idx:batch_start_idx+batch_size]

                for i, j in batch_idxs:
                    crop_img = img[:, :, i*16:i*16+64, j*16:j*16+64].to(device)
                    crop_output = pretrain_model(crop_img)
                    """ flatten the dimension of H and W """
                    out_ = crop_output.flatten(1,2).flatten(1,2)
                    out = pca.transform(out_.detach().cpu().numpy())
                    out_label = kmeans.predict(out)
                    out_label = torch.from_numpy(out_label).to(device)
                    out = pil_to_tensor(out).squeeze().to(device)
                    crop_list.append(out)

                    mask = torch.ones(1, 1, 1024, 1024)
                    mask[:, :, i*16:i*16+64, j*16:j*16+64] = 0
                    mask = mask.to(device)
                    x = img * mask
                    x = torch.cat((x, mask), 1)

                    xs.append(x)
                    ys.append(out_label)

                x = torch.cat(xs, 0)
                y = torch.stack(ys).squeeze().to(device)                        
                output = model(x)
                y_ = output.argmax(-1).detach().cpu().numpy()

                for n, (i, j) in enumerate(batch_idxs):
                    output_center = kmeans.cluster_centers_[y_[n]]
                    output_center = np.reshape(output_center, (1, -1))
                    output_center = pil_to_tensor(output_center).to(device)
                    output_center = torch.squeeze(output_center)

                    isWrongLabel = int(y_[n] != y[n].item())
                    diff = isWrongLabel * nn.MSELoss()(output_center, crop_list[n])
                    
                    each_pixel_err_sum[i*16:i*16+64, j*16:j*16+64] += diff.item()
                    each_pixel_err_count[i*16:i*16+64, j*16:j*16+64] += 1

            # """ batch = 61 """ 
            # for i in range(map_num):
            #     print(f'map_num:{i}')
            #     xs = []
            #     ys = []
            #     crop_list = []

            #     for j in range(map_num):
            #         crop_img = img[:, :, i*16:i*16+64, j*16:j*16+64].to(device)
            #         crop_output = pretrain_model(crop_img)
            #         """ flatten the dimension of H and W """
            #         out_ = crop_output.flatten(1,2).flatten(1,2)
            #         out = pca.transform(out_.detach().cpu().numpy())
            #         out_label = kmeans.predict(out)
            #         out_label = torch.from_numpy(out_label).to(device)
            #         out = pil_to_tensor(out).squeeze().to(device)
            #         crop_list.append(out)

            #         mask = torch.ones(1, 1, 1024, 1024)
            #         mask[:, :, i*16:i*16+64, j*16:j*16+64] = 0
            #         mask = mask.to(device)
            #         x = img * mask
            #         x = torch.cat((x, mask), 1)

            #         xs.append(x)
            #         ys.append(out_label)
                
            #     x = torch.cat(xs, 0)
            #     y = torch.stack(ys).squeeze().to(device)                        
            #     output = model(x)
            #     y_ = output.argmax(-1).detach().cpu().numpy()

            #     for k in range(map_num):
            #         output_center = kmeans.cluster_centers_[y_[k]]
            #         output_center = np.reshape(output_center, (1, -1))
            #         output_center = pil_to_tensor(output_center).to(device)
            #         output_center = torch.squeeze(output_center)

            #         isWrongLabel = int(y_[k] != y[k].item())
            #         diff = isWrongLabel * nn.MSELoss()(output_center, crop_list[k])
                    
            #         for a in range(64):
            #             for b in range(64):
            #                 each_pixel_list[i*16+a][k*16+b].append(diff.item())
            
            pixel_feature = each_pixel_err_sum / each_pixel_err_count

            # for m in range(1024):
            #     for n in range(1024):
            #         pixel_feature.append(sum(each_pixel_list[m][n]) / len(each_pixel_list[m][n]))

            # pixel_feature = np.array(pixel_feature).reshape((1024, -1))
            img_feature.append(pixel_feature)

    print(np.array(img_feature).shape)
    img_feature = np.array(img_feature).reshape((len(test_loader), -1))
    return img_feature
    
if __name__ == "__main__":

    """ set parameters """
    parser = argparse.ArgumentParser()
    parser.add_argument('--kmeans', type=int, default=128)
    parser.add_argument('--data', type=str, default='bottle')
    parser.add_argument('--index', type=int, default=30)
    parser.add_argument('--resume', type=bool, default=False)
    args = parser.parse_args()

    global_index = args.index
    test_data = args.data
    
    scratch_model = nn.Sequential(
        resnet.resnet18(pretrained=False, num_classes=args.kmeans)
    )
    scratch_model = nn.DataParallel(scratch_model).cuda()
    scratch_model.load_state_dict(torch.load('models/vgg19/{}/exp1_{}_{}.ckpt'.format(args.data, args.kmeans, global_index)))


    ### DataSet for all defect type
    test_all_path = "/home/dinosaur/bag_of_words/dataset/{}/test_resize/all/".format(args.data)
    test_all_dataset = dataloaders.MvtecLoader(test_all_path)
    test_all_loader = DataLoader(test_all_dataset, batch_size=1, shuffle=False)

    test_good_path = "/home/dinosaur/bag_of_words/dataset/{}/test_resize/good/".format(args.data)
    test_good_dataset = dataloaders.MvtecLoader(test_good_path)
    test_good_loader = DataLoader(test_good_dataset, batch_size=1, shuffle=False)

    mask_path = "dataset/{}/ground_truth_resize/all/".format(args.data)
    mask_dataset = dataloaders.MaskLoader(mask_path)
    mask_loader = DataLoader(mask_dataset, batch_size=1, shuffle=False)

    ## Models
    pretrain_model = nn.DataParallel(pretrain_vgg.model).cuda()

    ## Clusters
    kmeans_path = "preprocessData/kmeans/{}/vgg19_{}_100_16.pickle".format(args.data, args.kmeans)
    kmeans = pickle.load(open(kmeans_path, "rb"))

    pca_path = "preprocessData/PCA/{}/vgg19_{}_100_16.pickle".format(args.data, args.kmeans)
    pca = pickle.load(open(pca_path, "rb"))

    ## Label
    test_all_label_name = "preprocessData/label/vgg19/{}/test/all_{}_100.pth".format(args.data, args.kmeans)
    test_all_label = torch.tensor(torch.load(test_all_label_name))

    test_good_label_name = "preprocessData/label/vgg19/{}/test/good_{}_100.pth".format(args.data, args.kmeans)
    test_good_label = torch.tensor(torch.load(test_good_label_name))
    
    print("----- defect -----")
    if args.resume and os.path.isfile('testing_multiMap/{}/all/img_all_feature_{}.pickle'.format(args.data, args.index)):
        print("load from testing_multiMap/{}/all/img_all_feature_{}.pickle".format(args.data, args.index))
        img_all_feature = pickle.load(open('testing_multiMap/{}/all/img_all_feature_{}.pickle'.format(args.data, args.index), 'rb'))
    else:
        img_all_feature = eval_feature(pretrain_model, scratch_model, test_all_loader, kmeans, pca, args.data, global_index, good=False)

    print("----- good -----")
    if args.resume and os.path.isfile('testing_multiMap/{}/good/img_good_feature_{}.pickle'.format(args.data, args.index)):
        print("load from testing_multiMap/{}/good/img_good_feature_{}.pickle".format(args.data, args.index))
        img_all_feature = pickle.load(open('testing_multiMap/{}/good/img_good_feature_{}.pickle'.format(args.data, args.index), 'rb'))
    else:
        img_good_feature = eval_feature(pretrain_model, scratch_model, test_good_loader, kmeans, pca, args.data, global_index, good=True)
    
    """ save feature """ 
    
    save_all_path = "testing_multiMap/{}/all/".format(args.data)
    if not os.path.isdir(save_all_path):
        os.makedirs(save_all_path)
    save_all_name = "img_all_feature_{}.pickle".format(args.index)
    pickle.dump(img_all_feature, open(save_all_path+save_all_name, "wb"))
    
    save_good_path = "testing_multiMap/{}/good/".format(args.data)
    if not os.path.isdir(save_good_path):
        os.makedirs(save_good_path)
    save_good_name = "img_good_feature_{}.pickle".format(args.index)
    pickle.dump(img_good_feature, open(save_good_path+save_good_name, "wb"))


    label_pred = []
    label_true = []

    """ for defect type """ 
    for ((idx, img), (idx2, img2)) in zip(test_all_loader, mask_loader):
        img = img.cuda()
        idx = idx[0].item()


        errorMap = img_all_feature[idx].reshape((1024, 1024))
        # """  draw errorMap """
        # img_ = np.squeeze(img.detach().cpu().numpy()).transpose((1,2,0))
        # defect_gt = np.squeeze(img2.cpu().numpy()).transpose((1,2,0))
        # ironman_grid = plt.GridSpec(1, 3)
        # fig = plt.figure(figsize=(18,6), dpi=100)
        # ax1 = fig.add_subplot(ironman_grid[0,0])
        # im1 = ax1.imshow(errorMap, cmap="Blues")
        # ax2 = fig.add_subplot(ironman_grid[0,1])
        # ax3 = fig.add_subplot(ironman_grid[0,2])
        # im2 = ax2.imshow(img_)
        # im3 = ax3.imshow(defect_gt)
        # errorMapPath = "testing_multiMap/{}/all/{}/".format(test_data, args.kmeans)
        # if not os.path.isdir(errorMapPath):
        #     os.makedirs(errorMapPath)
        #     print("----- create folder for type: all -----")
        
        # errorMapName = "{}_{}.png".format(
        #     str(idx),
        #     str(global_index)
        # )

        # plt.savefig(errorMapPath+errorMapName, dpi=100)
        # plt.close(fig)

        """ for computing aucroc score """
        defect_gt = np.squeeze(img2.cpu().numpy()).transpose(1,2,0)
        true_mask = defect_gt[:, :, 0].astype('int32')
        label_pred.append(errorMap)
        label_true.append(true_mask)    
        print(f'EP={global_index} defect_img_idx={idx}')

        

    """ for good type """
    for (idx, img) in test_good_loader:
        img = img.cuda()
        idx = idx[0].item()
        
        errorMap = img_good_feature[idx].reshape((1024, 1024))
        
        # """ draw errorMap """
        # img_ = np.squeeze(img.detach().cpu().numpy()).transpose((1,2,0))
        # ironman_grid = plt.GridSpec(1, 2)
        # fig = plt.figure(figsize=(12,6), dpi=100)
        # ax1 = fig.add_subplot(ironman_grid[0,0])
        # im1 = ax1.imshow(errorMap, cmap="Blues")
        # ax2 = fig.add_subplot(ironman_grid[0,1])
        # im2 = ax2.imshow(img_)
        
        # errorMapPath = "testing_multiMap/{}/good/{}/".format(test_data, args.kmeans)
        # if not os.path.isdir(errorMapPath):
        #     os.makedirs(errorMapPath)
        #     print("----- create folder for type: good -----")        

        # errorMapName = "{}_{}.png".format(
        #     str(idx),
        #     str(global_index)
        # )

        # plt.savefig(errorMapPath+errorMapName, dpi=100)
        # plt.close(fig)

        """ for computing aucroc score """
        defect_gt = np.zeros((1024, 1024, 3))
        true_mask = defect_gt[:, :, 0].astype('int32')
        label_pred.append(errorMap)
        label_true.append(true_mask)    
        print(f'EP={global_index} good_img_idx={idx}')

    label_pred = norm(np.array(label_pred))
    print(label_pred.shape)
    print(np.array(label_true).shape)
    # auc = roc_auc_score(np.array(label_true).flatten(), label_pred.flatten())
    proc = pROC(np.array(label_true).flatten(), label_pred.flatten())

    f = open("score.txt", "a")
    f.write("pROC score for testing data {}: {}".format(args.data, proc))
    f.close()
    # print("AUC score for testing data {}: {}".format(args.data, auc))

