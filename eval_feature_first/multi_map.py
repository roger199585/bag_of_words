"""
    Author: Yong Yu Chen
    Collaborator: Corn

    Update: 2021/1/11
    History: 
        2021/1/11 -> 修正計算速度很慢的問題(原因應該是 PCA 的部分每個 patch 就做一次，其實可以將 patch 全部搜集起來，再一次降維)

    Description: Use multi error map to calculate auc roc score
"""
import os
import sys
import cv2
import time
import pickle
import random
import argparse
import itertools
import numpy as np
from tqdm import tqdm

""" Pytorch Library """
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader

""" sklearn Library """
from sklearn.metrics import roc_auc_score

""" Custom Library """
import networks.resnet as resnet
import preprocess_feature_first.pretrain_vgg as pretrain_vgg

import dataloaders
from config import ROOT

from ei import patch
patch(select=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pil_to_tensor = transforms.ToTensor()

def norm(feature):
    if feature.max() == 0:
        return feature
    else:
        return feature / feature.max()

def eval_OriginFeature(pretrain_model, model, test_loader, kmeans, pca, test_data, global_index, good=False):
    global label_pred
    global label_true

    model.eval()
    pretrain_model.eval()

    with torch.no_grad():
        img_feature = []
        
        start = time.time()

        chunk_num = int(args.image_size / args.patch_size)
        for (idx, img) in test_loader:
            sss = time.time()
            each_pixel_err_sum = np.zeros([1024, 1024])
            each_pixel_err_count = np.zeros([1024, 1024])

            img = img.to(device)
            idx = idx[0].item()
            
            print(f'eval phase: img idx={idx}')

            """ slide window = 16 """
            map_num = int((args.image_size - args.patch_size) / 16 + 1)   ## = 61
            indices = list(itertools.product(range(map_num), range(map_num)))
            
            """ batch """
            batch_size = 32

            label_idx = []
            label_gt = []

            for batch_start_idx in tqdm(range(0, len(indices), batch_size)):
                b_start = time.time()
                xs = []
                ys = []
                crop_list = []

                batch_idxs = indices[batch_start_idx:batch_start_idx+batch_size]

                patches = []

                crop_output = pretrain_model(img)
                for i, j in batch_idxs:
                    # crop_img = img[:, :, i*chunk_num:i*chunk_num+args.patch_size, j*chunk_num:j*chunk_num+args.patch_size].to(device)
                    # crop_output = pretrain_model(crop_img)
                    """ flatten the dimension of H and W """
                    if i % 4 == 0:
                        if j % 4 == 0:
                            out = crop_output[0, :, i // 4, j // 4]
                        else:
                            j1 = j // 4
                            j2 = j1 + 1

                            out1 = crop_output[0, :, i // 4, j1] * (1 - ((j % 4) / 4))
                            out2 = crop_output[0, :, i // 4, j2] * ((j % 4) / 4)

                            out = out1 + out2
                    else:
                        if j % 4 == 0:
                            i1 = i // 4
                            i2 = i1 + 1

                            out1 = crop_output[0, :, i1, j // 4] * (1 - ((i % 4) / 4))
                            out2 = crop_output[0, :, i2, j // 4] * ((i % 4) / 4)
                        else:
                            i1 = i // 4
                            i2 = i1 + 1
                            
                            j1 = j // 4
                            j2 = j1 + 1

                            out1 = crop_output[0, :, i1, j1] * (1 - ((j % 4) / 4))
                            out2 = crop_output[0, :, i1, j2] * ((j % 4) / 4)
                            out_1 = out1 + out2

                            out1 = crop_output[0, :, i2, j1] * (1 - ((j % 4) / 4))
                            out2 = crop_output[0, :, i2, j2] * ((j % 4) / 4)
                            out_2 = out1 + out2

                            out = out_1 * (1 - ((i % 4) / 4)) + out_2 * ((i % 4) / 4)


                    # out = crop_output[0, :, i, j]
                    patches.append(out.detach().cpu().numpy())
                    crop_list.append(out)

                    mask = torch.ones(1, 1, 1024, 1024)
                    mask[:, :, i*chunk_num:i*chunk_num+args.patch_size, j*chunk_num:j*chunk_num+args.patch_size] = 0
                    mask = mask.to(device)
                    x = img * mask
                    x = torch.cat((x, mask), 1)

                    xs.append(x)
                
                patches = np.array(patches)
                patches = patches.reshape(-1, patches.shape[-1])
                
                new_outs = pca.transform(patches)
                for i in range(new_outs.shape[0]):
                    out_label = kmeans.predict(new_outs[i].reshape(1, -1))
                    out_label = torch.from_numpy(out_label).to(device)
                    ys.append(out_label)

                x = torch.cat(xs, 0)
                y = torch.stack(ys).squeeze().to(device)                        
                output = model(x)
                print(aaaaaa)
                y_ = output.argmax(-1).detach().cpu().numpy()
                for n, (i, j) in enumerate(batch_idxs):
                    output_feature = np.expand_dims(cluster_features[y_[n]], axis=0)
                    output_feature = torch.from_numpy(output_feature).cuda()

                    isWrongLabel = int(y_[n] != y[n].item())
                    diff = isWrongLabel * nn.MSELoss()(output_feature.squeeze(), crop_list[n])
                    
                    each_pixel_err_sum[i*chunk_num:i*chunk_num+args.patch_size, j*chunk_num:j*chunk_num+args.patch_size] += diff.item()
                    each_pixel_err_count[i*chunk_num:i*chunk_num+args.patch_size, j*chunk_num:j*chunk_num+args.patch_size] += 1
            pixel_feature = each_pixel_err_sum / each_pixel_err_count
            img_feature.append(pixel_feature)
            print(f'spend {time.time() - sss}')


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
    parser.add_argument('--image_size', type=int, default=1024)
    parser.add_argument('--patch_size', type=int, default=64)
    parser.add_argument('--dim_reduction', type=str, default='PCA')
    args = parser.parse_args()

    global_index = args.index
    test_data = args.data
    
    scratch_model = nn.Sequential(
        resnet.resnet50(pretrained=False, num_classes=args.kmeans)
    )
    scratch_model = nn.DataParallel(scratch_model).cuda()
    scratch_model.load_state_dict(torch.load('{}/models/vgg19/{}/exp1_{}_{}.ckpt'.format(ROOT, args.data, args.kmeans, global_index)))


    ### DataSet for all defect type
    test_all_path = "{}/dataset/{}/test_resize/all/".format(ROOT, args.data)
    test_all_dataset = dataloaders.MvtecLoader(test_all_path)
    test_all_loader = DataLoader(test_all_dataset, batch_size=1, shuffle=False)

    test_good_path = "{}/dataset/{}/test_resize/good/".format(ROOT, args.data)
    test_good_dataset = dataloaders.MvtecLoader(test_good_path)
    test_good_loader = DataLoader(test_good_dataset, batch_size=1, shuffle=False)

    mask_path = "{}/dataset/{}/ground_truth_resize/all/".format(ROOT, args.data)
    mask_dataset = dataloaders.MaskLoader(mask_path)
    mask_loader = DataLoader(mask_dataset, batch_size=1, shuffle=False)

    ## Models
    pretrain_model = nn.DataParallel(pretrain_vgg.model).cuda()

    ## Clusters
    kmeans_path = "{}/preprocessData/kmeans/{}/{}/vgg19_{}_100_128.pickle".format(ROOT, args.dim_reduction, args.data, args.kmeans)
    kmeans = pickle.load(open(kmeans_path, "rb"))

    pca_path = "{}/preprocessData/{}/{}/vgg19_{}_100_128.pickle".format(ROOT, args.dim_reduction, args.data, args.kmeans)
    pca = pickle.load(open(pca_path, "rb"))

    ## Cluster Center Features
    center_features_path = "{}/preprocessData/cluster_center/128/{}.pickle".format(ROOT, args.data)
    cluster_features = pickle.load(open(center_features_path, "rb"))
    
    print("----- defect -----")
    if args.resume and os.path.isfile('{}/Results/testing_multiMap/{}/all/img_all_feature_{}.pickle'.format(ROOT, args.data, args.index)):
        print("load from {}/Results/testing_multiMap/{}/all/img_all_feature_{}.pickle".format(ROOT, args.data, args.index))
        img_all_feature = pickle.load(open('{}/Results/testing_multiMap/{}/all/img_all_feature_{}.pickle'.format(ROOT, args.data, args.index), 'rb'))
    else:
        img_all_feature = eval_OriginFeature(pretrain_model, scratch_model, test_all_loader, kmeans, pca, args.data, global_index, good=False)

    print("----- good -----")
    if args.resume and os.path.isfile('{}/Results/testing_multiMap/{}/good/img_good_feature_{}.pickle'.format(ROOT, args.data, args.index)):
        print("load from {}/Results/testing_multiMap/{}/good/img_good_feature_{}.pickle".format(ROOT, args.data, args.index))
        img_all_feature = pickle.load(open('{}/Results/testing_multiMap/{}/good/img_good_feature_{}.pickle'.format(ROOT, args.data, args.index), 'rb'))
    else:
        img_good_feature = eval_OriginFeature(pretrain_model, scratch_model, test_good_loader, kmeans, pca, args.data, global_index, good=True)
    
    """ save feature """ 
    
    save_all_path = "{}/Results/testing_multiMap/{}/all/".format(ROOT, args.data)
    if not os.path.isdir(save_all_path):
        os.makedirs(save_all_path)
    save_all_name = "{}_img_all_feature_{}_Origin.pickle".format(args.kmeans, args.index)
    pickle.dump(img_all_feature, open(save_all_path+save_all_name, "wb"))
    
    save_good_path = "{}/Results/testing_multiMap/{}/good/".format(ROOT, args.data)
    if not os.path.isdir(save_good_path):
        os.makedirs(save_good_path)
    save_good_name = "{}_img_good_feature_{}_Origin.pickle".format(args.kmeans, args.index)
    pickle.dump(img_good_feature, open(save_good_path+save_good_name, "wb"))


    label_pred = []
    label_true = []

    """ for defect type """ 
    for ((idx, img), (idx2, img2)) in zip(test_all_loader, mask_loader):
        img = img.cuda()
        idx = idx[0].item()


        errorMap = img_all_feature[idx].reshape((1024, 1024))
        
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
        
        """ for computing aucroc score """
        defect_gt = np.zeros((1024, 1024, 3))
        true_mask = defect_gt[:, :, 0].astype('int32')
        label_pred.append(errorMap)
        label_true.append(true_mask)    
        print(f'EP={global_index} good_img_idx={idx}')

    label_pred = norm(np.array(label_pred))
    print(label_pred.shape)
    print(np.array(label_true).shape)
    auc = roc_auc_score(np.array(label_true).flatten(), label_pred.flatten())
    print("MultiMap AUC score for testing data {}: {}".format(args.data, auc))
