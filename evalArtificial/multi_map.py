"""
Finial version
use multi error map to calculate auc roc score
"""
import os
import sys
import time
import pickle
import argparse
import itertools
import numpy as np

import torch
import torchvision
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader

import dataloaders
import networks.resnet as resnet
from preprocess_Artificial.artificial_feature import to_int, to_hist

from sklearn.metrics import roc_auc_score
from config import ROOT

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pil_to_tensor = transforms.ToTensor()

def norm(feature):
    return feature / feature.max()

def eval_OriginFeature(model, test_loader, kmeans, test_data, global_index, good=False):
    global label_pred
    global label_true

    model.eval()

    with torch.no_grad():

        img_feature = []
        
        start = time.time()

        chunk_num = int(args.image_size / args.patch_size)
        for (idx, img) in test_loader:
            sss = time.time()
            each_pixel_err_sum = np.zeros([1024, 1024])
            each_pixel_err_count = np.zeros([1024, 1024])

            # pixel_feature = []  
            img = img.to(device)
            idx = idx[0].item()
            
            print(f'eval phase: img idx={idx}')

            """ slide window = 16 """
            map_num = int((args.image_size - args.patch_size) / chunk_num + 1)   ## = 61
            indices = list(itertools.product(range(map_num), range(map_num)))
            
            """ batch """
            batch_size = 32

            label_idx = []
            label_gt = []

            for batch_start_idx in range(0, len(indices), batch_size):
                b_start = time.time()
                xs = []
                ys = []
                crop_list = []

                batch_idxs = indices[batch_start_idx:batch_start_idx+batch_size]

                for i, j in batch_idxs:
                    crop_img = img[:, :, i*chunk_num:i*chunk_num+args.patch_size, j*chunk_num:j*chunk_num+args.patch_size].to(device)
                    """ flatten the dimension of H and W """
                    out = to_hist(crop_img)
                    out_ = out.reshape(1, -1).detach().cpu().numpy()
                    # out = latent_code[7].flatten(1,2).flatten(1,2)
                    # out_ = out.detach().cpu().numpy()
                    out_label = kmeans.predict(out_)
                    out_label = torch.from_numpy(out_label).to(device)
                    crop_list.append(out)

                    mask = torch.ones(1, 1, 1024, 1024)
                    mask[:, :, i*chunk_num:i*chunk_num+args.patch_size, j*chunk_num:j*chunk_num+args.patch_size] = 0
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

                    output_feature = np.expand_dims(cluster_features[y_[n]], axis=0)
                    output_feature = torch.from_numpy(output_feature).cuda()

                    isWrongLabel = int(y_[n] != y[n].item())
                    diff = isWrongLabel * nn.MSELoss()(output_feature, crop_list[n])
                    
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
    args = parser.parse_args()

    global_index = args.index
    test_data = args.data
    
    # AE
    scratch_model = nn.Sequential(
        resnet.resnet50(pretrained=False, num_classes=args.kmeans)
    )
    scratch_model = nn.DataParallel(scratch_model).to(device)
    scratch_model.load_state_dict(torch.load(f"{ ROOT }/models/artificial/{ args.data }/exp_{ args.kmeans }_{ args.index }.ckpt"))


    ### DataSet for all defect type
    test_all_path = f"{ ROOT }/dataset/{ args.data }/test_resize/all/"
    test_all_dataset = dataloaders.MvtecLoader(test_all_path)
    test_all_loader = DataLoader(test_all_dataset, batch_size=1, shuffle=False)

    test_good_path = f"{ ROOT }/dataset/{ args.data }/test_resize/good/"
    test_good_dataset = dataloaders.MvtecLoader(test_good_path)
    test_good_loader = DataLoader(test_good_dataset, batch_size=1, shuffle=False)

    mask_path = f"{ ROOT }/dataset/{ args.data }/ground_truth_resize/all/"
    mask_dataset = dataloaders.MaskLoader(mask_path)
    mask_loader = DataLoader(mask_dataset, batch_size=1, shuffle=False)

    ## Models

    ## Clusters
    kmeans_path = f"{ ROOT }/preprocessData/kmeans/artificial/{ args.data }/artificial_{ args.kmeans }.pickle"
    kmeans = pickle.load(open(kmeans_path, "rb"))

    ## Cluster Center Features
    center_features_path = f"{ ROOT }/preprocessData/cluster_center/artificial/{ args.data }/{ args.kmeans}.pickle"
    cluster_features = pickle.load(open(center_features_path, "rb"))
    
    print("----- defect -----")
    if args.resume and os.path.isfile(f"{ ROOT }/Results/testing_multiMap/artificial/{ args.data }/all/img_all_feature_{ args.index }.pickle"):
        print(f"load from { ROOT }/Results/testing_multiMap/artificial/{ args.data }/all/img_all_feature_{ args.index }.pickle")
        img_all_feature = pickle.load(open(f"{ ROOT }/Results/testing_multiMap/artificial/{ args.data }/all/img_all_feature_{ args.index }.pickle", 'rb'))
    else:
        img_all_feature = eval_OriginFeature(scratch_model, test_all_loader, kmeans, args.data, global_index, good=False)

    print("----- good -----")
    if args.resume and os.path.isfile(f"{ ROOT }/Results/testing_multiMap/artificial/{ args.data }/good/img_good_feature_{ args.index }.pickle"):
        print(f"load from { ROOT }/Results/testing_multiMap/artificial/{ args.data }/good/img_good_feature_{ args.index }.pickle")
        img_all_feature = pickle.load(open(f"{ ROOT }/Results/testing_multiMap/artificial/{ args.data }/good/img_good_feature_{ args.index }.pickle", 'rb'))
    else:
        img_good_feature = eval_OriginFeature(scratch_model, test_good_loader, kmeans, args.data, global_index, good=True)
    
    """ save feature """ 
    
    save_all_path = f"{ ROOT }/Results/testing_multiMap/artificial/{ args.data }/all/".format(ROOT, args.data)
    if not os.path.isdir(save_all_path):
        os.makedirs(save_all_path)
    save_all_name = f"{ args.kmeans }_img_all_feature_{ args.index }_Origin.pickle"
    pickle.dump(img_all_feature, open(save_all_path+save_all_name, "wb"))
    
    save_good_path = f"{ ROOT }/Results/testing_multiMap/artificial/{ args.data }/good/"
    if not os.path.isdir(save_good_path):
        os.makedirs(save_good_path)
    save_good_name = f"{ args.kmeans }_img_good_feature_{ args.index }_Origin.pickle"
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
    auc = roc_auc_score(np.array(label_true).flatten(), label_pred.flatten())
    print("Multi Map AUC score for testing data {}: {}".format(args.data, auc))

