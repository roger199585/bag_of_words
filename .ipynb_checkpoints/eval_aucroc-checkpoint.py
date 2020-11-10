import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import os
import cv2
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

import resnet
import pretrain_vgg
import dataloaders



scratch_model = nn.Sequential(
    resnet.resnet18(pretrained=False, num_classes=128)
)
scratch_model = nn.DataParallel(scratch_model).cuda()


### DataSet
test_path = "/home/dinosaur/bag_of_words/dataset/leather/test/good"
test_dataset = dataloaders.MvtecLoader(test_path)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# mask_path = "dataset/bottle/ground_truth/broken_small_resize/"
mask_path = "dataset/leather/ground_truth/broken_small_resize2/"
mask_dataset = dataloaders.MaskLoader(mask_path)
mask_loader = DataLoader(mask_dataset, batch_size=1, shuffle=False)

## Models
pretrain_model = nn.DataParallel(pretrain_vgg.model).cuda()

## Clusters
kmeans_path = "kmeans/bottle/vgg19_128_100_16.pickle"
kmeans = pickle.load(open(kmeans_path, "rb"))

pca_path = "PCA/bottle/vgg19_128_100_16.pickle"
pca = pickle.load(open(pca_path, "rb"))

## Label
test_label_name = "label/vgg19/bottle/test/broken_small_128_100.pth"
test_label = torch.tensor(torch.load(test_label_name))

## Others
pil_to_tensor = transforms.ToTensor()

eval_fea_count = 0
test_type = 'broken_small'

def eval_feature(epoch, model, test_loader, mask_loader):
    global eval_fea_count
    global pretrain_model
    global kmeans

    model.eval()
    pretrain_model.eval()
    split_size = 8

    with torch.no_grad():
        img_feature = []
        total_gt = []
        total_idx = []

        for ((idx, img), (idx2, img2)) in zip(test_loader, mask_loader):
            img = img.cuda()
            idx = idx[0].item()

            print(f'eval phase: img idx={idx}')

            value_feature = []
            value_label = []
            label_idx = []
            label_gt = []

            for i in range(16):
                xs = []
                ys = []

                crop_list = []
                loss = 0.0
                loss_alpha = 0.0

                for j in range(16):
                    crop_img = img[:, :, i*64:i*64+64, j*64:j*64+64].cuda()
                    crop_output = pretrain_model(crop_img)
                    """ flatten the dimension of H and W """
                    out_ = crop_output.flatten(1,2).flatten(1,2)
                    out = pca.transform(out_.detach().cpu().numpy())
                    out = pil_to_tensor(out).squeeze().cuda()
                    crop_list.append(out)

                    mask = torch.ones(1, 1, 1024, 1024)
                    mask[:, :, i*64:i*64+64, j*64:j*64+64] = 0
                    mask = mask.cuda()
                    x = img * mask
                    x = torch.cat((x, mask), 1)
                    label = test_label[idx][i*16+j].cuda()
                   
                    xs.append(x)
                    ys.append(label)
            
                x = torch.cat(xs, 0)
                y = torch.stack(ys).squeeze().cuda()

                output = model(x)
                y_ = output.argmax(-1).detach().cpu().numpy()

                for k in range(16):
                    label_idx.append(y_[k])
                    label_gt.append(y[k].item())
                    output_center = kmeans.cluster_centers_[y_[k]]
                    output_center = np.reshape(output_center, (1, -1))
                    output_center = pil_to_tensor(output_center).cuda()
                    output_center = torch.squeeze(output_center)

                    if y_[k] == y[k].item():
                        isWrongLabel = 0
                    else:
                        isWrongLabel = 1

                    diff = isWrongLabel * nn.MSELoss()(output_center, crop_list[k])
                    value_feature.append(diff.item())
                    
                    eval_fea_count += 1


            total_gt.append(label_gt)
            total_idx.append(label_idx)
            img_feature.append(value_feature)


    img_feature = np.array(img_feature).reshape((len(test_loader), -1))
    total_gt = np.array(total_gt).reshape((len(test_loader), -1))
    total_idx = np.array(total_idx).reshape((len(test_loader), -1))

    return img_feature, total_gt, total_idx

for global_index in range(1, 6):
    scratch_model.load_state_dict(torch.load('/home/dinosaur/bag_of_words/models/vgg19/bottle/exp6_128_{}.ckpt'.format(global_index)))

    value_feature, total_gt, total_idx = eval_feature(global_index, scratch_model, test_loader, mask_loader)

    for ((idx, img), (idx2, img2)) in zip(test_loader, mask_loader):
        img = img.cuda()
        idx = idx[0].item()


        error_map = np.zeros((1024, 1024))
        for index, scalar in enumerate(value_feature[idx]):
            mask = cv2.imread('dataset/big_mask/mask{}.png'.format(index), cv2.IMREAD_GRAYSCALE)
            mask = np.invert(mask)
            mask[mask==255]=1
            
            error_map += mask * scalar

        
        

        if (test_type == 'good'):
            img_ = np.squeeze(img.detach().cpu().numpy()).transpose((1,2,0))
            ironman_grid = plt.GridSpec(1, 2)
            fig = plt.figure(figsize=(12,6), dpi=100)
            ax1 = fig.add_subplot(ironman_grid[0,0])
            im1 = ax1.imshow(error_map, cmap="Blues")
            ax2 = fig.add_subplot(ironman_grid[0,1])
            im2 = ax2.imshow(img_)
        else:
            img_ = np.squeeze(img.detach().cpu().numpy()).transpose((1,2,0))
            defect_gt = np.squeeze(img2.cpu().numpy()).transpose((1,2,0))
            ironman_grid = plt.GridSpec(1, 3)
            fig = plt.figure(figsize=(18,6), dpi=100)
            ax1 = fig.add_subplot(ironman_grid[0,0])
            im1 = ax1.imshow(error_map, cmap="Blues")
            ax2 = fig.add_subplot(ironman_grid[0,1])
            ax3 = fig.add_subplot(ironman_grid[0,2])
            im2 = ax2.imshow(img_)
            im3 = ax3.imshow(defect_gt)

            ## 可以在這邊算
            true_mask = defect_gt[:, :, 0].astype('int32') 
            auc = roc_auc_score(true_mask.flatten(), error_map.flatten())

            ax1.set_title(auc)



        """ add label text to each patch """ 
        for i in range(16):
            for j in range(16):
                ax1.text((j+0.2)*64, (i+0.6)*64, total_idx[idx][i*16+j], fontsize=10)
                ax2.text((j+0.2)*64, (i+0.6)*64, total_gt[idx][i*16+j], fontsize=10)


        errorMapPath = "./testing/"
        if not os.path.isdir(errorMapPath):
            os.makedirs(errorMapPath)
            print("----- create folder for type:{} -----".format(test_type))
        
        errorMapName = "{}_{}.png".format(
            str(idx),
            str(global_index) 
        )

        plt.savefig(errorMapPath+errorMapName, dpi=100)
        plt.close(fig)


        print(f'EP={global_index} img_idx={idx}')
