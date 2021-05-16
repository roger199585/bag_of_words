# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
import torchvision.models as models

# STL Library
import os
import sys
import cv2
import time
import pickle
import random
import argparse
import itertools
import numpy as np
from PIL import Image
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt

# customize
import networks.resnet as resnet
import dataloaders
from preprocess_SSL.SSL import model as ssl_model
from config import ROOT, RESULT_PATH

# evaluations
from sklearn.metrics import roc_auc_score


# from config import gamma

""" set parameters """
parser = argparse.ArgumentParser()
parser.add_argument('--kmeans', type=int, default=128)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--epoch', type=int, default=10)
parser.add_argument('--data', type=str, default='bottle')
parser.add_argument('--type', type=str, default='good')
parser.add_argument('--batch', type=int, default=100)
parser.add_argument('--dim', type=int, default=128)
parser.add_argument('--model', type=str, default='ssl')
parser.add_argument('--train_batch', type=int, default=16)
parser.add_argument('--test_batch_size', type=int, default=64)
parser.add_argument('--with_mask', type=str, default='True')
parser.add_argument('--patch_size', type=int, default=16)
parser.add_argument('--image_size', type=int, default=256)
parser.add_argument('--dim_reduction', type=str, default='PCA')
args = parser.parse_args()

kmeans_path = f"{ ROOT }/preprocessData/kmeans/{ args.dim_reduction }/{ args.data }/{ args.model }_{ args.kmeans }_{ args.batch }_{ args.dim }.pickle"
pca_path    = f"{ ROOT }/preprocessData/{ args.dim_reduction }/{ args.data }/{ args.model }_{ str(args.kmeans) }_{ str(args.batch) }_{ str(args.dim) }.pickle"

left_i_path = f"{ ROOT }/preprocessData/coordinate/{ args.model }/{ args.dim_reduction }/{ args.data }/left_i.pickle"
left_j_path = f"{ ROOT }/preprocessData/coordinate/{ args.model }/{ args.dim_reduction }/{ args.data }/left_j.pickle"

pca = pickle.load(open(pca_path, "rb"))
kmeans = pickle.load(open(kmeans_path, "rb"))


""" image transform """
pil_to_tensor = transforms.ToTensor()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#-------------------------------------

out = args.kmeans
scratch_model = nn.Sequential(
    resnet.resnet50(pretrained=False, num_classes=args.kmeans),
)

""" training """
train_path = f"{ ROOT }/dataset/{ args.data }/train_resize/good"
label_name = f"{ ROOT }/preprocessData/label/fullPatch/{ args.model }/{ args.data }/kmeans_{ args.kmeans }_{ args.batch }.pth"

model_path = f"{ ROOT }/preprocess_SSL/SSL/KNN/exp3/{ args.data }/2048_2000.pth"

pretrain_model = ssl_model.Model()
pretrain_model = nn.DataParallel(pretrain_model).cuda()
pretrain_model.load_state_dict(torch.load(model_path))
pretrain_model.eval()

print('training data: ', train_path)
print('training label: ', label_name)

""" testing """
if (args.type == 'good'):
    test_path           = f"{ ROOT }/dataset/{ args.data }/test_resize/good"
    test_label_name     = f"{ ROOT }/preprocessData/label/{ args.model }/{ args.dim_reduction }/{ args.data }/test/good_{ str(args.kmeans) }_{ str(args.batch) }.pth"
    all_test_label_name = f"{ ROOT }/preprocessData/label/{ args.model }/{ args.dim_reduction }/{ args.data }/test/all_{ str(args.kmeans) }_{ str(args.batch) }.pth"
else:
    test_path       = f"{ ROOT }/dataset/{ args.data }/test_resize/{ args.type }"
    test_label_name = f"{ ROOT }/preprocessData/label/{ args.model }/{ args.dim_reduction }/{ args.data }/test/{ args.type }_{ str(args.kmeans) }_{ str(args.batch) }.pth"
    defect_gt_path  = f"{ ROOT }/dataset/{ args.data }/ground_truth_resize/{ args.type }/"


test_label = torch.tensor(torch.load(test_label_name))
all_test_label = torch.tensor(torch.load(all_test_label_name))
print(test_label.shape)
print(all_test_label.shape)

""" eval """
eval_path = "{}/dataset/{}/test_resize/all".format(ROOT, args.data)
eval_mask_path = "{}/dataset/{}/ground_truth_resize/all/".format(ROOT, args.data)

print('testing data: ', test_path)
print('testing label: ', test_label_name)

eval_fea_count = 0

def myNorm(features):
    if features.max() > 0:
        return features / features.max()
    else:
        return features

def eval_feature(epoch, model, test_loader, __labels, isGood):
    global eval_fea_count
    global pretrain_model
    global kmeans

    model.eval()
    pretrain_model.eval()

    with torch.no_grad():
        img_feature = []

        for (idx, img) in test_loader:
            img = img.to(device)
            idx = idx[0].item()

            print(f'eval phase: img idx={idx}')

            value_feature = []
            value_label = []
            label_idx = []
            label_gt = []

            xs = []
            ys = []
            crop_list = []
            origin_feature_list = []

            patches = []

            chunk_num = int(args.image_size / args.patch_size)
            for i in range(chunk_num):
                for j in range(chunk_num):
                    crop_img = img[:, :, i*args.patch_size:i*args.patch_size+args.patch_size, j*args.patch_size:j*args.patch_size+args.patch_size].to(device)
                    crop_output, _ = pretrain_model(crop_img)

                    """ flatten the dimension of H and W """
                    out_ = crop_output.flatten(1,2).flatten(1,2)
                    patches.append(out_.detach().cpu().numpy())
                    origin_feature_list.append(out_)

                    mask = torch.ones(1, 1, 256, 256)
                    mask[:, :, i*args.patch_size:i*args.patch_size+args.patch_size, j*args.patch_size:j*args.patch_size+args.patch_size] = 0
                    mask = mask.to(device)
                    x = img * mask if args.with_mask == 'True' else img
                    x = torch.cat((x, mask), 1)
                    label = __labels[idx][i*chunk_num+j].to(device)
                    ys.append(label)
                    xs.append(x)


                if (len(xs) == args.test_batch_size):
                    np_patches = np.array(patches)
                    np_patches = np_patches.reshape(-1, np_patches.shape[-1])

                    new_outs = pca.transform(np_patches)
                    for i in range(new_outs.shape[0]):
                        f = new_outs[i].reshape(1, -1)
                        f = pil_to_tensor(f).to(device)
                        f = torch.squeeze(f)

                        crop_list.append(f)

                    x = torch.cat(xs, 0)
                    y = torch.stack(ys).squeeze().to(device)
                    xs.clear()
                    ys.clear()
                    patches.clear()

                    output = model(x)
                    y_ = output.argmax(-1).detach().cpu().numpy()
                    acc = (output.argmax(-1).detach() == y).float().mean()  

                    for k in range(args.test_batch_size):
                        label_idx.append(y_[k])
                        label_gt.append(y[k].item())
                        output_center = kmeans.cluster_centers_[y_[k]]
                        output_center = np.reshape(output_center, (1, -1))
                        output_center = pil_to_tensor(output_center).to(device)
                        output_center = torch.squeeze(output_center)

                        isWrongLabel = int(y_[k] != y[k].item())

                        un_out = torch.unsqueeze(output[k], dim=0)
                        un_y = torch.unsqueeze(y[k], dim=0).long()
                        diff_label = nn.CrossEntropyLoss()(un_out, un_y)

                        diff = isWrongLabel * nn.MSELoss()(output_center, crop_list[k])
                        value_feature.append(diff.item())

                        output_feature = np.expand_dims(cluster_features[y_[k]], axis=0)
                        output_feature = torch.from_numpy(output_feature).cuda()

                        isWrongLabel = int(y_[k] != y[k].item())
                        
                        origin_feature_diff = isWrongLabel * nn.MSELoss()(output_feature, origin_feature_list[k])
                    

                        if isGood:
                            writer.add_scalar('test_feature_loss', diff.item(), eval_fea_count)
                            writer.add_scalar('test_origin_feature_loss', origin_feature_diff.item(), eval_fea_count)
                            writer.add_scalar('test_label_loss', diff_label.item(), eval_fea_count)
                            writer.add_scalar('test_label_acc', acc.item(), eval_fea_count)
                            eval_fea_count += 1

                    crop_list.clear()
                    origin_feature_list.clear()

            img_feature.append(value_feature)
    print(np.array(img_feature).shape)
    print(len(test_loader))
    img_feature = np.array(img_feature).reshape((len(test_loader), -1))

    return img_feature

if __name__ == "__main__":

    """ Summary Writer """
    writer = SummaryWriter(log_dir="{}/Finial_SSL_mask_{}_patch_{}_{}_{}_{}_{}".format(RESULT_PATH, args.with_mask, args.patch_size, args.data, args.type, args.kmeans, datetime.now()))

    """ weight sampling with noise patch in training data """
    train_dataset = dataloaders.NoisePatchDataloader(train_path, label_name, left_i_path, left_j_path)
    samples_weights = torch.from_numpy(train_dataset.samples_weights)
    sampler = WeightedRandomSampler(samples_weights.type('torch.DoubleTensor'), len(samples_weights))
    train_loader = DataLoader(train_dataset, batch_size=args.train_batch, num_workers=0, sampler=sampler)

    # testing set (normal data)
    test_dataset = dataloaders.MvtecLoader(test_path)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    # testing set (defect data)
    eval_dataset = dataloaders.MvtecLoader(eval_path)
    eval_loader = DataLoader(eval_dataset, batch_size=1, shuffle=False)
    eval_mask_dataset = dataloaders.MaskLoader(eval_mask_path)
    eval_mask_loader = DataLoader(eval_mask_dataset, batch_size=1, shuffle=False)

    ## Cluster Center Features
    center_features_path = "{}/preprocessData/cluster_center/{}/{}.pickle".format(ROOT, args.kmeans, args.data)
    cluster_features = pickle.load(open(center_features_path, "rb"))

    scratch_model = nn.DataParallel(scratch_model).to(device)
    epoch_num = 0

    """ training config """ 
    # criterion = nn.MSELoss()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(scratch_model.parameters(), lr = args.lr)
    
    iter_count = 1
    
    for epoch in range(args.epoch): 
        """ noise version 2 """
        print("------- For defect type -------")
        value_feature = eval_feature(epoch, scratch_model, eval_loader, all_test_label, isGood=False)
        print("------- For good type -------")
        value_good_feature = eval_feature(epoch, scratch_model, test_loader, test_label, isGood=True)

        label_pred = []
        label_gt = []

        """ for defect type """ 
        for ((idx, img), (idx2, img2)) in zip(eval_loader, eval_mask_loader):
            img = img.cuda()
            idx = idx[0].item()

            error_map = np.zeros((256, 256))
            for index, scalar in enumerate(value_feature[idx]):
                mask = cv2.imread('{}/dataset/ssl_mask/mask{}.png'.format(ROOT, index), cv2.IMREAD_GRAYSCALE)
                mask = np.invert(mask)
                mask[mask==255]=1
                
                error_map += mask * scalar

            ## 可以在這邊算
            defect_gt = np.squeeze(img2.cpu().numpy()).transpose(1,2,0)
            true_mask = defect_gt[:, :, 0].astype('int32')
            label_pred.append(error_map)
            label_gt.append(true_mask)    
            print(f'EP={epoch} defect_img_idx={idx}')


        """ for good type """
        for (idx, img) in test_loader:
            img = img.cuda()
            idx = idx[0].item()

            error_map = np.zeros((256, 256))
            for index, scalar in enumerate(value_good_feature[idx]):
                mask = cv2.imread('{}/dataset/ssl_mask/mask{}.png'.format(ROOT, index), cv2.IMREAD_GRAYSCALE)
                mask = np.invert(mask)
                mask[mask==255]=1
                error_map += mask * scalar

            defect_gt = np.zeros((256, 256, 3))
            true_mask = defect_gt[:, :, 0].astype('int32')
            label_pred.append(error_map)
            label_gt.append(true_mask)    
            print(f'EP={epoch} good_img_idx={idx}')

        label_pred = myNorm(np.array(label_pred))
        auc = roc_auc_score(np.array(label_gt).flatten(), label_pred.flatten())

        writer.add_scalar('roc_auc_score', auc, epoch)
        print("AUC score for testing data {}: {}".format(auc, args.data))
        
        for (idx, img, left_i, left_j, label, mask) in train_loader:
            scratch_model.train()
            idx = idx[0].item()
            
            img = img.to(device)
            mask = mask.to(device)

            x = img * mask if args.with_mask == 'True' else img
            x = torch.cat((x, mask), 1)
            label = label.squeeze().to(device, dtype=torch.long)

            output = scratch_model(x)
            
            loss = criterion(output, label)
            acc = (output.argmax(-1).detach() == label).float().mean()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            writer.add_scalar('loss', loss.item(), iter_count)
            writer.add_scalar('acc', acc.item(), iter_count)            
            
            print(f'Training EP={epoch+epoch_num} it={iter_count} loss={loss.item()}')
            
            iter_count += 1

            if iter_count % 1000 == 0:
                value_good_feature = eval_feature(epoch, scratch_model, test_loader, test_label, isGood=True)
        
        if not os.path.isdir('{}/models/{}/{}'.format(ROOT, args.model, args.data)):
            os.makedirs('{}/models/{}/{}'.format(ROOT, args.model, args.data))
        
        path = "{}/models/{}/{}/ssl_exp1_{}_{}.ckpt".format(
            ROOT,
            args.model, 
            args.data, 
            str(out), 
            str(epoch+1+epoch_num)
        )
        torch.save(scratch_model.state_dict(), path)
