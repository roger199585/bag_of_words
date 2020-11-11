# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
import torchvision.models as models

import os
import sys
import cv2
import pickle
import random
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt

# customize
import resnet
import dataloaders
import pretrain_vgg
import pretrain_resnet
from config import ROOT
from visualize import errorMap
from utils.tools import one_hot, one_hot_forMap, draw_errorMap


# evaluations
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_auc_score, average_precision_score


# from config import gamma

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
parser.add_argument('--train_batch', type=int, default=16)
parser.add_argument('--test_batch_size', type=int, default=64)
parser.add_argument('--pretrain', type=str, default='False')
args = parser.parse_args()

kmeans_path = "{}/preprocessData/kmeans/{}/{}_{}_{}_{}.pickle".format(
    ROOT,
    args.data,
    str(args.model),
    str(args.kmeans),
    str(args.batch),
    str(args.dim)
)

pca_path = "{}/preprocessData/PCA/{}/{}_{}_{}_{}.pickle".format(
    ROOT,
    args.data, 
    str(args.model), 
    str(args.kmeans), 
    str(args.batch), 
    str(args.dim)
)

left_i_path = "{}/preprocessData/coordinate/vgg19/{}/left_i.pickle".format(ROOT, args.data)
left_j_path = "{}/preprocessData/coordinate/vgg19/{}/left_j.pickle".format(ROOT, args.data)

kmeans = pickle.load(open(kmeans_path, "rb"))
pca = pickle.load(open(pca_path, "rb"))
# gmm = pickle.load(open(gmm_path, "rb"))


""" image transform """
pil_to_tensor = transforms.ToTensor()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#-------------------------------------

out = args.kmeans
scratch_model = nn.Sequential(
    resnet.resnet18(pretrained=False, num_classes=out),
)

""" training """
train_path = "{}/dataset/{}/train_resize/good".format(ROOT, args.data)
label_name = "{}/preprocessData/label/fullPatch/{}/{}/kmeans_{}_{}.pth".format(
    ROOT,
    str(args.model),
    args.data,
    str(out),
    str(args.batch)
)
mask_path = "{}/dataset/big_mask/".format(ROOT)

if args.model == 'vgg19':
    pretrain_model = nn.DataParallel(pretrain_vgg.model).to(device)

if args.model == 'resnet34':
    pretrain_model = nn.DataParallel(pretrain_resnet.model).to(device)

print('training data: ', train_path)
print('training label: ', label_name)

""" testing """
if (args.type == 'good'):
    test_path = "{}/dataset/{}/test_resize/good".format(ROOT, args.data)
    test_label_name = "{}/preprocessData/label/{}/{}/test/good_{}_{}.pth".format(
        ROOT,
        str(args.model),
        args.data,
        str(out),
        str(args.batch)
    )
    
    all_test_label_name = "{}/preprocessData/label/{}/{}/test/all_{}_{}.pth".format(
        ROOT,
        str(args.model),
        args.data,
        str(out),
        str(args.batch)
    )

else:
    test_path = "{}/dataset/{}/test_resize/{}".format(ROOT, args.data, args.type)
    test_label_name = "{}/preprocessData/label/{}/{}/test/{}_{}_{}.pth".format(
        ROOT,
        str(args.model),
        args.data,
        args.type,
        str(out),
        str(args.batch)
    )
    defect_gt_path = "{}/dataset/{}/ground_truth_resize/{}/".format(ROOT, args.data, args.type)


test_label = torch.tensor(torch.load(test_label_name))
print(test_label.shape)
all_test_label = torch.tensor(torch.load(all_test_label_name))
print(all_test_label.shape)

""" eval """
eval_path = "{}/dataset/{}/test_resize/all".format(ROOT, args.data)
eval_mask_path = "{}/dataset/{}/ground_truth_resize/all/".format(ROOT, args.data)
    

""" the ground truth defective mask path """ 
# defect_gt_path = "dataset/{}/ground_truth/{}_resize/".format(args.data, args.type)

print('testing data: ', test_path)
print('testing label: ', test_label_name)

eval_fea_count = 0

def myNorm(features):
    return features / features.max()

def eval_feature(epoch, model, test_loader, __labels, isGood):
    global eval_fea_count
    global pretrain_model
    global kmeans

    model.eval()
    pretrain_model.eval()

    with torch.no_grad():
        img_feature = []
        total_gt = []
        total_idx = []

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

            for i in range(16):
                for j in range(16):
                    crop_img = img[:, :, i*64:i*64+64, j*64:j*64+64].to(device)
                    crop_output = pretrain_model(crop_img)
                    """ flatten the dimension of H and W """
                    out_ = crop_output.flatten(1,2).flatten(1,2)
                    out = pca.transform(out_.detach().cpu().numpy())
                    out = pil_to_tensor(out).squeeze().to(device)
                    crop_list.append(out)

                    mask = torch.ones(1, 1, 1024, 1024)
                    mask[:, :, i*64:i*64+64, j*64:j*64+64] = 0
                    mask = mask.to(device)
                    x = img * mask
                    x = torch.cat((x, mask), 1)
                    label = __labels[idx][i*16+j].to(device)
                   
                    xs.append(x)
                    ys.append(label)

                if (len(xs) == args.test_batch_size):
                    x = torch.cat(xs, 0)
                    y = torch.stack(ys).squeeze().to(device)
                    xs.clear()
                    ys.clear()

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

                        if isGood:
                            writer.add_scalar('test_feature_loss', diff.item(), eval_fea_count)
                            writer.add_scalar('test_label_loss', diff_label.item(), eval_fea_count)
                            writer.add_scalar('test_label_acc', acc.item(), eval_fea_count)
                            eval_fea_count += 1

                    crop_list.clear()

            total_gt.append(label_gt)
            total_idx.append(label_idx)
            img_feature.append(value_feature)
    print(np.array(img_feature).shape)
    print(len(test_loader))
    img_feature = np.array(img_feature).reshape((len(test_loader), -1))
    total_gt = np.array(total_gt).reshape((len(test_loader), -1))
    total_idx = np.array(total_idx).reshape((len(test_loader), -1))

    return img_feature, total_gt, total_idx

def noise_training(train_loader, pretrain_model, scratch_model, criterion, optimizer, writer, kmeans, pca, batch_size, epoch, epoch_num):
    
    global iter_count
    scratch_model.train()
    for (idx, img) in train_loader:
        idx = idx[0].item()

        """ batch = 32 """ 
        indexs_i = list(range(256))
        indexs_j = list(range(256))

        random.shuffle(indexs_i)
        random.shuffle(indexs_j)

        # for train data
        xs = []
        ys = []

        for (i, j) in zip(indexs_i, indexs_j):
            i = int(i / 16)
            j = j % 16

            """ add noise """
            noise_x = random.randint(-32,32)
            noise_y = random.randint(-32,32)

            img = img.to(device)

            mask = torch.ones(1, 1, 1024, 1024)
            if (i*64+64+noise_x > 1024 or i*64+noise_x < 0):
                noise_x = 0
            if (j*64+64+noise_y > 1024 or j*64+noise_y < 0):
                noise_y = 0

            mask[:, :, i*64+noise_x:i*64+64+noise_x, j*64+noise_y:j*64+64+noise_y] = 0
            img_ = img[:, :, i*64+noise_x:i*64+64+noise_x, j*64+noise_y:j*64+64+noise_y]
            mask = mask.to(device)
            img_ = img_.to(device)

            x = img * mask
            x = torch.cat((x, mask), 1)
            
            out = pretrain_model(img_)
            out_ = out.flatten(1,2).flatten(1,2)
            out = pca.transform(out_.detach().cpu().numpy())
            img_idx = torch.tensor(kmeans.predict(out))

            xs.append(x)
            ys.append(img_idx)

            if len(xs) == batch_size:
                scratch_model = scratch_model.train()
                x = torch.cat(xs, 0)
                y = torch.stack(ys).long().squeeze().to(device)
                
                xs.clear()
                ys.clear()
                
                output = scratch_model(x)

                output = nn.Softmax(dim=1)(output)

                """ MSE loss """
                label_onehot = one_hot(y, args.kmeans, batch_size)
                loss = criterion(output, label_onehot)
                acc = (output.argmax(-1).detach() == y).float().mean()  

                optimizer.zero_grad()
                loss.backward()                
                optimizer.step()

                writer.add_scalar('loss', loss.item(), iter_count)
                writer.add_scalar('acc', acc.item(), iter_count)
                print(f'Training EP={epoch+epoch_num} it={iter_count} loss={loss.item()} acc={acc.item()}')

                if (iter_count % 200 == 0):
                    img_feature, total_gt, total_idx = eval_feature(epoch+epoch_num, scratch_model, test_loader, test_loader)
                iter_count += 1
            

    return scratch_model, img_feature, total_gt, total_idx


if __name__ == "__main__":

    """ Summary Writer """
    writer = SummaryWriter(log_dir="../tensorboard/{}_{}_{}_{}".format(args.data, args.type, args.kmeans, datetime.now()))

    """ weight sampling with noise patch in training data """
    train_dataset = dataloaders.NoisePatchDataloader(train_path, label_name, left_i_path, left_j_path)
    samples_weights = torch.from_numpy(train_dataset.samples_weights)
    sampler = WeightedRandomSampler(samples_weights.type('torch.DoubleTensor'), len(samples_weights))
    train_loader = DataLoader(train_dataset, batch_size=args.train_batch, num_workers=1, sampler=sampler)


    test_dataset = dataloaders.MvtecLoader(test_path)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    # mask_dataset = dataloaders.MaskLoader(defect_gt_path)
    # mask_loader = DataLoader(mask_dataset, batch_size=1, shuffle=False)
    
    eval_dataset = dataloaders.MvtecLoader(eval_path)
    eval_loader = DataLoader(eval_dataset, batch_size=1, shuffle=False)
    eval_mask_dataset = dataloaders.MaskLoader(eval_mask_path)
    eval_mask_loader = DataLoader(eval_mask_dataset, batch_size=1, shuffle=False)

    scratch_model = nn.DataParallel(scratch_model).to(device)
    if (args.pretrain == 'True'):
        scratch_model.load_state_dict(torch.load('models/{}/{}/exp6_128_5.ckpt'.format(
            args.model, 
            args.data
            )   
        ))
        print("--- Load pretrain model ---")
        epoch_num = 5
    else:
        epoch_num = 0

    """ training config """ 
    # criterion = nn.MSELoss()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(scratch_model.parameters(), lr = args.lr)
    
    iter_count = 1
    
    for epoch in range(args.epoch): 
        """ noise version 2 """
        print("------- For defect type -------")
        value_feature, total_gt, total_idx = eval_feature(epoch, scratch_model, eval_loader, all_test_label, isGood=False)
        print("------- For good type -------")
        value_good_feature, total_good_gt, total_good_idx = eval_feature(epoch, scratch_model, test_loader, test_label, isGood=True)

        label_pred = []
        label_gt = []

        """ for defect type """ 
        for ((idx, img), (idx2, img2)) in zip(eval_loader, eval_mask_loader):
            img = img.cuda()
            idx = idx[0].item()

            error_map = np.zeros((1024, 1024))
            for index, scalar in enumerate(value_feature[idx]):
                mask = cv2.imread('dataset/big_mask/mask{}.png'.format(index), cv2.IMREAD_GRAYSCALE)
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

            error_map = np.zeros((1024, 1024))
            for index, scalar in enumerate(value_good_feature[idx]):
                mask = cv2.imread('dataset/big_mask/mask{}.png'.format(index), cv2.IMREAD_GRAYSCALE)
                mask = np.invert(mask)
                mask[mask==255]=1
                error_map += mask * scalar

            defect_gt = np.zeros((1024, 1024, 3))
            true_mask = defect_gt[:, :, 0].astype('int32')
            label_pred.append(error_map)
            label_gt.append(true_mask)    
            print(f'EP={epoch} good_img_idx={idx}')

        label_pred = myNorm(np.array(label_pred))
        auc = roc_auc_score(np.array(label_gt).flatten(), label_pred.flatten())
        writer.add_scalars('eval_score', {
            'roc_auc_score': auc
        }, epoch)
        print("AUC score for testing data {}: {}".format(auc, args.data))
        
        for (idx, img, left_i, left_j, label, mask) in train_loader:
            scratch_model.train()
            idx = idx[0].item()
            
            img = img.to(device)
            mask = mask.to(device)

            x = img * mask
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

            if (iter_count % 200 == 0):
                value_good_feature, total_good_gt, total_good_idx = eval_feature(epoch, scratch_model, test_loader, test_label, isGood=True)
            
            iter_count += 1
        
        if not os.path.isdir('{}/models/{}/{}'.format(ROOT, args.model, args.data)):
            os.makedirs('{}/models/{}/{}'.format(ROOT, args.model, args.data))
        
        path = "{}/models/{}/{}/exp1_{}_{}.ckpt".format(
            ROOT,
            args.model, 
            args.data, 
            str(out), 
            str(epoch+1+epoch_num)
        )
        torch.save(scratch_model.state_dict(), path)

