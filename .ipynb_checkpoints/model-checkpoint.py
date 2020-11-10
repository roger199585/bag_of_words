import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from sklearn import preprocessing
import torchvision.models as models
from torchvision import transforms
import os
from PIL import Image
from tqdm import tqdm
import pretrain_vgg
import pretrain_resnet
import resnet
import argparse
import pickle
import cv2
from visualize import errorMap
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import random
from utils.tools import one_hot, one_hot_forMap, errorMap_forTrain
import sys
import dataloaders

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
args = parser.parse_args()

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


""" image transform """
pil_to_tensor = transforms.ToTensor()
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#-------------------------------------

out = args.kmeans
scratch_model = nn.Sequential(
    resnet.resnet18(pretrained=False, num_classes=out),
    nn.Sigmoid()
    # nn.ReLU(True),
    # nn.Linear(256, out)
)

""" training """
train_path = "./dataset/{}/train_resize".format(args.data)
label_name = "label/{}/{}/train/{}_{}.pth".format(
    str(args.model),
    args.data,
    str(out),
    str(args.batch)
)
index_label = torch.tensor(torch.load(label_name))

if args.model == 'vgg19':
    pretrain_model = nn.DataParallel(pretrain_vgg.model, device_ids=[0,1]).to(device)
    # pretrain_model = pretrain_vgg.model
    # if torch.cuda.device_count() > 1:
    #     pretrain_model = nn.DataParallel(pretrain_model, device_ids=[0,1])

if args.model == 'resnet34':
    pretrain_model = nn.DataParallel(pretrain_resnet.model, device_ids=[0,1]).to(device)

print('training data: ', train_path)
print('training label: ', label_name)

""" testing """
test_good_path = "./dataset/{}/test/good_resize".format(args.data)

if (args.type == 'good'):
    test_path = test_good_path
    test_label_name = "label/{}/{}/test/good_{}_{}.pth".format(
        str(args.model),
        args.data,
        str(out),
        str(args.batch)
    )

else:
    test_path = "./dataset/{}/test/{}_resize".format(args.data, args.type)
    test_label_name = "label/{}/{}/test/{}_{}_{}.pth".format(
        str(args.model),
        args.data,
        args.type,
        str(out),
        str(args.batch)
    )

test_label = torch.tensor(torch.load(test_label_name))


""" the ground truth defective mask path """ 
defect_gt_path = "dataset/{}/ground_truth/{}_resize/".format(args.data, args.type)

print('testing data: ', test_path)
print('testing label: ', test_label_name)

eval_count = 0

def eval(epoch, model, data_loader):
    global eval_count
    model.eval()

    with torch.no_grad():
        for idx, img in (data_loader):
            idx = idx[0].item()
            acc = 0.0
            for i in range(32):
                xs = []
                ys = []
                for j in range(32):
                    img = img.to(device)
                    mask = torch.ones(1, 1, 256, 256)
                    mask[:, :, i*8:i*8+8, j*8:j*8+8] = 0
                    mask = mask.to(device)
                    x = img * mask
                    x = torch.cat((x, mask), 1)

                    label = test_label[idx][i*32+j].to(device)

                    xs.append(x)
                    ys.append(label)

                x = torch.cat(xs, 0)
                y = torch.stack(ys).long().squeeze().to(device)

                output = model(x)
                acc += (output.argmax(-1).detach() == y).float().mean()

            acc = 100.0 * acc / 32
            print(f'EP={epoch} img_idx={idx} test_accuracy={acc}')
            writer.add_scalar('test_acc', acc, eval_count)
            eval_count += 1


norm_count = 0
eval_fea_count = 0

def eval_feature(epoch, model, test_loader, mask_loader):
    global eval_fea_count
    global norm_count
    global pretrain_model
    global kmeans
    global mask_dir

    model.eval()
    pretrain_model.eval()
    split_size = 8

    with torch.no_grad():
        for ((idx, img), (idx2, img2)) in zip(test_loader, mask_loader):
            img = img.to(device)

            loss = 0.0
            idx = idx[0].item()

            outs = []
            value = []
            label_idx = []
            label_gt = []

            for i in range(32):
                xs = []
                ys = []

                crop_list = []

                for j in range(32):
                    crop_img = img[:, :, i*8:i*8+8, j*8:j*8+8].to(device)
                    crop_output = pretrain_model(crop_img)

                    crop_output = crop_output.detach().cpu().numpy()
                    crop_output = np.squeeze(crop_output).reshape((1, -1))
                    crop_output = pca.transform(crop_output)
                    crop_output = torch.from_numpy(crop_output).squeeze().to(device)
                    crop_list.append(crop_output)

                    mask = torch.ones(1, 1, 256, 256)
                    mask[:, :, i*8:i*8+8, j*8:j*8+8] = 0
                    mask = mask.to(device)
                    x = img * mask
                    x = torch.cat((x, mask), 1)
                    label = test_label[idx][i*32+j].to(device)

                    xs.append(x)
                    ys.append(label)
            
                x = torch.cat(xs, 0)
                y = torch.stack(ys).long().squeeze().to(device)

                output = model(x)

                y_ = output.argmax(-1).detach()

                for k in range(32):
                    label_idx.append(y_[k].item())
                    label_gt.append(y[k].item())

                    output_center = kmeans.cluster_centers_[y_[k]]
                    output_center = np.reshape(output_center, (1, -1))
                    output_center = pil_to_tensor(output_center).to(device)
                    output_center = torch.squeeze(output_center)

                    diff = nn.MSELoss()(output_center, crop_list[k])
                    value.append(diff.item())
                    loss += diff


            loss /= 1024
            writer.add_scalar('test_loss', loss, eval_fea_count)
            eval_fea_count += 1

            """ draw errorMap for feature loss"""
            value = preprocessing.minmax_scale(value, feature_range=(0,1), axis=0)
            
            error_map = np.zeros((256, 256))
            for index, scalar in enumerate(value):
                mask = cv2.imread('dataset/big_mask/mask{}.png'.format(index), cv2.IMREAD_GRAYSCALE)
                mask = np.invert(mask)
                mask[mask==255]=1
                error_map += mask * scalar
        
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


            """ add label text to each patch """ 
            for i in range(32):
                for j in range(32):
                    ax1.text((j+0.2)*8, (i+0.6)*8, label_idx[i*32+j], fontsize=5)
                    ax2.text((j+0.2)*8, (i+0.6)*8, label_gt[i*32+j], fontsize=5)


            errorMapPath = "./errorMap/{}/{}/{}/".format(
                args.model,
                args.data,
                args.type
            )
            if not os.path.isdir(errorMapPath):
                os.makedirs(errorMapPath)
                print("-----create folder for type:{} -----".format(args.type))
            
            errorMapName = "{}_{}_{}_{}.png".format(
                str(args.kmeans),
                str(args.dim),
                str(idx),
                str(epoch) 
            )

            plt.savefig(errorMapPath+errorMapName, dpi=100)
            plt.close(fig)


            for i in range(len(value)):
                writer.add_scalar('testLoss_norm', value[i], norm_count)
                norm_count += 1
            
            
            print(f'EP={epoch} img_idx={idx}')
        

train_dataset = dataloaders.MvtecLoader(train_path)
test_dataset = dataloaders.MvtecLoader(test_path)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
mask_dataset = dataloaders.MaskLoader(defect_gt_path)
mask_loader = DataLoader(mask_dataset, batch_size=1, shuffle=False)

scratch_model = nn.DataParallel(scratch_model).to(device)
criterion = nn.CrossEntropyLoss()
# criterion = nn.MSELoss()
optimizer = torch.optim.Adam(scratch_model.parameters(), lr = args.lr)

# writer = SummaryWriter(comment='_corn_' + str(out) + '_' + args.data + '_' + args.type + '_' + args.model + '_MSE')
writer = SummaryWriter(comment='_resNet18_shuffle_batch1024_' + str(out) + '_' + args.data + '_' + args.type + '_' + args.model)

iter_count = 0
maxVal = 0
minVal = 0
for epoch in range(args.epoch):
    # eval_feature_batch1(epoch, scratch_model, test_loader)
    eval_feature(epoch, scratch_model, test_loader, mask_loader)
        
    acc_list = []

    for (idx, img) in train_loader:
        scratch_model.train()
        idx = idx[0].item()

        """ batch = 32 """ 
        indexs_i = list(range(1024))
        indexs_j = list(range(1024))
        random.shuffle(indexs_i)
        random.shuffle(indexs_j)
        xs = []
        ys = []

        for (i, j) in zip(indexs_i, indexs_j):
            i = int(i / 32)
            j = j % 32
        
            img = img.to(device)
            mask = torch.ones(1, 1, 256, 256)
            """ [batch, channels, height, width] """ 
            mask[:, :, i*8:i*8+8, j*8:j*8+8] = 0
            mask = mask.to(device)

            x = img * mask
            x = torch.cat((x, mask), 1)
            label = index_label[idx][i*32+j]

            xs.append(x)
            ys.append(label)

            if len(xs) == 1024:
                x = torch.cat(xs, 0)
                y = torch.stack(ys).long().squeeze().to(device)
                xs.clear()
                ys.clear()
                
                # y_onehot = one_hot(y, out, 32).to(device)
        
                output = scratch_model(x)
        
                # out_soft = nn.Softmax(dim=1)(output)
                acc = (output.argmax(-1).detach() == y).float().mean()

                loss = criterion(output, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                writer.add_scalar('loss', loss.item(), iter_count)
                writer.add_scalar('acc', acc.item(), iter_count)
                
        
                print(f'EP={epoch} img_idx={idx}  it={iter_count+1} loss={loss.item()} acc={acc}')
                iter_count += 1

        if (idx == 1):
            errorMap_forTrain(idx, img, pretrain_model, scratch_model, pca, kmeans, index_label, epoch)

    eval(epoch, scratch_model, test_loader)

    path = "models/{}/{}/kmeans{}_{}.ckpt".format(
        args.model, 
        args.data, 
        str(out), 
        str(epoch+1)
    )
    torch.save(scratch_model.state_dict(), path)

