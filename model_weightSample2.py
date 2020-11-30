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
import pretrain_resnet
import resnet
import argparse
import pickle
import cv2
from visualize import errorMap
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import random
from utils.tools import one_hot, one_hot_forMap, draw_errorMap
import sys
import dataloaders
import torch.nn.functional as F


# from config import gamma

""" set parameters """
parser = argparse.ArgumentParser()
parser.add_argument('--kmeans', type=int, default=16)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--epoch', type=int, default=10)
parser.add_argument('--data', type=str, default='bottle')
parser.add_argument('--type', type=str, default='good')
parser.add_argument('--batch', type=int, default=100)
parser.add_argument('--dim', type=int, default=128)
parser.add_argument('--model', type=str, default='vgg19')
parser.add_argument('--train_batch', type=int, default=32)
parser.add_argument('--pretrain', type=str, default='False')
args = parser.parse_args()

kmeans_path = "./preprocessData/kmeans/{}/{}_{}_{}_{}.pickle".format(
    args.data,
    str(args.model),
    str(args.kmeans),
    str(args.batch),
    str(args.dim)
)

# gmm_path = "gmm/{}/{}_{}_{}_{}.pickle".format(
#     args.data,
#     args.model, 
#     str(args.kmeans),
#     str(args.batch),
#     str(args.dim)
# )

pca_path = "./preprocessData/PCA/{}/{}_{}_{}_{}.pickle".format(
    args.data, 
    str(args.model), 
    str(args.kmeans), 
    str(args.batch), 
    str(args.dim)
)

left_i_path = "./preprocessData/coordinate/vgg19/{}/left_i.pickle".format(args.data)
left_j_path = "./preprocessData/coordinate/vgg19/{}/left_j.pickle".format(args.data)

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
train_path = "./dataset/{}/train_resize/good/".format(args.data)
label_name = "./label/fullPatch/{}/{}/kmeans_{}_{}.pth".format(
    str(args.model),
    args.data,
    str(out),
    str(args.batch)
)
mask_path = "./dataset/big_mask/"

# label_sample_path = "label/{}/{}/train/{}_{}_argmax.pth".format(
#         args.model,
#         args.data,
#         str(args.kmeans),
#         str(args.batch)
#     )

# label_train_path = "label/{}/{}/train/{}_{}.pth".format(
#         args.model,
#         args.data,
#         str(args.kmeans),
#         str(args.batch)
#     )

if args.model == 'vgg19':
    pretrain_model = nn.DataParallel(pretrain_vgg.model).to(device)

if args.model == 'resnet34':
    pretrain_model = nn.DataParallel(pretrain_resnet.model).to(device)

print('training data: ', train_path)
print('training label: ', label_name)

""" testing """
if (args.type == 'good'):
    test_path = "./dataset/{}/test/good_resize".format(args.data)
    test_label_name = "preprocessData/label/{}/{}/test/good_{}_{}.pth".format(
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


""" the ground truth defective mask path """ 
# defect_gt_path = "dataset/{}/ground_truth/{}_resize/".format(args.data, args.type)

print('testing data: ', test_path)
print('testing label: ', test_label_name)

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
        
        img_feature = []
        total_gt = []
        total_idx = []

        for ((idx, img), (idx2, img2)) in zip(test_loader, mask_loader):
            img = img.to(device)
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
                    label = test_label[idx][i*16+j].to(device)
                        
                    # label = test_label[idx*1024 + i*32+j].to(device, dtype=torch.float)
                    # print(label)
                   
                    xs.append(x)
                    ys.append(label)
            
                x = torch.cat(xs, 0)
                y = torch.stack(ys).squeeze().to(device)

                output = model(x)
                # y_ = output.argmax(-1).detach()
                y_ = output.argmax(-1).detach().cpu().numpy()

                for k in range(16):
                    label_idx.append(y_[k])
                    label_gt.append(y[k].item())
                    output_center = kmeans.cluster_centers_[y_[k]]
                    output_center = np.reshape(output_center, (1, -1))
                    output_center = pil_to_tensor(output_center).to(device)
                    output_center = torch.squeeze(output_center)

                    if y_[k] == y[k].item():
                        isWrongLabel = 0
                    else:
                        isWrongLabel = 1

                    un_out = torch.unsqueeze(output[k], dim=0)
                    un_y = torch.unsqueeze(y[k], dim=0).long()
                    diff = isWrongLabel * nn.MSELoss()(output_center, crop_list[k])
                    diff_label = nn.CrossEntropyLoss()(un_out, un_y)
                    value_feature.append(diff.item())

                    writer.add_scalar('test_feature_loss', diff.item(), eval_fea_count)
                    writer.add_scalar('test_label_loss', diff_label.item(), eval_fea_count)

                    # print(f'Testing i={i} j={k} loss={diff.item()}')
                    
                    eval_fea_count += 1

                """ feature error """
                # diff = torch.mm( (y - output), torch.tensor(gmm.means_ ).to(device, dtype=torch.float)).sum(dim=1).abs() / 32
                # print(f'Testing i={i} feature loss={diff.sum().item()}')

                """ alpha error """
                # diff_alpha = nn.MSELoss()(output[k], y[k])
                # value_label.append(diff_alpha.item())
                # loss_alpha += diff_alpha.item()
            
                """ loss unit: per patch """
                # loss /= 32
                # writer.add_scalar('test_loss', diff.sum().item(), eval_fea_count)
                # loss_alpha /= 32
                # writer.add_scalar('test_alpha_loss', loss_alpha, eval_fea_count)
                # eval_fea_count += 1

            total_gt.append(label_gt)
            total_idx.append(label_idx)
            img_feature.append(value_feature)


    img_feature = np.array(img_feature).reshape((len(test_loader), -1))
    total_gt = np.array(total_gt).reshape((len(test_loader), -1))
    total_idx = np.array(total_idx).reshape((len(test_loader), -1))

    return img_feature, total_gt, total_idx
    

def SoftCrossEntropy(input, target):
    log_likelihood = -F.log_softmax(input, dim=1)
    batch = input.shape[0]
    loss = torch.sum(torch.mul(log_likelihood, target)) / batch
    return loss

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

                if (iter_count % 88 == 0):
                    img_feature, total_gt, total_idx = eval_feature(epoch+epoch_num, scratch_model, test_loader, mask_loader)
                iter_count += 1
            

    return scratch_model, img_feature, total_gt, total_idx


if __name__ == "__main__":

    """ Summary Writer """
    writer = SummaryWriter(comment='_Size1024_WeightSample_exp1_' + args.data + '_' + args.type)
  
    """ normal training data """
    # train_dataset = dataloaders.MvtecLoader(train_path)
    # train_loader = DataLoader(train_dataset, batch_size=args.train_batch)

    """ weight sampling with noise patch in training data """
    train_dataset = dataloaders.NoisePatchDataloader(train_path, label_name, left_i_path, left_j_path)
    samples_weights = torch.from_numpy(train_dataset.samples_weights)
    sampler = WeightedRandomSampler(samples_weights.type('torch.DoubleTensor'), len(samples_weights))
    train_loader = DataLoader(train_dataset, batch_size=args.train_batch, num_workers=1, sampler=sampler)


    test_dataset = dataloaders.MvtecLoader(test_path)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    mask_dataset = dataloaders.MaskLoader(defect_gt_path)
    mask_loader = DataLoader(mask_dataset, batch_size=1, shuffle=False)

    scratch_model = nn.DataParallel(scratch_model).to(device)
    if (args.pretrain == 'True'):
        scratch_model.load_state_dict(torch.load('models/{}/{}/exp1_128_10.ckpt'.format(
            args.model, 
            args.data
            )   
        ))
        print("--- Load pretrain model ---")
        epoch_num = 10
    else:
        epoch_num = 0

    """ training config """ 
    # criterion = nn.MSELoss()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(scratch_model.parameters(), lr = args.lr)
    
    iter_count = 1
    
    for epoch in range(args.epoch):

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

            if (iter_count % 50 == 0):
                img_feature, total_gt, total_idx = eval_feature(epoch+epoch_num, scratch_model, test_loader, mask_loader)
            
            iter_count += 1

        # img_feature, total_gt, total_idx = eval_feature(epoch+epoch_num, scratch_model, test_loader, mask_loader)
        # draw_errorMap(img_feature, total_gt, total_idx, epoch, scratch_model, test_loader, mask_loader, args.data, args.type)
        
        path = "models/{}/{}/exp1_{}_{}.ckpt".format(
            args.model, 
            args.data, 
            str(out), 
            str(epoch+1+epoch_num)
        )

        """ save model and optimizer """
        torch.save({
            'model_state_dict': scratch_model.state_dict(), 
            'optim_state_dict': optimizer.state_dict(),
            }, path)

