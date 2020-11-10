import torch
import torch.nn as nn
from torchvision import transforms

import os 
import cv2
import numpy as np
import matplotlib.pyplot as plt


from sklearn import preprocessing
from sklearn.metrics import roc_auc_score, average_precision_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pil_to_tensor = transforms.ToTensor()

def one_hot(label, cluster, number):
    label = label.view(-1, 1)
    label_onehot = torch.FloatTensor(number, cluster).to(device)
    label_onehot.zero_()
    label_onehot.scatter_(1, label, 1)
    return label_onehot

def one_hot_forMap(label, cluster):
    label = np.array(label)
    oneHot = np.eye(cluster)[label]
    return oneHot


def errorMap_forTrain(idx, img, pretrain_model, model, pca, kmeans, index_label, epoch):
    label_idx = []
    label_gt = []
    value = []

    pretrain_model.train()
    model.train()
    
    with torch.no_grad():
        for i in range(32):
            for j in range(32):
                crop_img = img[:, :, i*8:i*8+8, j*8:j*8+8].to(device)
                # crop_img = img[:, :, j*8:j*8+8, i*8:i*8+8].to(device)
                crop_output = pretrain_model(crop_img)

                crop_output = crop_output.detach().cpu().numpy()
                crop_output = np.squeeze(crop_output).reshape((1, -1))
                crop_output = pca.transform(crop_output)
                crop_output = torch.from_numpy(crop_output).squeeze().to(device)

                mask = torch.ones(1, 1, 256, 256)
                mask[:, :, i*8:i*8+8, j*8:j*8+8] = 0
                # mask[:, :, j*8:j*8+8, i*8:i*8+8] = 0
                mask = mask.to(device)
                
                x = img * mask
                x = torch.cat((x, mask), 1)
                label = index_label[idx][i*32+j].to(device)
                output = model(x)

                y = output.argmax(-1).detach()
                label_idx.append(y.item())
                label_gt.append(label.item())

                output_center = kmeans.cluster_centers_[y]
                output_center = np.reshape(output_center, (1, -1))
                output_center = pil_to_tensor(output_center).to(device)
                output_center = torch.squeeze(output_center)

                diff = nn.MSELoss()(output_center, crop_output)
                value.append(diff.item())
    
    
        value = preprocessing.minmax_scale(value, feature_range=(0,1), axis=0)
                
        error_map = np.zeros((256, 256))
        for index, scalar in enumerate(value):
            mask = cv2.imread('dataset/big_mask/mask{}.png'.format(index), cv2.IMREAD_GRAYSCALE)
            mask = np.invert(mask)
            mask[mask==255]=1
            error_map += mask * scalar

        img_ = np.squeeze(img.detach().cpu().numpy()).transpose((1,2,0))
        ironman_grid = plt.GridSpec(1, 2)
        fig = plt.figure(figsize=(12,6), dpi=100)
        ax1 = fig.add_subplot(ironman_grid[0,0])
        im1 = ax1.imshow(error_map, cmap="Blues")
        # plt.colorbar(im1, extend='neither',ax=ax1)
        # im1.set_clim(0, 1) # 這邊設定差值的上下界
        ax2 = fig.add_subplot(ironman_grid[0,1])
        im2 = ax2.imshow(img_, cmap="Blues")


        """ add label text to each patch """ 
        for i in range(32):
            for j in range(32):
                ax1.text((j+0.2)*8, (i+0.6)*8, label_idx[i*32+j], fontsize=5)
                ax2.text((j+0.2)*8, (i+0.6)*8, label_gt[i*32+j], fontsize=5)

        plt.colorbar(im1, extend='both')
        im1.set_clim(0, 1) 
        im2.set_clim(0, 1) 
        pathToSave = "./errorMap_train/{}.png".format(
            str(epoch)
        )

        plt.savefig(pathToSave, dpi=100)
        plt.close(fig)

def addMaskEdge(image_path, maskArea, maskEdge, dist):
    image = cv2.imread(image_path)

    mask_area = cv2.imread(maskArea, cv2.IMREAD_GRAYSCALE)
    mask_edge = cv2.imread(maskEdge, cv2.IMREAD_GRAYSCALE)

    edge_contours, _ = cv2.findContours(mask_edge, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    area_contours, _ = cv2.findContours(mask_area, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image, area_contours, -1, (0,255,0, 0.2), -1)
    cv2.drawContours(image, edge_contours, -1, (0,0,255), 1)
    cv2.imwrite(dist, image)

def createImg(value, maxVal, minVal, epoch):

    thres = (maxVal + minVal) / 2
    print(thres)

    img_list = []
    for i in range(len(value)):
        if (value[i] > thres):
            """ black """
            frame = np.zeros((8, 8, 3), np.uint8)
        else:
            """ white """
            frame = 255 * np.ones((8, 8, 3), np.uint8)
        frame = Image.fromarray(frame)
        img_list.append(frame)

    newImg = Image.new('RGB', (256, 256), (255,255,255))

    x_= 0
    y_= 0
    for i in range(32):
        for j in range(32):
            newImg.paste(img_list[i*8+j], (x_, y_))
            x_ += 8

        x_ = 0
        y_ += 8

    img_path = 'result/' + args.data + '/'
    if not os.path.isdir(img_path):
                os.makedirs(img_path)
    img_name = 'result_' + str(epoch) + '.png'
    newImg.save(img_path+img_name, 'PNG')


""" draw errorMap for label"""
    # img = img.detach().cpu().numpy().squeeze().transpose(1,2,0)
    # error_map = errorMap('./dataset/big_mask', inverse=True, autoSave=False)
    # # print(mask_dir[idx])
    # if (args.type == 'bad_small'):
    #     gt_path = './dataset/bottle/ground_truth/broken_small_resize/'
    # elif (args.type == 'bad_large'):
    #     gt_path = './dataset/bottle/ground_truth/broken_large_resize/'
    # gt = gt_path + mask_dir[idx]
    # result = error_map.generateMap(img, gt, outs)
    # pathToSave = './labelMap/'+ args.data + '/' + args.type + '/errorMap_' + str(idx) + '_' + str(epoch) + '.png'
    # error_map.saveMap(result, pathToSave)






def eval_feature_batch1(epoch, model, test_loader, mask_loader):
    global eval_fea_count
    global norm_count
    global pretrain_model
    global kmeans
    global mask_dir

    model.eval()
    pretrain_model.eval()
    split_size = 8

    # value = []
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
                for j in range(32):
                    crop_img = img[:, :, i*8:i*8+8, j*8:j*8+8].to(device)
                    # crop_img = img[:, :, j*8:j*8+8, i*8:i*8+8].to(device)
                    crop_output = pretrain_model(crop_img)
                    # crop_output = torch.squeeze(crop_output)

                    crop_output = crop_output.detach().cpu().numpy()
                    crop_output = np.squeeze(crop_output).reshape((1, -1))
                    crop_output = pca.transform(crop_output)
                    crop_output = torch.from_numpy(crop_output).squeeze().to(device)

                    mask = torch.ones(1, 1, 256, 256)
                    mask[:, :, i*8:i*8+8, j*8:j*8+8] = 0
                    mask = mask.to(device)
                    
                    x = img * mask
                    x = torch.cat((x, mask), 1)
                    label = test_label[idx][i*32+j].to(device)

                    output = model(x)

                    y_ = output.argmax(-1).detach()

                    label_idx.append(y_.item())
                    label_gt.append(label.item())
       
                    output_center = kmeans.cluster_centers_[y_]
                    output_center = np.reshape(output_center, (1, -1))
                    output_center = pil_to_tensor(output_center).to(device)
                    output_center = torch.squeeze(output_center)

                    diff = nn.MSELoss()(output_center, crop_output)
                    value.append(diff.item())
                    loss += diff


            loss /= 1024
            #value.append(loss)
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


def draw_errorMap(value_feature, total_gt, total_idx, epoch, model, test_loader, mask_loader, test_data, test_type, writer):
    with torch.no_grad():
        label_pred = []
        label_true = []

        for ((idx, img), (idx2, img2)) in zip(test_loader, mask_loader):  
            img = img.to(device)
            idx = idx[0].item()

            # value = preprocessing.minmax_scale(value_feature, feature_range=(0,1), axis=1)
            error_map = np.zeros((1024, 1024))
            for index, scalar in enumerate(value_feature[idx]):
                mask = cv2.imread('dataset/big_mask/mask{}.png'.format(index), cv2.IMREAD_GRAYSCALE)
                mask = np.invert(mask)
                mask[mask==255]=1
                
                error_map += mask * scalar
            # if (test_type == 'good'):
            #     img_ = np.squeeze(img.detach().cpu().numpy()).transpose((1,2,0))
            #     ironman_grid = plt.GridSpec(1, 2)
            #     fig = plt.figure(figsize=(12,6), dpi=100)
            #     ax1 = fig.add_subplot(ironman_grid[0,0])
            #     im1 = ax1.imshow(error_map, cmap="Blues")
            #     ax2 = fig.add_subplot(ironman_grid[0,1])
            #     im2 = ax2.imshow(img_)
            # else:
            #     img_ = np.squeeze(img.detach().cpu().numpy()).transpose((1,2,0))
            #     defect_gt = np.squeeze(img2.cpu().numpy()).transpose((1,2,0))
            #     ironman_grid = plt.GridSpec(1, 3)
            #     fig = plt.figure(figsize=(18,6), dpi=100)
            #     ax1 = fig.add_subplot(ironman_grid[0,0])
            #     im1 = ax1.imshow(error_map, cmap="Blues")
            #     ax2 = fig.add_subplot(ironman_grid[0,1])
            #     ax3 = fig.add_subplot(ironman_grid[0,2])
            #     im2 = ax2.imshow(img_)
            #     im3 = ax3.imshow(defect_gt)
                
            defect_gt = np.squeeze(img2.cpu().numpy()).transpose((1,2,0))
            true_mask = defect_gt[:, :, 0].astype('int32') 

            label_pred.append(error_map)
            label_true.append(true_mask)
            # ax1.set_title(auc)


            # """ add label text to each patch """ 
            # for i in range(16):
            #     for j in range(16):
            #         ax1.text((j+0.2)*64, (i+0.6)*64, total_idx[idx][i*16+j], fontsize=10)
            #         ax2.text((j+0.2)*64, (i+0.6)*64, total_gt[idx][i*16+j], fontsize=10)


            # errorMapPath = "errorMap_1024_exp1/vgg19/{}/{}/".format(
            #     test_data,
            #     test_type
            # )
            # if not os.path.isdir(errorMapPath):
            #     os.makedirs(errorMapPath)
            #     print("----- create folder for type:{} -----".format(test_type))
            
            # errorMapName = "{}_{}.png".format(
            #     str(idx),
            #     str(epoch) 
            # )

            # plt.savefig(errorMapPath+errorMapName, dpi=100)
            # plt.close(fig)
            
            print(f'EP={epoch} img_idx={idx}')
        
        auc = roc_auc_score(np.array(label_true).flatten(), np.array(label_pred).flatten())
        print(auc)
        writer.add_scalars('eval_score', {
            'roc_auc': auc
        }, epoch)


from math import log, e
def entropy2(labels, base=None):
    """ Computes entropy of label distribution. """

    n_labels = len(labels)

    if n_labels <= 1:
        return 0

    value,counts = np.unique(labels, return_counts=True)
    probs = counts / n_labels
    print(probs)
    n_classes = np.count_nonzero(probs)

    if n_classes <= 1:
        return 0

    ent = 0.

    # Compute entropy
    base = e if base is None else base
    for i in probs:
        print(i, ent)
        ent -= i * log(i, base)

    return ent

from scipy.stats import entropy
def entropy1(labels, base=e):

    value,counts = np.unique(labels, return_counts=True)
    return entropy(counts, base=base)
    
def compute_entropy():
    alpha_path = "/home/dinosaur/bag_of_words/label/vgg19/bottle/train/16_100_gmm.pth"
    alpha = torch.load(alpha_path)
    entropy_list = []

    f = open("entropy.txt", "w")

    ironman_grid = plt.GridSpec(1, 1)
    fig = plt.figure(figsize=(6,6), dpi=100)
    ax1 = fig.add_subplot(ironman_grid[0,0])

    # for i in range(32):
    #     for j in range(32):
    #         """ img = 0 """
    #         label = alpha[i*32+j]
    #         label = np.squeeze(label).tolist()
    #         ent = entropy2(label)
    #         entropy_list.append(ent)

    #         f.write("%.5f\r" % (ent))
    #         # print(ent)
    #         ax1.text((j+0.2)*8, (i+0.6)*8, ent, fontsize=10)
            
            # if (ent > 0.9):
            #     print("case: ent > 0.9: " , i, j)
            # elif (ent > 0.6 and ent < 0.9):
            #     print("case: ent=0.6-0.9: ", i, j)
        
    #     f.write("\n")
    
    # f.close()

    plt.savefig("ent.png", dpi=100)
    plt.close(fig)
    value,counts = np.unique(entropy_list, return_counts=True)
    
    # print(value, counts)

    a = [9.9859e-01, 0.0000e+00, 0.0000e+00, 0.0000e+00, 2.2301e-04, 4.3254e-04,
        0.0000e+00, 0.0000e+00, 0.0000e+00, 2.7255e-04, 0.0000e+00, 0.0000e+00,
        0.0000e+00, 0.0000e+00, 4.8004e-04, 0.0000e+00]
    print(entropy2(a))
    print(entropy(a))


eval_count = 0

def eval(epoch, model, data_loader):
    global eval_count

    model.eval()

    with torch.no_grad():
        for idx, img in (data_loader):
            idx = idx[0].item()
            for i in range(32):

                xs = []
                ys = []
                value_label = []
                loss_alpha = 0.0
                acc = 0.0

                for j in range(32):
                    img = img.to(device)
                    mask = torch.ones(1, 1, 256, 256)
                    mask[:, :, i*8:i*8+8, j*8:j*8+8] = 0
                    mask = mask.to(device)
                    x = img * mask
                    x = torch.cat((x, mask), 1)

                    label = test_label[idx*1024+i*32+j].to(device)

                    xs.append(x)
                    ys.append(label)

                x = torch.cat(xs, 0)
                y = torch.stack(ys).squeeze().to(device)

                output = model(x)
                acc = (output.detach() == y).float().mean()

                print(f'EP={epoch} img_idx={idx} test_accuracy={acc}')
                writer.add_scalar('test_acc', acc, eval_count)

                diff_alpha = nn.MSELoss()(output, y)
                value_label.append(diff_alpha.item())
                loss_alpha += diff_alpha.item()
                
                loss_alpha /= 32
                writer.add_scalar('test_alpha_loss', loss_alpha, eval_count)

                eval_count += 1

                # loss = nn.MSELoss()(output, y)
                # writer.add_scalar('test_loss_label', loss.item(), eval_count_forloss)
                # eval_count_forloss += 1

                # acc += (output.argmax(-1).detach() == y).float().mean()

            # acc = 100.0 * acc / 32
           