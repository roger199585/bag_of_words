import torch
import pickle

import argparse
import numpy as np
import os

from ei import patch
patch(select=True)

from config import ROOT
""" set parameters """
parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='ALL')
parser.add_argument('--kmeans', type=int, default=128)
parser.add_argument('--patch_size', type=int, default=64)
parser.add_argument('--image_size', type=int, default=1024)
parser.add_argument('--dim_reduction', type=str, default='PCA')
args = parser.parse_args()

total = ['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather', 'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor', 'wood', 'zipper']

if args.data == "ALL":
    for idx, data in enumerate(total):
        train_label_name = "{}/preprocessData/label/vgg19/{}/train/{}_100.pth".format(ROOT, data, args.kmeans)
        train_label = torch.tensor(torch.load(train_label_name))

        origin_features_path = "{}/preprocessData/chunks/vgg19/chunks_{}_train.pickle".format(ROOT, data)
        origin_features = pickle.load(open(origin_features_path, "rb"))
        origin_features = np.array(origin_features)

        center_features = np.zeros((args.kmeans, origin_features.shape[1]) )
        count = np.zeros(args.kmeans)

        chunk_num = int(args.image_size / args.patch_size)
        for i in range(train_label.shape[0]):
            for j in range(train_label.shape[1]):
                center_features[train_label[i][j][0]] += origin_features[i * chunk_num * chunk_num + j]
                count[train_label[i][j][0]] += 1

        for i in range(args.kmeans):
            # print("cluster center {}'s feature is".format(i))
            # print( center_features[i] / count[i] )
            center_features[i] = center_features[i] / count[i]

        save_path = "{}/preprocessData/cluster_center/{}/".format(ROOT, args.kmeans)
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        save_name = "{}.pickle".format(data)
        pickle.dump(center_features, open(save_path+save_name, "wb"))
        print("save center feature for {}".format(data))
else:
    train_label_name = f"{ ROOT }/preprocessData/label/vgg19/{ args.dim_reduction }/{ args.data }/train/{ args.kmeans }_100.pth"
    train_label = torch.tensor(torch.load(train_label_name))

    origin_features_path = f"{ ROOT }/preprocessData/chunks/vgg19/chunks_{ args.data }_train.pickle"
    origin_features = pickle.load(open(origin_features_path, "rb"))
    origin_features = np.array(origin_features)


    center_features = np.zeros((args.kmeans, origin_features.shape[1]) )
    count = np.zeros(args.kmeans)

    chunk_num = int(args.image_size / args.patch_size)
    for i in range(train_label.shape[0]):
        for j in range(train_label.shape[1]):
            center_features[train_label[i][j][0]] += origin_features[i * chunk_num * chunk_num + j]
            count[train_label[i][j][0]] += 1

    for i in range(args.kmeans):
        # print("cluster center {}'s feature is".format(i))
        # print( center_features[i] / count[i] )
        # print(count[i])
        center_features[i] = center_features[i] / count[i]

    print(center_features.shape)

    save_path = f"{ ROOT }/preprocessData/cluster_center/{ args.kmeans }/"
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    save_name = "{}.pickle".format(args.data)
    pickle.dump(center_features, open(save_path+save_name, "wb"))
    print("save center feature for {}".format(args.data))
