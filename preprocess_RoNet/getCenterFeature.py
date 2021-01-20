import torch
import pickle

import argparse
import numpy as np
import os

from config import ROOT

if __name__ == "__main__":
    """ set parameters """
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='bottle')
    parser.add_argument('--kmeans', type=int, default=128)
    parser.add_argument('--patch_size', type=int, default=64)
    parser.add_argument('--image_size', type=int, default=1024)
    args = parser.parse_args()

    train_label_name     = f"{ ROOT }/preprocessData/label/RoNet/{ args.data }/train/{ args.kmeans }.pth"
    train_label          = torch.tensor(torch.load(train_label_name))

    origin_features_path = f"{ ROOT }/preprocessData/chunks/RoNet/{ args.data }/chunks_{ args.data }_train.pickle".format(ROOT, args.data)
    origin_features      = pickle.load(open(origin_features_path, "rb"))
    origin_features      = np.array(origin_features)


    center_features      = np.zeros((args.kmeans, origin_features.shape[1]) )
    count                = np.zeros(args.kmeans)

    chunk_num = int(args.image_size / args.patch_size)
    for i in range(train_label.shape[0]):
        for j in range(train_label.shape[1]):
            center_features[train_label[i][j][0]] += origin_features[i * chunk_num * chunk_num + j]
            count[train_label[i][j][0]] += 1

    for i in range(args.kmeans):
        print("cluster center {}'s feature is".format(i))
        print( center_features[i] / count[i] )
        center_features[i] = center_features[i] / count[i]

    print(center_features.shape)

    save_path = f"{ ROOT }/preprocessData/cluster_center/RoNet/{ args.data }/"
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    save_name = f"{ args.kmeans }.pickle"
    pickle.dump(center_features, open(save_path+save_name, "wb"))
    print(f"save center feature for { args.data }")
