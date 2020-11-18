import torch
import pickle

import argparse
import numpy as np
import os

from config import ROOT
""" set parameters """
parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='leather')
parser.add_argument('--kmeans', type=int, default=128)
args = parser.parse_args()


train_label_name = "{}/preprocessData/label/vgg19/{}/train/{}_100.pth".format(ROOT, args.data, args.kmeans)
train_label = torch.tensor(torch.load(train_label_name))

origin_features_path = "{}/preprocessData/chunks/vgg19/chunks_{}_train.pickle".format(ROOT, args.data)
origin_features = pickle.load(open(origin_features_path, "rb"))
origin_features = np.array(origin_features)

print(np.array(origin_features).shape)
print(train_label.shape)

center_features = np.zeros((args.kmeans, origin_features.shape[1]) )
count = np.zeros(args.kmeans)

for i in range(train_label.shape[0]):
    for j in range(train_label.shape[1]):
        center_features[train_label[i][j][0]] += origin_features[i * 256 + j]
        count[train_label[i][j][0]] += 1

for i in range(args.kmeans):
    print("cluster center {}'s feature is".format(i))
    print( center_features[i] / count[i] )
    center_features[i] = center_features[i] / count[i]

print(center_features.shape)

save_path = "{}/preprocessData/cluster_center/{}/".format(ROOT, args.kmeans)
if not os.path.isdir(save_path):
    os.makedirs(save_path)
save_name = "{}.pickle".format(args.data)
pickle.dump(center_features, open(save_path+save_name, "wb"))
print("save center feature for {}".format(args.data))
