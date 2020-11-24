import torch
import pickle

import numpy as np

train_label_name = "./preprocessData/label/vgg19/grid/train/128_100.pth"
train_label = torch.tensor(torch.load(train_label_name))

origin_features_path = "./preprocessData/chunks/vgg19/chunks_grid_train.pickle"
origin_features = pickle.load(open(origin_features_path, "rb"))
origin_features = np.array(origin_features)

print(np.array(origin_features).shape)
print(train_label.shape)

center_features = np.zeros((128, origin_features.shape[1]) )
count = np.zeros(128)

for i in range(train_label.shape[0]):
    for j in range(train_label.shape[1]):
        center_features[train_label[i][j][0]] += origin_features[i * 256 + j]
        count[train_label[i][j][0]] += 1

for i in range(128):
    print("cluster center {}'s feature is".format(i))
    print( center_features[i] / count[i] )
    center_features[i] = center_features[i] / count[i]