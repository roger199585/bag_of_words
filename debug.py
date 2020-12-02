# import pickle

# from config import ROOT

# DATA = 'capsule'
# ## Cluster Center Features
# center_features_path = "{}/preprocessData/cluster_center/128/{}.pickle".format(ROOT, DATA)
# cluster_features = pickle.load(open(center_features_path, "rb"))

# ## 經常誤判的兩群
# cluster_features[26]
# cluster_features[36]


"""
    Author: Yong Yu Chen
    Collaborator: Corn

    Update: 2020/12/2
    History: 
        2020/12/2 -> code refactor

    Description: This file is to assign label to each patch, includes the patch in training set and testing set
"""

""" STD Library """
import os
import sys
import pickle
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm

""" sklearn Library """
from sklearn.decomposition import PCA

""" Pytorch Library """
import torch
import torchvision
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

def save_img():
    print(args)
    

if __name__ == "__main__":
    """ set parameters """ 
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='bottle')
    parser.add_argument('--test', type=str, default='test')
    
    args = parser.parse_args()

    save_img()

