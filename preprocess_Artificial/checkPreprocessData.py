import torch

import os
import argparse
import numpy as np

from config import ROOT

# allTypes = os.listdir(f"{ ROOT }/preprocessData/label/fullPatch/artificial")

# for _type in allTypes:
if __name__ == "__main__":
    """ set parameters """
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='bottle')
    parser.add_argument('--kmeans', type=int, default=128)
    args = parser.parse_args()
    
    label_path = f"{ ROOT }/preprocessData/label/fullPatch/artificial/{ args.data }/kmeans_128.pth"
    label_list = torch.load(label_path)

    class_sample_count = np.array([len(np.where(label_list==t)[0]) for t in np.unique(label_list)])

    if class_sample_count.shape[0] != args.kmeans:
        print(1)
    else:
        print(0)