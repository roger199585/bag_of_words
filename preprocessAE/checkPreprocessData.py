import torch

import os
import numpy as np

from config import ROOT

allTypes = os.listdir(f'{ ROOT }/preprocessData/label/fullPatch/AE')

for _type in allTypes:
    resolutions = os.listdir(f'{ ROOT }/preprocessData/label/fullPatch/AE/{ _type }/')
    for resolution in resolutions:
        label_path = f'{ ROOT }/preprocessData/label/fullPatch/AE/{ _type }/{ resolution }/kmeans_128.pth'
        label_list = torch.load(label_path)

        class_sample_count = np.array([len(np.where(label_list==t)[0]) for t in np.unique(label_list)])

        if class_sample_count.shape[0] != 128:
            print(class_sample_count.shape[0])
            print(f'{_type} { resolution } 分群失敗')