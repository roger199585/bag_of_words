import torch

import os
import numpy as np

from config import ROOT

allTypes = os.listdir(f'{ROOT}/preprocessData/label/fullPatch/vgg19')

# for _type in allTypes:
for _type in ['tile']:
    try:
        label_path = f'{ROOT}/preprocessData/label/fullPatch/vgg19/{_type}/kmeans_64_100.pth'
        label_list = torch.load(label_path)

        class_sample_count = np.array([len(np.where(label_list==t)[0]) for t in np.unique(label_list)])

        if class_sample_count.shape[0] != 64:
            # print(f'{_type} 分群失敗')
            print(0)
        else:
            print(1)
    except:
        print(0)