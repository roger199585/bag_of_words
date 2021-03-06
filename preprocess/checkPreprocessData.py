import torch

import os
import numpy as np

from config import ROOT

allTypes = os.listdir(f'{ROOT}/preprocessData/label/fullPatch/vgg19')

for _type in allTypes:
# for _type in ['toothbrush']:
    # try:
    label_path = f'{ROOT}/preprocessData/label/fullPatch/vgg19/{_type}/kmeans_128_100.pth'
    label_list = torch.load(label_path)

    class_sample_count = np.array([len(np.where(label_list==t)[0]) for t in np.unique(label_list)])

    if class_sample_count.shape[0] != 128:
        print(class_sample_count.shape[0])
        print(f'{_type} 分群失敗')
        # print(0)
    else:
        # print(f'{_type} 分群成功')
        print(1)
    # except:
    #     print(0)