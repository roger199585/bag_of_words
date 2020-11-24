#!/bin/bash

# cd /workspace

# 進行圖片大小的轉換
# echo "Preparing images, covert all image in dataset into 1024x1024"
# python preprocess.py --types transistor

# 建立 chunks and coordinates (切 chunk 以及讓他有位移)
python pretrain_vgg.py --data wood

# 透過上一步切好的資料給 kmeans 分群
python BoW_PCA.py --data wood --kmeans 16 --dim 128

# # 給定每個 patch 的 label
python assign_idx.py --data wood --kmeans 16 --dim 128 --type train &
python assign_idx.py --data wood --kmeans 16 --dim 128 --type test &
python assign_idx.py --data wood --kmeans 16 --dim 128 --type all &

wait
# # 我不知道這邊在幹嘛 我就爛
python dataloaders.py --data wood --kmeans 16