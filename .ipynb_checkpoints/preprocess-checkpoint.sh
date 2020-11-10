#!/bin/bash

# 先進到我們的專案目錄
cd ~/AFS/bag_of_words

# 進行圖片大小的轉換
# echo "Preparing images, covert all image in dataset into 1024x1024"
python preprocess.py --types ALL

# 建立 chunks and coordinates (切 chunk 以及讓他有位移)
python pretrain_vgg.py --data bottle
# python pretrain_vgg.py --data cable
# python pretrain_vgg.py --data capsule
# python pretrain_vgg.py --data carpet
# python pretrain_vgg.py --data grid
# python pretrain_vgg.py --data hazelnut
# python pretrain_vgg.py --data leather
# python pretrain_vgg.py --data metal_nut
# python pretrain_vgg.py --data pill
# python pretrain_vgg.py --data screw
# python pretrain_vgg.py --data tile
# python pretrain_vgg.py --data toothbrush
# python pretrain_vgg.py --data transistor
# python pretrain_vgg.py --data wood
# python pretrain_vgg.py --data zipper

# 透過上一步切好的資料給 kmeans 分群
python BoW_PCA.py --data bottle --kmeans 128
# python BoW_PCA.py --data cable --kmeans 128
# python BoW_PCA.py --data capsule --kmeans 128
# python BoW_PCA.py --data carpet --kmeans 128
# python BoW_PCA.py --data grid --kmeans 128
# python BoW_PCA.py --data hazelnut --kmeans 128
# python BoW_PCA.py --data leather --kmeans 128
# python BoW_PCA.py --data metal_nut --kmeans 128
# python BoW_PCA.py --data pill --kmeans 128
# python BoW_PCA.py --data screw --kmeans 128
# python BoW_PCA.py --data tile --kmeans 128
# python BoW_PCA.py --data toothbrush --kmeans 128
# python BoW_PCA.py --data transistor --kmeans 128
# python BoW_PCA.py --data wood --kmeans 128
# python BoW_PCA.py --data zipper --kmeans 128

# 給定每個 patch 的 label
python assign_idx.py --data bottle --kmeans 128 --type train
python assign_idx.py --data bottle --kmeans 128 --type test
python assign_idx.py --data bottle --kmeans 128 --type all
# python assign_idx.py --data cable --kmeans 128
# python assign_idx.py --data capsule --kmeans 128
# python assign_idx.py --data carpet --kmeans 128
# python assign_idx.py --data grid --kmeans 128
# python assign_idx.py --data hazelnut --kmeans 128
# python assign_idx.py --data leather --kmeans 128
# python assign_idx.py --data metal_nut --kmeans 128
# python assign_idx.py --data pill --kmeans 128
# python assign_idx.py --data screw --kmeans 128
# python assign_idx.py --data tile --kmeans 128
# python assign_idx.py --data toothbrush --kmeans 128
# python assign_idx.py --data transistor --kmeans 128
# python assign_idx.py --data wood --kmeans 128
# python assign_idx.py --data zipper --kmeans 128

# 我不知道這邊在幹嘛 我就爛
python dataloaders.py --data bottle --kmeans 128
# python dataloaders.py --data cable --kmeans 128
# python dataloaders.py --data capsule --kmeans 128
# python dataloaders.py --data carpet --kmeans 128
# python dataloaders.py --data grid --kmeans 128
# python dataloaders.py --data hazelnut --kmeans 128
# python dataloaders.py --data leather --kmeans 128
# python dataloaders.py --data metal_nut --kmeans 128
# python dataloaders.py --data pill --kmeans 128
# python dataloaders.py --data screw --kmeans 128
# python dataloaders.py --data tile --kmeans 128
# python dataloaders.py --data toothbrush --kmeans 128
# python dataloaders.py --data transistor --kmeans 128
# python dataloaders.py --data wood --kmeans 128
# python dataloaders.py --data zipper --kmeans 128