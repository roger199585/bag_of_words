#!/bin/bash

cd $PWD
# 進行圖片大小的轉換
# echo "Preparing images, covert all image in dataset into 1024x1024"
# python preprocess.py --types transistor

# 建立 chunks and coordinates (切 chunk 以及讓他有位移)
# python pretrain_vgg.py --data cable
# python pretrain_vgg.py --data capsule
# python pretrain_vgg.py --data toothbrush
# python pretrain_vgg.py --data zipper

# 透過上一步切好的資料給 kmeans 分群
# python BoW_PCA.py --data leather --kmeans 256

# 給定每個 patch 的 label
# python assign_idx.py --data leather --kmeans 256 --type train
# python assign_idx.py --data leather --kmeans 256 --type test
# python assign_idx.py --data leather --kmeans 256 --type all

# 我不知道這邊在幹嘛 我就爛
# python dataloaders.py --data leather --kmeans 256

# 找出 kmeans cluster center 的 feature 
python getCenterFeature.py --data bottle --kmeans 128 &
python getCenterFeature.py --data cable --kmeans 128 &
python getCenterFeature.py --data capsule --kmeans 128 &
python getCenterFeature.py --data carpet --kmeans 128 &
python getCenterFeature.py --data grid --kmeans 128 &
python getCenterFeature.py --data hazelnut --kmeans 128 &
python getCenterFeature.py --data leather --kmeans 128 &
python getCenterFeature.py --data metal_nut --kmeans 128 &
python getCenterFeature.py --data pill --kmeans 128 &
python getCenterFeature.py --data screw --kmeans 128 &
python getCenterFeature.py --data tile --kmeans 128 &
python getCenterFeature.py --data toothbrush --kmeans 128 &
python getCenterFeature.py --data transistor --kmeans 128 &
python getCenterFeature.py --data wood --kmeans 128 &
python getCenterFeature.py --data zipper --kmeans 128 &

wait

