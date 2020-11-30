#!/bin/bash

cd $PWD
# 進行圖片大小的轉換
# echo "Preparing images, covert all image in dataset into 1024x1024"
# python preprocess.py --types transistor

# 建立 chunks and coordinates (切 chunk 以及讓他有位移)
# CUDA_VISIBLE_DEVICES=0 python pretrain_vgg.py --data bottle &
# CUDA_VISIBLE_DEVICES=1 python pretrain_vgg.py --data cable &
# CUDA_VISIBLE_DEVICES=2 python pretrain_vgg.py --data capsule &
# CUDA_VISIBLE_DEVICES=3 python pretrain_vgg.py --data carpet &
# wait
# CUDA_VISIBLE_DEVICES=0 python pretrain_vgg.py --data grid &
# CUDA_VISIBLE_DEVICES=1 python pretrain_vgg.py --data hazelnut &
# CUDA_VISIBLE_DEVICES=2 python pretrain_vgg.py --data leather &
# CUDA_VISIBLE_DEVICES=3 python pretrain_vgg.py --data metal_nut &
# wait
# CUDA_VISIBLE_DEVICES=0 python pretrain_vgg.py --data pill &
# CUDA_VISIBLE_DEVICES=1 python pretrain_vgg.py --data screw &
# CUDA_VISIBLE_DEVICES=2 python pretrain_vgg.py --data tile &
# CUDA_VISIBLE_DEVICES=3 python pretrain_vgg.py --data toothbrush &
# wait
# CUDA_VISIBLE_DEVICES=0 python pretrain_vgg.py --data transistor &
# CUDA_VISIBLE_DEVICES=1 python pretrain_vgg.py --data wood &
CUDA_VISIBLE_DEVICES=2 python pretrain_vgg.py --data zipper &
wait

# # 透過上一步切好的資料給 kmeans 分群
# python BoW_PCA.py --data bottle --kmeans 128 --dim 128 &
# python BoW_PCA.py --data cable --kmeans 128 --dim 128 &
# python BoW_PCA.py --data capsule --kmeans 128 --dim 128 &
# python BoW_PCA.py --data carpet --kmeans 128 --dim 128 &
# python BoW_PCA.py --data grid --kmeans 128 --dim 128 &
# wait
# python BoW_PCA.py --data hazelnut --kmeans 128 --dim 128 &
# python BoW_PCA.py --data leather --kmeans 128 --dim 128 &
# python BoW_PCA.py --data metal_nut --kmeans 128 --dim 128 &
# python BoW_PCA.py --data pill --kmeans 128 --dim 128 &
# python BoW_PCA.py --data screw --kmeans 128 --dim 128 &
# wait
# python BoW_PCA.py --data tile --kmeans 128 --dim 128 &
# python BoW_PCA.py --data toothbrush --kmeans 128 --dim 128 &
# python BoW_PCA.py --data transistor --kmeans 128 --dim 128 &
# python BoW_PCA.py --data wood --kmeans 128 --dim 128 &
python BoW_PCA.py --data zipper --kmeans 128 --dim 128 &
wait

# # 給定每個 patch 的 label
# CUDA_VISIBLE_DEVICES=0 python assign_idx.py --data bottle --dim 128 --kmeans 128 --type train &
# CUDA_VISIBLE_DEVICES=0 python assign_idx.py --data bottle --dim 128 --kmeans 128 --type test &
# CUDA_VISIBLE_DEVICES=0 python assign_idx.py --data bottle --dim 128 --kmeans 128 --type all &
# CUDA_VISIBLE_DEVICES=1 python assign_idx.py --data cable --dim 128 --kmeans 128 --type train &
# CUDA_VISIBLE_DEVICES=1 python assign_idx.py --data cable --dim 128 --kmeans 128 --type test &
# CUDA_VISIBLE_DEVICES=1 python assign_idx.py --data cable --dim 128 --kmeans 128 --type all &
# CUDA_VISIBLE_DEVICES=2 python assign_idx.py --data capsule --dim 128 --kmeans 128 --type train &
# CUDA_VISIBLE_DEVICES=2 python assign_idx.py --data capsule --dim 128 --kmeans 128 --type test &
# CUDA_VISIBLE_DEVICES=2 python assign_idx.py --data capsule --dim 128 --kmeans 128 --type all &
# CUDA_VISIBLE_DEVICES=3 python assign_idx.py --data carpet --dim 128 --kmeans 128 --type train &
# CUDA_VISIBLE_DEVICES=3 python assign_idx.py --data carpet --dim 128 --kmeans 128 --type test &
# CUDA_VISIBLE_DEVICES=3 python assign_idx.py --data carpet --dim 128 --kmeans 128 --type all &
# wait
# CUDA_VISIBLE_DEVICES=0 python assign_idx.py --data grid --dim 128 --kmeans 128 --type train &
# CUDA_VISIBLE_DEVICES=0 python assign_idx.py --data grid --dim 128 --kmeans 128 --type test &
# CUDA_VISIBLE_DEVICES=0 python assign_idx.py --data grid --dim 128 --kmeans 128 --type all &
# CUDA_VISIBLE_DEVICES=1 python assign_idx.py --data hazelnut --dim 128 --kmeans 128 --type train &
# CUDA_VISIBLE_DEVICES=1 python assign_idx.py --data hazelnut --dim 128 --kmeans 128 --type test &
# CUDA_VISIBLE_DEVICES=1 python assign_idx.py --data hazelnut --dim 128 --kmeans 128 --type all &
# CUDA_VISIBLE_DEVICES=2 python assign_idx.py --data leather --dim 128 --kmeans 128 --type train &
# CUDA_VISIBLE_DEVICES=2 python assign_idx.py --data leather --dim 128 --kmeans 128 --type test &
# CUDA_VISIBLE_DEVICES=2 python assign_idx.py --data leather --dim 128 --kmeans 128 --type all &
# CUDA_VISIBLE_DEVICES=3 python assign_idx.py --data metal_nut --dim 128 --kmeans 128 --type train &
# CUDA_VISIBLE_DEVICES=3 python assign_idx.py --data metal_nut --dim 128 --kmeans 128 --type test &
# CUDA_VISIBLE_DEVICES=3 python assign_idx.py --data metal_nut --dim 128 --kmeans 128 --type all &
# wait
# CUDA_VISIBLE_DEVICES=0 python assign_idx.py --data pill --dim 128 --kmeans 128 --type train &
# CUDA_VISIBLE_DEVICES=0 python assign_idx.py --data pill --dim 128 --kmeans 128 --type test &
# CUDA_VISIBLE_DEVICES=0 python assign_idx.py --data pill --dim 128 --kmeans 128 --type all &
# CUDA_VISIBLE_DEVICES=1 python assign_idx.py --data screw --dim 128 --kmeans 128 --type train &
# CUDA_VISIBLE_DEVICES=1 python assign_idx.py --data screw --dim 128 --kmeans 128 --type test &
# CUDA_VISIBLE_DEVICES=1 python assign_idx.py --data screw --dim 128 --kmeans 128 --type all &
# CUDA_VISIBLE_DEVICES=2 python assign_idx.py --data tile --dim 128 --kmeans 128 --type train &
# CUDA_VISIBLE_DEVICES=2 python assign_idx.py --data tile --dim 128 --kmeans 128 --type test &
# CUDA_VISIBLE_DEVICES=2 python assign_idx.py --data tile --dim 128 --kmeans 128 --type all &
# CUDA_VISIBLE_DEVICES=3 python assign_idx.py --data toothbrush --dim 128 --kmeans 128 --type train &
# CUDA_VISIBLE_DEVICES=3 python assign_idx.py --data toothbrush --dim 128 --kmeans 128 --type test &
# CUDA_VISIBLE_DEVICES=3 python assign_idx.py --data toothbrush --dim 128 --kmeans 128 --type all &
# wait
# CUDA_VISIBLE_DEVICES=0 python assign_idx.py --data transistor --dim 128 --kmeans 128 --type train &
# CUDA_VISIBLE_DEVICES=0 python assign_idx.py --data transistor --dim 128 --kmeans 128 --type test &
# CUDA_VISIBLE_DEVICES=0 python assign_idx.py --data transistor --dim 128 --kmeans 128 --type all &
# CUDA_VISIBLE_DEVICES=1 python assign_idx.py --data wood --dim 128 --kmeans 128 --type train &
# CUDA_VISIBLE_DEVICES=1 python assign_idx.py --data wood --dim 128 --kmeans 128 --type test &
# CUDA_VISIBLE_DEVICES=1 python assign_idx.py --data wood --dim 128 --kmeans 128 --type all &
CUDA_VISIBLE_DEVICES=2 python assign_idx.py --data zipper --dim 128 --kmeans 128 --type train &
CUDA_VISIBLE_DEVICES=2 python assign_idx.py --data zipper --dim 128 --kmeans 128 --type test &
CUDA_VISIBLE_DEVICES=2 python assign_idx.py --data zipper --dim 128 --kmeans 128 --type all &
wait

# 我不知道這邊在幹嘛 我就爛
# python dataloaders.py --data bottle --kmeans 128 &
# python dataloaders.py --data cable --kmeans 128 &
# python dataloaders.py --data capsule --kmeans 128 &
# python dataloaders.py --data carpet --kmeans 128 &
# python dataloaders.py --data grid --kmeans 128 &
# python dataloaders.py --data hazelnut --kmeans 128 &
# python dataloaders.py --data leather --kmeans 128 &
# python dataloaders.py --data metal_nut --kmeans 128 &
# python dataloaders.py --data pill --kmeans 128 &
# python dataloaders.py --data screw --kmeans 128 &
# python dataloaders.py --data tile --kmeans 128 &
# python dataloaders.py --data toothbrush --kmeans 128 &
# python dataloaders.py --data transistor --kmeans 128 &
# python dataloaders.py --data wood --kmeans 128 &
python dataloaders.py --data zipper --kmeans 128 &
wait

# 找出 kmeans cluster center 的 feature 
# python getCenterFeature.py --data bottle --kmeans 128 &
# python getCenterFeature.py --data cable --kmeans 128 &
# python getCenterFeature.py --data capsule --kmeans 128 &
# python getCenterFeature.py --data carpet --kmeans 128 &
# python getCenterFeature.py --data grid --kmeans 128 &
# python getCenterFeature.py --data hazelnut --kmeans 128 &
# python getCenterFeature.py --data leather --kmeans 128 &
# python getCenterFeature.py --data metal_nut --kmeans 128 &
# python getCenterFeature.py --data pill --kmeans 128 &
# python getCenterFeature.py --data screw --kmeans 128 &
# python getCenterFeature.py --data tile --kmeans 128 &
# python getCenterFeature.py --data toothbrush --kmeans 128 &
# python getCenterFeature.py --data transistor --kmeans 128 &
# python getCenterFeature.py --data wood --kmeans 128 &
python getCenterFeature.py --data zipper --kmeans 128 &

wait

RETRY=`python checkPreprocessData.py
