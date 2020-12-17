#!/bin/bash

cd $PWD

# 進行圖片大小的轉換
# echo "Preparing images, covert all image in dataset into 1024x1024"
# python preprocess.py --types transistor

# 建立 chunks and coordinates (切 chunk 以及讓他有位移)
CUDA_VISIBLE_DEVICES=0 python pretrain_vgg.py --data bottle --patch_size 128 &
CUDA_VISIBLE_DEVICES=1 python pretrain_vgg.py --data cable --patch_size 128 &
CUDA_VISIBLE_DEVICES=2 python pretrain_vgg.py --data capsule --patch_size 128 &
CUDA_VISIBLE_DEVICES=3 python pretrain_vgg.py --data carpet --patch_size 128 &
wait
CUDA_VISIBLE_DEVICES=0 python pretrain_vgg.py --data grid --patch_size 128 &
CUDA_VISIBLE_DEVICES=1 python pretrain_vgg.py --data hazelnut --patch_size 128 &
CUDA_VISIBLE_DEVICES=2 python pretrain_vgg.py --data leather --patch_size 128 &
CUDA_VISIBLE_DEVICES=3 python pretrain_vgg.py --data metal_nut --patch_size 128 &
wait
CUDA_VISIBLE_DEVICES=0 python pretrain_vgg.py --data pill --patch_size 128 &
CUDA_VISIBLE_DEVICES=1 python pretrain_vgg.py --data screw --patch_size 128 &
CUDA_VISIBLE_DEVICES=2 python pretrain_vgg.py --data tile --patch_size 128 &
CUDA_VISIBLE_DEVICES=3 python pretrain_vgg.py --data toothbrush --patch_size 128 &
wait
CUDA_VISIBLE_DEVICES=0 python pretrain_vgg.py --data transistor --patch_size 128 &
CUDA_VISIBLE_DEVICES=1 python pretrain_vgg.py --data wood --patch_size 128 &
CUDA_VISIBLE_DEVICES=2 python pretrain_vgg.py --data zipper --patch_size 128 &
wait

# 透過上一步切好的資料給 kmeans 分群
python BoW_PCA.py --data bottle --kmeans 128 --dim 128 &
python BoW_PCA.py --data cable --kmeans 128 --dim 128 &
python BoW_PCA.py --data capsule --kmeans 128 --dim 128 &
python BoW_PCA.py --data carpet --kmeans 128 --dim 128 &
python BoW_PCA.py --data grid --kmeans 128 --dim 128 &
wait
python BoW_PCA.py --data hazelnut --kmeans 128 --dim 128 &
python BoW_PCA.py --data leather --kmeans 128 --dim 128 &
python BoW_PCA.py --data metal_nut --kmeans 128 --dim 128 &
python BoW_PCA.py --data pill --kmeans 128 --dim 128 &
python BoW_PCA.py --data screw --kmeans 128 --dim 128 &
wait
python BoW_PCA.py --data tile --kmeans 128 --dim 128 &
python BoW_PCA.py --data toothbrush --kmeans 128 --dim 128 &
python BoW_PCA.py --data transistor --kmeans 128 --dim 128 &
python BoW_PCA.py --data wood --kmeans 128 --dim 128 &
python BoW_PCA.py --data zipper --kmeans 128 --dim 128 &
wait

# 給定每個 patch 的 label
CUDA_VISIBLE_DEVICES=0 python assign_idx.py --patch_size 128 --data bottle --dim 128 --kmeans 128 --type train &
CUDA_VISIBLE_DEVICES=0 python assign_idx.py --patch_size 128 --data bottle --dim 128 --kmeans 128 --type test &
CUDA_VISIBLE_DEVICES=0 python assign_idx.py --patch_size 128 --data bottle --dim 128 --kmeans 128 --type all &
CUDA_VISIBLE_DEVICES=1 python assign_idx.py --patch_size 128 --data cable --dim 128 --kmeans 128 --type train &
CUDA_VISIBLE_DEVICES=1 python assign_idx.py --patch_size 128 --data cable --dim 128 --kmeans 128 --type test &
CUDA_VISIBLE_DEVICES=1 python assign_idx.py --patch_size 128 --data cable --dim 128 --kmeans 128 --type all &
CUDA_VISIBLE_DEVICES=2 python assign_idx.py --patch_size 128 --data capsule --dim 128 --kmeans 128 --type train &
CUDA_VISIBLE_DEVICES=2 python assign_idx.py --patch_size 128 --data capsule --dim 128 --kmeans 128 --type test &
CUDA_VISIBLE_DEVICES=2 python assign_idx.py --patch_size 128 --data capsule --dim 128 --kmeans 128 --type all &
CUDA_VISIBLE_DEVICES=3 python assign_idx.py --patch_size 128 --data carpet --dim 128 --kmeans 128 --type train &
CUDA_VISIBLE_DEVICES=3 python assign_idx.py --patch_size 128 --data carpet --dim 128 --kmeans 128 --type test &
CUDA_VISIBLE_DEVICES=3 python assign_idx.py --patch_size 128 --data carpet --dim 128 --kmeans 128 --type all &
wait
CUDA_VISIBLE_DEVICES=0 python assign_idx.py --patch_size 128 --data grid --dim 128 --kmeans 128 --type train &
CUDA_VISIBLE_DEVICES=0 python assign_idx.py --patch_size 128 --data grid --dim 128 --kmeans 128 --type test &
CUDA_VISIBLE_DEVICES=0 python assign_idx.py --patch_size 128 --data grid --dim 128 --kmeans 128 --type all &
CUDA_VISIBLE_DEVICES=1 python assign_idx.py --patch_size 128 --data hazelnut --dim 128 --kmeans 128 --type train &
CUDA_VISIBLE_DEVICES=1 python assign_idx.py --patch_size 128 --data hazelnut --dim 128 --kmeans 128 --type test &
CUDA_VISIBLE_DEVICES=1 python assign_idx.py --patch_size 128 --data hazelnut --dim 128 --kmeans 128 --type all &
CUDA_VISIBLE_DEVICES=2 python assign_idx.py --patch_size 128 --data leather --dim 128 --kmeans 128 --type train &
CUDA_VISIBLE_DEVICES=2 python assign_idx.py --patch_size 128 --data leather --dim 128 --kmeans 128 --type test &
CUDA_VISIBLE_DEVICES=2 python assign_idx.py --patch_size 128 --data leather --dim 128 --kmeans 128 --type all &
CUDA_VISIBLE_DEVICES=3 python assign_idx.py --patch_size 128 --data metal_nut --dim 128 --kmeans 128 --type train &
CUDA_VISIBLE_DEVICES=3 python assign_idx.py --patch_size 128 --data metal_nut --dim 128 --kmeans 128 --type test &
CUDA_VISIBLE_DEVICES=3 python assign_idx.py --patch_size 128 --data metal_nut --dim 128 --kmeans 128 --type all &
wait
CUDA_VISIBLE_DEVICES=0 python assign_idx.py --patch_size 128 --data pill --dim 128 --kmeans 128 --type train &
CUDA_VISIBLE_DEVICES=0 python assign_idx.py --patch_size 128 --data pill --dim 128 --kmeans 128 --type test &
CUDA_VISIBLE_DEVICES=0 python assign_idx.py --patch_size 128 --data pill --dim 128 --kmeans 128 --type all &
CUDA_VISIBLE_DEVICES=1 python assign_idx.py --patch_size 128 --data screw --dim 128 --kmeans 128 --type train &
CUDA_VISIBLE_DEVICES=1 python assign_idx.py --patch_size 128 --data screw --dim 128 --kmeans 128 --type test &
CUDA_VISIBLE_DEVICES=1 python assign_idx.py --patch_size 128 --data screw --dim 128 --kmeans 128 --type all &
CUDA_VISIBLE_DEVICES=2 python assign_idx.py --patch_size 128 --data tile --dim 128 --kmeans 128 --type train &
CUDA_VISIBLE_DEVICES=2 python assign_idx.py --patch_size 128 --data tile --dim 128 --kmeans 128 --type test &
CUDA_VISIBLE_DEVICES=2 python assign_idx.py --patch_size 128 --data tile --dim 128 --kmeans 128 --type all &
CUDA_VISIBLE_DEVICES=3 python assign_idx.py --patch_size 128 --data toothbrush --dim 128 --kmeans 128 --type train &
CUDA_VISIBLE_DEVICES=3 python assign_idx.py --patch_size 128 --data toothbrush --dim 128 --kmeans 128 --type test &
CUDA_VISIBLE_DEVICES=3 python assign_idx.py --patch_size 128 --data toothbrush --dim 128 --kmeans 128 --type all &
wait
CUDA_VISIBLE_DEVICES=0 python assign_idx.py --patch_size 128 --data transistor --dim 128 --kmeans 128 --type train &
CUDA_VISIBLE_DEVICES=0 python assign_idx.py --patch_size 128 --data transistor --dim 128 --kmeans 128 --type test &
CUDA_VISIBLE_DEVICES=0 python assign_idx.py --patch_size 128 --data transistor --dim 128 --kmeans 128 --type all &
CUDA_VISIBLE_DEVICES=1 python assign_idx.py --patch_size 128 --data wood --dim 128 --kmeans 128 --type train &
CUDA_VISIBLE_DEVICES=1 python assign_idx.py --patch_size 128 --data wood --dim 128 --kmeans 128 --type test &
CUDA_VISIBLE_DEVICES=1 python assign_idx.py --patch_size 128 --data wood --dim 128 --kmeans 128 --type all &
CUDA_VISIBLE_DEVICES=2 python assign_idx.py --patch_size 128 --data zipper --dim 128 --kmeans 128 --type train &
CUDA_VISIBLE_DEVICES=2 python assign_idx.py --patch_size 128 --data zipper --dim 128 --kmeans 128 --type test &
CUDA_VISIBLE_DEVICES=2 python assign_idx.py --patch_size 128 --data zipper --dim 128 --kmeans 128 --type all &
wait

# 我不知道這邊在幹嘛 我就爛
python dataloaders.py --patch_size 128 --data bottle --kmeans 128 &
python dataloaders.py --patch_size 128 --data cable --kmeans 128 &
python dataloaders.py --patch_size 128 --data capsule --kmeans 128 &
python dataloaders.py --patch_size 128 --data carpet --kmeans 128 &
python dataloaders.py --patch_size 128 --data grid --kmeans 128 &
python dataloaders.py --patch_size 128 --data hazelnut --kmeans 128 &
python dataloaders.py --patch_size 128 --data leather --kmeans 128 &
python dataloaders.py --patch_size 128 --data metal_nut --kmeans 128 &
python dataloaders.py --patch_size 128 --data pill --kmeans 128 &
python dataloaders.py --patch_size 128 --data screw --kmeans 128 &
python dataloaders.py --patch_size 128 --data tile --kmeans 128 &
python dataloaders.py --patch_size 128 --data toothbrush --kmeans 128 &
python dataloaders.py --patch_size 128 --data transistor --kmeans 128 &
python dataloaders.py --patch_size 128 --data wood --kmeans 128 &
python dataloaders.py --patch_size 128 --data zipper --kmeans 128 &
wait

# 找出 kmeans cluster center 的 feature 
python getCenterFeature.py --patch_size 128 --data bottle --kmeans 128 &
python getCenterFeature.py --patch_size 128 --data cable --kmeans 128 &
python getCenterFeature.py --patch_size 128 --data capsule --kmeans 128 &
python getCenterFeature.py --patch_size 128 --data carpet --kmeans 128 &
python getCenterFeature.py --patch_size 128 --data grid --kmeans 128 &
python getCenterFeature.py --patch_size 128 --data hazelnut --kmeans 128 &
python getCenterFeature.py --patch_size 128 --data leather --kmeans 128 &
python getCenterFeature.py --patch_size 128 --data metal_nut --kmeans 128 &
python getCenterFeature.py --patch_size 128 --data pill --kmeans 128 &
python getCenterFeature.py --patch_size 128 --data screw --kmeans 128 &
python getCenterFeature.py --patch_size 128 --data tile --kmeans 128 &
python getCenterFeature.py --patch_size 128 --data toothbrush --kmeans 128 &
python getCenterFeature.py --patch_size 128 --data transistor --kmeans 128 &
python getCenterFeature.py --patch_size 128 --data wood --kmeans 128 &
python getCenterFeature.py --patch_size 128 --data zipper --kmeans 128 &

wait

RETRY=`python checkPreprocessData.py`
