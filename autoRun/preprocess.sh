#!/bin/bash

cd $PWD

# 進行圖片大小的轉換
# echo "Preparing images, covert all image in dataset into 1024x1024"
python preprocess/preprocess.py --root $PWD -s 1024 --types bottle  &
# python preprocess/preprocess.py --root $PWD -s 1024 --types cable &
# python preprocess/preprocess.py --root $PWD -s 1024 -q True --types capsule &
# python preprocess/preprocess.py --root $PWD -s 1024 --types carpet &
# python preprocess/preprocess.py --root $PWD -s 1024 --types grid &
# python preprocess/preprocess.py --root $PWD -s 1024 --types hazelnut &
# python preprocess/preprocess.py --root $PWD -s 1024 --types leather &
# python preprocess/preprocess.py --root $PWD -s 1024 --types metal_nut &
# python preprocess/preprocess.py --root $PWD -s 1024 --types pill &
# python preprocess/preprocess.py --root $PWD -s 1024 --types screw &
# python preprocess/preprocess.py --root $PWD -s 1024 -q True --types tile &
# python preprocess/preprocess.py --root $PWD -s 1024 --types toothbrush &
# python preprocess/preprocess.py --root $PWD -s 1024 --types transistor &
# python preprocess/preprocess.py --root $PWD -s 1024 -q True --types wood &
# python preprocess/preprocess.py --root $PWD -s 1024 --types zipper &
wait

# 建立 chunks and coordinates (切 chunk 以及讓他有位移)
CUDA_VISIBLE_DEVICES=2 python preprocess/pretrain_vgg.py --dim_reduction PCA --data bottle --patch_size 64 --fine_tune_epoch 15 &
# CUDA_VISIBLE_DEVICES=1 python preprocess/pretrain_vgg.py --dim_reduction PCA --data cable --patch_size 64 &
# CUDA_VISIBLE_DEVICES=2 python preprocess/pretrain_vgg.py --dim_reduction PCA --data capsule --patch_size 64 &
# CUDA_VISIBLE_DEVICES=3 python preprocess/pretrain_vgg.py --dim_reduction PCA --data carpet --patch_size 64 &
# wait
# CUDA_VISIBLE_DEVICES=0 python preprocess/pretrain_vgg.py --dim_reduction PCA --data grid --patch_size 64 &
# CUDA_VISIBLE_DEVICES=1 python preprocess/pretrain_vgg.py --dim_reduction PCA --data hazelnut --patch_size 64 &
# CUDA_VISIBLE_DEVICES=2 python preprocess/pretrain_vgg.py --dim_reduction PCA --data leather --patch_size 64 &
# CUDA_VISIBLE_DEVICES=3 python preprocess/pretrain_vgg.py --dim_reduction PCA --data metal_nut --patch_size 64 &
# wait
# CUDA_VISIBLE_DEVICES=1 python preprocess/pretrain_vgg.py --dim_reduction PCA --data pill --patch_size 64 &
# CUDA_VISIBLE_DEVICES=1 python preprocess/pretrain_vgg.py --dim_reduction PCA --data screw --patch_size 64 &
# CUDA_VISIBLE_DEVICES=0 python preprocess/pretrain_vgg.py --dim_reduction PCA --data tile --patch_size 64 &
# CUDA_VISIBLE_DEVICES=3 python preprocess/pretrain_vgg.py --dim_reduction PCA --data toothbrush --patch_size 64 &
# wait
# CUDA_VISIBLE_DEVICES=0 python preprocess/pretrain_vgg.py --dim_reduction PCA --data transistor --patch_size 64 &
# CUDA_VISIBLE_DEVICES=1 python preprocess/pretrain_vgg.py --dim_reduction PCA --data wood --patch_size 64 &
# CUDA_VISIBLE_DEVICES=2 python preprocess/pretrain_vgg.py --dim_reduction PCA --data zipper --patch_size 64 &
wait

# 透過上一步切好的資料給 kmeans 分群
python preprocess/BoW_PCA.py --data bottle --kmeans 128 --dim 128 &
# python preprocess/BoW_PCA.py --data cable --kmeans 128 --dim 128 &
# wait
# python preprocess/BoW_PCA.py --data capsule --kmeans 128 --dim 128 &
# python preprocess/BoW_PCA.py --data carpet --kmeans 128 --dim 128 &
# wait
# python preprocess/BoW_PCA.py --data grid --kmeans 128 --dim 128 &
# python preprocess/BoW_PCA.py --data hazelnut --kmeans 128 --dim 128 &
# wait
# python preprocess/BoW_PCA.py --data leather --kmeans 128 --dim 128 &
# python preprocess/BoW_PCA.py --data metal_nut --kmeans 128 --dim 128 &
# wait
# python preprocess/BoW_PCA.py --data pill --kmeans 128 --dim 128 &
# python preprocess/BoW_PCA.py --data screw --kmeans 128 --dim 128 &
# wait
# python preprocess/BoW_PCA.py --data tile --kmeans 128 --dim 128 &
# python preprocess/BoW_PCA.py --data toothbrush --kmeans 128 --dim 128 &
wait
# python preprocess/BoW_PCA.py --data transistor --kmeans 128 --dim 128 &
# python preprocess/BoW_PCA.py --data wood --kmeans 128 --dim 128 &
# wait
# python preprocess/BoW_PCA.py --data zipper --kmeans 128 --dim 128 &
# wait

# 給定每個 patch 的 label
CUDA_VISIBLE_DEVICES=2 python preprocess/assign_idx.py --dim_reduction PCA --patch_size 64 --data bottle --dim 128 --kmeans 128 --type train --fine_tune_epoch 15 &
CUDA_VISIBLE_DEVICES=2 python preprocess/assign_idx.py --dim_reduction PCA --patch_size 64 --data bottle --dim 128 --kmeans 128 --type test --fine_tune_epoch 15 &
CUDA_VISIBLE_DEVICES=3 python preprocess/assign_idx.py --dim_reduction PCA --patch_size 64 --data bottle --dim 128 --kmeans 128 --type all --fine_tune_epoch 15 &
# CUDA_VISIBLE_DEVICES=1 python preprocess/assign_idx.py --dim_reduction PCA --patch_size 64 --data cable --dim 128 --kmeans 128 --type train &
# CUDA_VISIBLE_DEVICES=1 python preprocess/assign_idx.py --dim_reduction PCA --patch_size 64 --data cable --dim 128 --kmeans 128 --type test &
# CUDA_VISIBLE_DEVICES=1 python preprocess/assign_idx.py --dim_reduction PCA --patch_size 64 --data cable --dim 128 --kmeans 128 --type all &
# CUDA_VISIBLE_DEVICES=2 python preprocess/assign_idx.py --dim_reduction PCA --patch_size 64 --data capsule --dim 128 --kmeans 128 --type train &
# CUDA_VISIBLE_DEVICES=2 python preprocess/assign_idx.py --dim_reduction PCA --patch_size 64 --data capsule --dim 128 --kmeans 128 --type test &
# CUDA_VISIBLE_DEVICES=2 python preprocess/assign_idx.py --dim_reduction PCA --patch_size 64 --data capsule --dim 128 --kmeans 128 --type all &
# CUDA_VISIBLE_DEVICES=3 python preprocess/assign_idx.py --dim_reduction PCA --patch_size 64 --data carpet --dim 128 --kmeans 128 --type train &
# CUDA_VISIBLE_DEVICES=3 python preprocess/assign_idx.py --dim_reduction PCA --patch_size 64 --data carpet --dim 128 --kmeans 128 --type test &
# CUDA_VISIBLE_DEVICES=3 python preprocess/assign_idx.py --dim_reduction PCA --patch_size 64 --data carpet --dim 128 --kmeans 128 --type all &
# wait
# CUDA_VISIBLE_DEVICES=0 python preprocess/assign_idx.py --dim_reduction PCA --patch_size 64 --data grid --dim 128 --kmeans 128 --type train &
# CUDA_VISIBLE_DEVICES=0 python preprocess/assign_idx.py --dim_reduction PCA --patch_size 64 --data grid --dim 128 --kmeans 128 --type test &
# CUDA_VISIBLE_DEVICES=0 python preprocess/assign_idx.py --dim_reduction PCA --patch_size 64 --data grid --dim 128 --kmeans 128 --type all &
# CUDA_VISIBLE_DEVICES=1 python preprocess/assign_idx.py --dim_reduction PCA --patch_size 64 --data hazelnut --dim 128 --kmeans 128 --type train &
# CUDA_VISIBLE_DEVICES=1 python preprocess/assign_idx.py --dim_reduction PCA --patch_size 64 --data hazelnut --dim 128 --kmeans 128 --type test &
# CUDA_VISIBLE_DEVICES=1 python preprocess/assign_idx.py --dim_reduction PCA --patch_size 64 --data hazelnut --dim 128 --kmeans 128 --type all &
# CUDA_VISIBLE_DEVICES=2 python preprocess/assign_idx.py --dim_reduction PCA --patch_size 64 --data leather --dim 128 --kmeans 128 --type train &
# CUDA_VISIBLE_DEVICES=2 python preprocess/assign_idx.py --dim_reduction PCA --patch_size 64 --data leather --dim 128 --kmeans 128 --type test &
# CUDA_VISIBLE_DEVICES=2 python preprocess/assign_idx.py --dim_reduction PCA --patch_size 64 --data leather --dim 128 --kmeans 128 --type all &
# CUDA_VISIBLE_DEVICES=3 python preprocess/assign_idx.py --dim_reduction PCA --patch_size 64 --data metal_nut --dim 128 --kmeans 128 --type train &
# CUDA_VISIBLE_DEVICES=3 python preprocess/assign_idx.py --dim_reduction PCA --patch_size 64 --data metal_nut --dim 128 --kmeans 128 --type test &
# CUDA_VISIBLE_DEVICES=3 python preprocess/assign_idx.py --dim_reduction PCA --patch_size 64 --data metal_nut --dim 128 --kmeans 128 --type all &
# wait
# CUDA_VISIBLE_DEVICES=0 python preprocess/assign_idx.py --dim_reduction PCA --patch_size 64 --data pill --dim 128 --kmeans 128 --type train &
# CUDA_VISIBLE_DEVICES=0 python preprocess/assign_idx.py --dim_reduction PCA --patch_size 64 --data pill --dim 128 --kmeans 128 --type test &
# CUDA_VISIBLE_DEVICES=0 python preprocess/assign_idx.py --dim_reduction PCA --patch_size 64 --data pill --dim 128 --kmeans 128 --type all &
# CUDA_VISIBLE_DEVICES=1 python preprocess/assign_idx.py --dim_reduction PCA --patch_size 64 --data screw --dim 128 --kmeans 128 --type train &
# CUDA_VISIBLE_DEVICES=1 python preprocess/assign_idx.py --dim_reduction PCA --patch_size 64 --data screw --dim 128 --kmeans 128 --type test &
# CUDA_VISIBLE_DEVICES=1 python preprocess/assign_idx.py --dim_reduction PCA --patch_size 64 --data screw --dim 128 --kmeans 128 --type all &
# CUDA_VISIBLE_DEVICES=0 python preprocess/assign_idx.py --dim_reduction PCA --patch_size 64 --data tile --dim 128 --kmeans 128 --type train &
# CUDA_VISIBLE_DEVICES=0 python preprocess/assign_idx.py --dim_reduction PCA --patch_size 64 --data tile --dim 128 --kmeans 128 --type test &
# CUDA_VISIBLE_DEVICES=0 python preprocess/assign_idx.py --dim_reduction PCA --patch_size 64 --data tile --dim 128 --kmeans 128 --type all &
# CUDA_VISIBLE_DEVICES=3 python preprocess/assign_idx.py --dim_reduction PCA --patch_size 64 --data toothbrush --dim 128 --kmeans 128 --type train &
# CUDA_VISIBLE_DEVICES=3 python preprocess/assign_idx.py --dim_reduction PCA --patch_size 64 --data toothbrush --dim 128 --kmeans 128 --type test &
# CUDA_VISIBLE_DEVICES=3 python preprocess/assign_idx.py --dim_reduction PCA --patch_size 64 --data toothbrush --dim 128 --kmeans 128 --type all &
wait
# CUDA_VISIBLE_DEVICES=0 python preprocess/assign_idx.py --dim_reduction PCA --patch_size 64 --data transistor --dim 128 --kmeans 128 --type train &
# CUDA_VISIBLE_DEVICES=0 python preprocess/assign_idx.py --dim_reduction PCA --patch_size 64 --data transistor --dim 128 --kmeans 128 --type test &
# CUDA_VISIBLE_DEVICES=0 python preprocess/assign_idx.py --dim_reduction PCA --patch_size 64 --data transistor --dim 128 --kmeans 128 --type all &
# CUDA_VISIBLE_DEVICES=1 python preprocess/assign_idx.py --dim_reduction PCA --patch_size 64 --data wood --dim 128 --kmeans 128 --type train &
# CUDA_VISIBLE_DEVICES=1 python preprocess/assign_idx.py --dim_reduction PCA --patch_size 64 --data wood --dim 128 --kmeans 128 --type test &
# CUDA_VISIBLE_DEVICES=1 python preprocess/assign_idx.py --dim_reduction PCA --patch_size 64 --data wood --dim 128 --kmeans 128 --type all &
# CUDA_VISIBLE_DEVICES=2 python preprocess/assign_idx.py --dim_reduction PCA --patch_size 64 --data zipper --dim 128 --kmeans 128 --type train &
# CUDA_VISIBLE_DEVICES=2 python preprocess/assign_idx.py --dim_reduction PCA --patch_size 64 --data zipper --dim 128 --kmeans 128 --type test &
# CUDA_VISIBLE_DEVICES=2 python preprocess/assign_idx.py --dim_reduction PCA --patch_size 64 --data zipper --dim 128 --kmeans 128 --type all &
# wait

# 我不知道這邊在幹嘛 我就爛
python dataloaders.py --dim_reduction PCA --patch_size 64 --data bottle --kmeans 128 &
# python dataloaders.py --dim_reduction PCA --patch_size 64 --data cable --kmeans 128 &
# python dataloaders.py --dim_reduction PCA --patch_size 64 --data capsule --kmeans 128 &
# python dataloaders.py --dim_reduction PCA --patch_size 64 --data carpet --kmeans 128 &
# python dataloaders.py --dim_reduction PCA --patch_size 64 --data grid --kmeans 128 &
# python dataloaders.py --dim_reduction PCA --patch_size 64 --data hazelnut --kmeans 128 &
# python dataloaders.py --dim_reduction PCA --patch_size 64 --data leather --kmeans 128 &
# python dataloaders.py --dim_reduction PCA --patch_size 64 --data metal_nut --kmeans 128 &
# python dataloaders.py --dim_reduction PCA --patch_size 64 --data pill --kmeans 128 &
# python dataloaders.py --dim_reduction PCA --patch_size 64 --data screw --kmeans 128 &
# python dataloaders.py --dim_reduction PCA --patch_size 64 --data tile --kmeans 128 &
# python dataloaders.py --dim_reduction PCA --patch_size 64 --data toothbrush --kmeans 128 &
# python dataloaders.py --dim_reduction PCA --patch_size 64 --data transistor --kmeans 128 &
# python dataloaders.py --dim_reduction PCA --patch_size 64 --data wood --kmeans 128 &
# python dataloaders.py --dim_reduction PCA --patch_size 64 --data zipper --kmeans 128 &
wait

# 找出 kmeans cluster center 的 feature 
python preprocess/getCenterFeature.py --dim_reduction PCA --patch_size 64 --data bottle --kmeans 128 &
# python preprocess/getCenterFeature.py --dim_reduction PCA --patch_size 64 --data cable --kmeans 128 &
# python preprocess/getCenterFeature.py --dim_reduction PCA --patch_size 64 --data capsule --kmeans 128 &
# python preprocess/getCenterFeature.py --dim_reduction PCA --patch_size 64 --data carpet --kmeans 128 &
# python preprocess/getCenterFeature.py --dim_reduction PCA --patch_size 64 --data grid --kmeans 128 &
# python preprocess/getCenterFeature.py --dim_reduction PCA --patch_size 64 --data hazelnut --kmeans 128 &
# python preprocess/getCenterFeature.py --dim_reduction PCA --patch_size 64 --data leather --kmeans 128 &
# python preprocess/getCenterFeature.py --dim_reduction PCA --patch_size 64 --data metal_nut --kmeans 128 &
# python preprocess/getCenterFeature.py --dim_reduction PCA --patch_size 64 --data pill --kmeans 128 &
# python preprocess/getCenterFeature.py --dim_reduction PCA --patch_size 64 --data screw --kmeans 128 &
# python preprocess/getCenterFeature.py --dim_reduction PCA --patch_size 64 --data tile --kmeans 128 &
# python preprocess/getCenterFeature.py --dim_reduction PCA --patch_size 64 --data toothbrush --kmeans 128 &
# python preprocess/getCenterFeature.py --dim_reduction PCA --patch_size 64 --data transistor --kmeans 128 &
# python preprocess/getCenterFeature.py --dim_reduction PCA --patch_size 64 --data wood --kmeans 128 &
# python preprocess/getCenterFeature.py --dim_reduction PCA --patch_size 64 --data zipper --kmeans 128 &

wait

python preprocess/checkPreprocessData.py