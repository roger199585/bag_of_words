#!/bin/bash

cd $PWD

# 進行圖片大小的轉換
# echo "Preparing images, covert all image in dataset into 1024x1024"
# python preprocess/preprocess.py --root /train-data2/corn/bag_of_words -s 224 --types bottle  &
# python preprocess/preprocess.py --root /train-data2/corn/bag_of_words -s 224 --types cable &
# python preprocess/preprocess.py --root /train-data2/corn/bag_of_words -s 224 --types capsule &
# python preprocess/preprocess.py --root /train-data2/corn/bag_of_words -s 224 --types carpet &
# python preprocess/preprocess.py --root /train-data2/corn/bag_of_words -s 224 --types grid &
# python preprocess/preprocess.py --root /train-data2/corn/bag_of_words -s 224 --types hazelnut &
# python preprocess/preprocess.py --root /train-data2/corn/bag_of_words -s 224 --types leather &
# python preprocess/preprocess.py --root /train-data2/corn/bag_of_words -s 224 --types metal_nut &
# python preprocess/preprocess.py --root /train-data2/corn/bag_of_words -s 224 --types pill &
# python preprocess/preprocess.py --root /train-data2/corn/bag_of_words -s 224 --types screw &
# python preprocess/preprocess.py --root /train-data2/corn/bag_of_words -s 224 --types tile &
# python preprocess/preprocess.py --root /train-data2/corn/bag_of_words -s 224 --types toothbrush &
# python preprocess/preprocess.py --root /train-data2/corn/bag_of_words -s 224 --types transistor &
# python preprocess/preprocess.py --root /train-data2/corn/bag_of_words -s 224 --types wood &
# python preprocess/preprocess.py --root /train-data2/corn/bag_of_words -s 224 --types zipper &
# wait

# 建立 chunks
CUDA_VISIBLE_DEVICES=0 python preprocess_feature_first/pretrain_vgg.py --data bottle &
CUDA_VISIBLE_DEVICES=1 python preprocess_feature_first/pretrain_vgg.py --data cable &
CUDA_VISIBLE_DEVICES=2 python preprocess_feature_first/pretrain_vgg.py --data capsule &
CUDA_VISIBLE_DEVICES=3 python preprocess_feature_first/pretrain_vgg.py --data carpet &
wait
CUDA_VISIBLE_DEVICES=0 python preprocess_feature_first/pretrain_vgg.py --data grid &
CUDA_VISIBLE_DEVICES=1 python preprocess_feature_first/pretrain_vgg.py --data hazelnut &
CUDA_VISIBLE_DEVICES=2 python preprocess_feature_first/pretrain_vgg.py --data leather &
CUDA_VISIBLE_DEVICES=3 python preprocess_feature_first/pretrain_vgg.py --data metal_nut &
wait
CUDA_VISIBLE_DEVICES=0 python preprocess_feature_first/pretrain_vgg.py --data pill &
CUDA_VISIBLE_DEVICES=1 python preprocess_feature_first/pretrain_vgg.py --data screw &
CUDA_VISIBLE_DEVICES=2 python preprocess_feature_first/pretrain_vgg.py --data tile &
CUDA_VISIBLE_DEVICES=3 python preprocess_feature_first/pretrain_vgg.py --data toothbrush &
wait
CUDA_VISIBLE_DEVICES=0 python preprocess_feature_first/pretrain_vgg.py --data transistor &
CUDA_VISIBLE_DEVICES=1 python preprocess_feature_first/pretrain_vgg.py --data wood &
CUDA_VISIBLE_DEVICES=2 python preprocess_feature_first/pretrain_vgg.py --data zipper &
wait

# 透過上一步切好的資料給 kmeans 分群
python preprocess_feature_first/BoW_PCA.py --data bottle --kmeans 128 --dim 100 --dim_reduction PCA &
python preprocess_feature_first/BoW_PCA.py --data cable --kmeans 128 --dim 100 --dim_reduction PCA &
python preprocess_feature_first/BoW_PCA.py --data capsule --kmeans 128 --dim 100 --dim_reduction PCA &
python preprocess_feature_first/BoW_PCA.py --data carpet --kmeans 128 --dim 100 --dim_reduction PCA &
python preprocess_feature_first/BoW_PCA.py --data grid --kmeans 128 --dim 100 --dim_reduction PCA &
python preprocess_feature_first/BoW_PCA.py --data hazelnut --kmeans 128 --dim 100 --dim_reduction PCA &
python preprocess_feature_first/BoW_PCA.py --data leather --kmeans 128 --dim 100 --dim_reduction PCA &
python preprocess_feature_first/BoW_PCA.py --data metal_nut --kmeans 128 --dim 100 --dim_reduction PCA &
python preprocess_feature_first/BoW_PCA.py --data pill --kmeans 128 --dim 100 --dim_reduction PCA &
python preprocess_feature_first/BoW_PCA.py --data screw --kmeans 128 --dim 100 --dim_reduction PCA &
python preprocess_feature_first/BoW_PCA.py --data tile --kmeans 128 --dim 100 --dim_reduction PCA &
python preprocess_feature_first/BoW_PCA.py --data toothbrush --kmeans 128 --dim 100 --dim_reduction PCA &
python preprocess_feature_first/BoW_PCA.py --data transistor --kmeans 128 --dim 100 --dim_reduction PCA &
python preprocess_feature_first/BoW_PCA.py --data wood --kmeans 128 --dim 100 --dim_reduction PCA &
python preprocess_feature_first/BoW_PCA.py --data zipper --kmeans 128 --dim 100 --dim_reduction PCA &
wait

# # 給定每個 patch 的 label
CUDA_VISIBLE_DEVICES=0 python preprocess_feature_first/assign_idx.py --dim_reduction PCA --data bottle --dim 100 --kmeans 128 --type train &
CUDA_VISIBLE_DEVICES=0 python preprocess_feature_first/assign_idx.py --dim_reduction PCA --data bottle --dim 100 --kmeans 128 --type test &
CUDA_VISIBLE_DEVICES=0 python preprocess_feature_first/assign_idx.py --dim_reduction PCA --data bottle --dim 100 --kmeans 128 --type all &
CUDA_VISIBLE_DEVICES=1 python preprocess_feature_first/assign_idx.py --dim_reduction PCA --data cable --dim 100 --kmeans 128 --type train &
CUDA_VISIBLE_DEVICES=1 python preprocess_feature_first/assign_idx.py --dim_reduction PCA --data cable --dim 100 --kmeans 128 --type test &
CUDA_VISIBLE_DEVICES=1 python preprocess_feature_first/assign_idx.py --dim_reduction PCA --data cable --dim 100 --kmeans 128 --type all &
CUDA_VISIBLE_DEVICES=2 python preprocess_feature_first/assign_idx.py --dim_reduction PCA --data capsule --dim 100 --kmeans 128 --type train &
CUDA_VISIBLE_DEVICES=2 python preprocess_feature_first/assign_idx.py --dim_reduction PCA --data capsule --dim 100 --kmeans 128 --type test &
CUDA_VISIBLE_DEVICES=2 python preprocess_feature_first/assign_idx.py --dim_reduction PCA --data capsule --dim 100 --kmeans 128 --type all &
CUDA_VISIBLE_DEVICES=3 python preprocess_feature_first/assign_idx.py --dim_reduction PCA --data carpet --dim 100 --kmeans 128 --type train &
CUDA_VISIBLE_DEVICES=3 python preprocess_feature_first/assign_idx.py --dim_reduction PCA --data carpet --dim 100 --kmeans 128 --type test &
CUDA_VISIBLE_DEVICES=3 python preprocess_feature_first/assign_idx.py --dim_reduction PCA --data carpet --dim 100 --kmeans 128 --type all &
wait
CUDA_VISIBLE_DEVICES=0 python preprocess_feature_first/assign_idx.py --dim_reduction PCA --data grid --dim 100 --kmeans 128 --type train &
CUDA_VISIBLE_DEVICES=0 python preprocess_feature_first/assign_idx.py --dim_reduction PCA --data grid --dim 100 --kmeans 128 --type test &
CUDA_VISIBLE_DEVICES=0 python preprocess_feature_first/assign_idx.py --dim_reduction PCA --data grid --dim 100 --kmeans 128 --type all &
CUDA_VISIBLE_DEVICES=1 python preprocess_feature_first/assign_idx.py --dim_reduction PCA --data hazelnut --dim 100 --kmeans 128 --type train &
CUDA_VISIBLE_DEVICES=1 python preprocess_feature_first/assign_idx.py --dim_reduction PCA --data hazelnut --dim 100 --kmeans 128 --type test &
CUDA_VISIBLE_DEVICES=1 python preprocess_feature_first/assign_idx.py --dim_reduction PCA --data hazelnut --dim 100 --kmeans 128 --type all &
CUDA_VISIBLE_DEVICES=2 python preprocess_feature_first/assign_idx.py --dim_reduction PCA --data leather --dim 100 --kmeans 128 --type train &
CUDA_VISIBLE_DEVICES=2 python preprocess_feature_first/assign_idx.py --dim_reduction PCA --data leather --dim 100 --kmeans 128 --type test &
CUDA_VISIBLE_DEVICES=2 python preprocess_feature_first/assign_idx.py --dim_reduction PCA --data leather --dim 100 --kmeans 128 --type all &
CUDA_VISIBLE_DEVICES=3 python preprocess_feature_first/assign_idx.py --dim_reduction PCA --data metal_nut --dim 100 --kmeans 128 --type train &
CUDA_VISIBLE_DEVICES=3 python preprocess_feature_first/assign_idx.py --dim_reduction PCA --data metal_nut --dim 100 --kmeans 128 --type test &
CUDA_VISIBLE_DEVICES=3 python preprocess_feature_first/assign_idx.py --dim_reduction PCA --data metal_nut --dim 100 --kmeans 128 --type all &
wait
CUDA_VISIBLE_DEVICES=0 python preprocess_feature_first/assign_idx.py --dim_reduction PCA --data pill --dim 100 --kmeans 128 --type train &
CUDA_VISIBLE_DEVICES=0 python preprocess_feature_first/assign_idx.py --dim_reduction PCA --data pill --dim 100 --kmeans 128 --type test &
CUDA_VISIBLE_DEVICES=0 python preprocess_feature_first/assign_idx.py --dim_reduction PCA --data pill --dim 100 --kmeans 128 --type all &
CUDA_VISIBLE_DEVICES=1 python preprocess_feature_first/assign_idx.py --dim_reduction PCA --data screw --dim 100 --kmeans 128 --type train &
CUDA_VISIBLE_DEVICES=1 python preprocess_feature_first/assign_idx.py --dim_reduction PCA --data screw --dim 100 --kmeans 128 --type test &
CUDA_VISIBLE_DEVICES=1 python preprocess_feature_first/assign_idx.py --dim_reduction PCA --data screw --dim 100 --kmeans 128 --type all &
CUDA_VISIBLE_DEVICES=2 python preprocess_feature_first/assign_idx.py --dim_reduction PCA --data tile --dim 100 --kmeans 128 --type train &
CUDA_VISIBLE_DEVICES=2 python preprocess_feature_first/assign_idx.py --dim_reduction PCA --data tile --dim 100 --kmeans 128 --type test &
CUDA_VISIBLE_DEVICES=2 python preprocess_feature_first/assign_idx.py --dim_reduction PCA --data tile --dim 100 --kmeans 128 --type all &
CUDA_VISIBLE_DEVICES=3 python preprocess_feature_first/assign_idx.py --dim_reduction PCA --data toothbrush --dim 100 --kmeans 128 --type train &
CUDA_VISIBLE_DEVICES=3 python preprocess_feature_first/assign_idx.py --dim_reduction PCA --data toothbrush --dim 100 --kmeans 128 --type test &
CUDA_VISIBLE_DEVICES=3 python preprocess_feature_first/assign_idx.py --dim_reduction PCA --data toothbrush --dim 100 --kmeans 128 --type all &
wait
CUDA_VISIBLE_DEVICES=0 python preprocess_feature_first/assign_idx.py --dim_reduction PCA --data transistor --dim 100 --kmeans 128 --type train &
CUDA_VISIBLE_DEVICES=0 python preprocess_feature_first/assign_idx.py --dim_reduction PCA --data transistor --dim 100 --kmeans 128 --type test &
CUDA_VISIBLE_DEVICES=0 python preprocess_feature_first/assign_idx.py --dim_reduction PCA --data transistor --dim 100 --kmeans 128 --type all &
CUDA_VISIBLE_DEVICES=1 python preprocess_feature_first/assign_idx.py --dim_reduction PCA --data wood --dim 100 --kmeans 128 --type train &
CUDA_VISIBLE_DEVICES=1 python preprocess_feature_first/assign_idx.py --dim_reduction PCA --data wood --dim 100 --kmeans 128 --type test &
CUDA_VISIBLE_DEVICES=1 python preprocess_feature_first/assign_idx.py --dim_reduction PCA --data wood --dim 100 --kmeans 128 --type all &
CUDA_VISIBLE_DEVICES=2 python preprocess_feature_first/assign_idx.py --dim_reduction PCA --data zipper --dim 100 --kmeans 128 --type train &
CUDA_VISIBLE_DEVICES=2 python preprocess_feature_first/assign_idx.py --dim_reduction PCA --data zipper --dim 100 --kmeans 128 --type test &
CUDA_VISIBLE_DEVICES=2 python preprocess_feature_first/assign_idx.py --dim_reduction PCA --data zipper --dim 100 --kmeans 128 --type all &
wait

# 我不知道這邊在幹嘛 我就爛
python dataloaders.py --dim_reduction PCA --image_size 224 --patch_size 32 --data bottle --kmeans 128 &
python dataloaders.py --dim_reduction PCA --image_size 224 --patch_size 32 --data cable --kmeans 128 &
python dataloaders.py --dim_reduction PCA --image_size 224 --patch_size 32 --data capsule --kmeans 128 &
python dataloaders.py --dim_reduction PCA --image_size 224 --patch_size 32 --data carpet --kmeans 128 &
python dataloaders.py --dim_reduction PCA --image_size 224 --patch_size 32 --data grid --kmeans 128 &
python dataloaders.py --dim_reduction PCA --image_size 224 --patch_size 32 --data hazelnut --kmeans 128 &
python dataloaders.py --dim_reduction PCA --image_size 224 --patch_size 32 --data leather --kmeans 128 &
python dataloaders.py --dim_reduction PCA --image_size 224 --patch_size 32 --data metal_nut --kmeans 128 &
python dataloaders.py --dim_reduction PCA --image_size 224 --patch_size 32 --data pill --kmeans 128 &
python dataloaders.py --dim_reduction PCA --image_size 224 --patch_size 32 --data screw --kmeans 128 &
python dataloaders.py --dim_reduction PCA --image_size 224 --patch_size 32 --data tile --kmeans 128 &
python dataloaders.py --dim_reduction PCA --image_size 224 --patch_size 32 --data toothbrush --kmeans 128 &
python dataloaders.py --dim_reduction PCA --image_size 224 --patch_size 32 --data transistor --kmeans 128 &
python dataloaders.py --dim_reduction PCA --image_size 224 --patch_size 32 --data wood --kmeans 128 &
python dataloaders.py --dim_reduction PCA --image_size 224 --patch_size 32 --data zipper --kmeans 128 &
wait

# 找出 kmeans cluster center 的 feature 
python preprocess_feature_first/getCenterFeature.py --dim_reduction PCA --image_size 224 --patch_size 32 --data bottle --kmeans 128 &
python preprocess_feature_first/getCenterFeature.py --dim_reduction PCA --image_size 224 --patch_size 32 --data cable --kmeans 128 &
python preprocess_feature_first/getCenterFeature.py --dim_reduction PCA --image_size 224 --patch_size 32 --data capsule --kmeans 128 &
python preprocess_feature_first/getCenterFeature.py --dim_reduction PCA --image_size 224 --patch_size 32 --data carpet --kmeans 128 &
python preprocess_feature_first/getCenterFeature.py --dim_reduction PCA --image_size 224 --patch_size 32 --data grid --kmeans 128 &
python preprocess_feature_first/getCenterFeature.py --dim_reduction PCA --image_size 224 --patch_size 32 --data hazelnut --kmeans 128 &
python preprocess_feature_first/getCenterFeature.py --dim_reduction PCA --image_size 224 --patch_size 32 --data leather --kmeans 128 &
python preprocess_feature_first/getCenterFeature.py --dim_reduction PCA --image_size 224 --patch_size 32 --data metal_nut --kmeans 128 &
python preprocess_feature_first/getCenterFeature.py --dim_reduction PCA --image_size 224 --patch_size 32 --data pill --kmeans 128 &
python preprocess_feature_first/getCenterFeature.py --dim_reduction PCA --image_size 224 --patch_size 32 --data screw --kmeans 128 &
python preprocess_feature_first/getCenterFeature.py --dim_reduction PCA --image_size 224 --patch_size 32 --data tile --kmeans 128 &
python preprocess_feature_first/getCenterFeature.py --dim_reduction PCA --image_size 224 --patch_size 32 --data toothbrush --kmeans 128 &
python preprocess_feature_first/getCenterFeature.py --dim_reduction PCA --image_size 224 --patch_size 32 --data transistor --kmeans 128 &
python preprocess_feature_first/getCenterFeature.py --dim_reduction PCA --image_size 224 --patch_size 32 --data wood --kmeans 128 &
python preprocess_feature_first/getCenterFeature.py --dim_reduction PCA --image_size 224 --patch_size 32 --data zipper --kmeans 128 &
wait

python preprocess_feature_first/checkPreprocessData.py