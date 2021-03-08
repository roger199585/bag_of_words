#!/bin/bash

# CUDA_VISIBLE_DEVICES=2,3 python model_weightSample.py --patch_size 64 --train_batch 32 --kmeans=128 --data=tile --with_mask True --dim_reduction UMAP --type=good --epoch 40 &
# CUDA_VISIBLE_DEVICES=0,1 python model_weightSample.py --patch_size 64 --train_batch 32 --kmeans=128 --data=capsule --with_mask True --dim_reduction UMAP --type=good --epoch 40 &
# wait
# CUDA_VISIBLE_DEVICES=2,3 python model_weightSample.py --patch_size 64 --train_batch 32 --kmeans=128 --data=grid --with_mask True --dim_reduction UMAP --type=good --epoch 40 &
# CUDA_VISIBLE_DEVICES=0,1 python model_weightSample.py --patch_size 64 --train_batch 32 --kmeans=128 --data=bottle --with_mask True --dim_reduction UMAP --type=good --epoch 40 &
# wait
# CUDA_VISIBLE_DEVICES=0,1 python model_weightSample.py --patch_size 64 --train_batch 32 --kmeans=128 --data=hazelnut --with_mask True --dim_reduction UMAP --type=good --epoch 40 &
# CUDA_VISIBLE_DEVICES=2,3 python model_weightSample.py --patch_size 64 --train_batch 32 --kmeans=128 --data=leather --with_mask True --dim_reduction UMAP --type=good --epoch 40 &
# wait
# CUDA_VISIBLE_DEVICES=0,1 python model_weightSample.py --patch_size 64 --train_batch 32 --kmeans=128 --data=metal_nut --with_mask True --dim_reduction UMAP --type=good --epoch 40 &
# CUDA_VISIBLE_DEVICES=2,3 python model_weightSample.py --patch_size 64 --train_batch 32 --kmeans=128 --data=pill --with_mask True --dim_reduction UMAP --type=good --epoch 40 &
# wait
# CUDA_VISIBLE_DEVICES=0,1 python model_weightSample.py --patch_size 64 --train_batch 32 --kmeans=128 --data=screw --with_mask True --dim_reduction UMAP --type=good --epoch 40 &
# CUDA_VISIBLE_DEVICES=2,3 python model_weightSample.py --patch_size 64 --train_batch 32 --kmeans=128 --data=cable --with_mask True --dim_reduction UMAP --type=good --epoch 40 &
# wait
# CUDA_VISIBLE_DEVICES=2,3 python model_weightSample.py --patch_size 64 --train_batch 16 --kmeans=128 --data=toothbrush --with_mask True --dim_reduction PCA --type=good --epoch 40 &
# CUDA_VISIBLE_DEVICES=2,3 python model_weightSample.py --patch_size 64 --train_batch 32 --kmeans=128 --data=wood --with_mask True --dim_reduction UMAP --type=good --epoch 40 &
# wait
# CUDA_VISIBLE_DEVICES=0,1 python model_weightSample.py --patch_size 64 --train_batch 16 --kmeans=128 --data=zipper --with_mask True --dim_reduction PCA --type=good --epoch 40 &

# wait
# CUDA_VISIBLE_DEVICES=0,1 python model_weightSample.py --patch_size 64 --train_batch 32 --kmeans=128 --data=metal_nut --with_mask True --dim_reduction UMAP --type=good --epoch 40 &
# CUDA_VISIBLE_DEVICES=0,1 python model_weightSample.py --patch_size 64 --train_batch 32 --kmeans=128 --data=screw --with_mask True --dim_reduction UMAP --type=good --epoch 40 &
# # CUDA_VISIBLE_DEVICES=2,3 python model_weightSample.py --patch_size 64 --train_batch 32 --kmeans=128 --data=pill --with_mask True --dim_reduction UMAP --type=good --epoch 40 &

# wait

# CUDA_VISIBLE_DEVICES=2,3 python model_weightSample.py --patch_size 64 --train_batch 32 --kmeans=128 --data=cable --with_mask True --dim_reduction UMAP --type=good --epoch 40 &

# wait
# Try different initialuzation
# Train on 136
# CUDA_VISIBLE_DEVICES=0,1 python model_weightSample.py --patch_size 64 --train_batch 16 --test_batch_size 32 --kmeans=128 --data=tile --with_mask True --dim_reduction PCA --type=good --epoch 40

CUDA_VISIBLE_DEVICES=2,3 python model_feature_first_weightSample.py --image_size 224 --patch_size 32 --train_batch 16 --test_batch_size 7 --dim 100 --kmeans=128 --data=bottle --with_mask True --dim_reduction PCA --type=good --epoch 80 &
CUDA_VISIBLE_DEVICES=0,1 python model_feature_first_weightSample.py --image_size 224 --patch_size 32 --train_batch 16 --test_batch_size 7 --dim 100 --kmeans=128 --data=cable --with_mask True --dim_reduction PCA --type=good --epoch 80 &
CUDA_VISIBLE_DEVICES=2,3 python model_feature_first_weightSample.py --image_size 224 --patch_size 32 --train_batch 16 --test_batch_size 7 --dim 100 --kmeans=128 --data=capsule --with_mask True --dim_reduction PCA --type=good --epoch 80 &
CUDA_VISIBLE_DEVICES=0,1 python model_feature_first_weightSample.py --image_size 224 --patch_size 32 --train_batch 16 --test_batch_size 7 --dim 100 --kmeans=128 --data=carpet --with_mask True --dim_reduction PCA --type=good --epoch 80 &
wait
CUDA_VISIBLE_DEVICES=2,3 python model_feature_first_weightSample.py --image_size 224 --patch_size 32 --train_batch 16 --test_batch_size 7 --dim 100 --kmeans=128 --data=grid --with_mask True --dim_reduction PCA --type=good --epoch 80 &
CUDA_VISIBLE_DEVICES=0,1 python model_feature_first_weightSample.py --image_size 224 --patch_size 32 --train_batch 16 --test_batch_size 7 --dim 100 --kmeans=128 --data=hazelnut --with_mask True --dim_reduction PCA --type=good --epoch 80 &
CUDA_VISIBLE_DEVICES=2,3 python model_feature_first_weightSample.py --image_size 224 --patch_size 32 --train_batch 16 --test_batch_size 7 --dim 100 --kmeans=128 --data=leather --with_mask True --dim_reduction PCA --type=good --epoch 80 &
CUDA_VISIBLE_DEVICES=0,1 python model_feature_first_weightSample.py --image_size 224 --patch_size 32 --train_batch 16 --test_batch_size 7 --dim 100 --kmeans=128 --data=metal_nut --with_mask True --dim_reduction PCA --type=good --epoch 80 &
wait
# CUDA_VISIBLE_DEVICES=2,3 python model_feature_first_weightSample.py --image_size 224 --patch_size 32 --train_batch 16 --test_batch_size 7 --dim 100 --kmeans=128 --data=pill --with_mask True --dim_reduction PCA --type=good --epoch 80 &
# CUDA_VISIBLE_DEVICES=0,1 python model_feature_first_weightSample.py --image_size 224 --patch_size 32 --train_batch 16 --test_batch_size 7 --dim 100 --kmeans=128 --data=screw --with_mask True --dim_reduction PCA --type=good --epoch 80 &
# CUDA_VISIBLE_DEVICES=2,3 python model_feature_first_weightSample.py --image_size 224 --patch_size 32 --train_batch 16 --test_batch_size 7 --dim 100 --kmeans=128 --data=tile --with_mask True --dim_reduction PCA --type=good --epoch 80 &
# CUDA_VISIBLE_DEVICES=0,1 python model_feature_first_weightSample.py --image_size 224 --patch_size 32 --train_batch 16 --test_batch_size 7 --dim 100 --kmeans=128 --data=toothbrush --with_mask True --dim_reduction PCA --type=good --epoch 80 &
# wait
# CUDA_VISIBLE_DEVICES=2,3 python model_feature_first_weightSample.py --image_size 224 --patch_size 32 --train_batch 16 --test_batch_size 7 --dim 100 --kmeans=128 --data=transistor --with_mask True --dim_reduction PCA --type=good --epoch 80 &
# CUDA_VISIBLE_DEVICES=0,1 python model_feature_first_weightSample.py --image_size 224 --patch_size 32 --train_batch 16 --test_batch_size 7 --dim 100 --kmeans=128 --data=wood --with_mask True --dim_reduction PCA --type=good --epoch 80 &
# CUDA_VISIBLE_DEVICES=2,3 python model_feature_first_weightSample.py --image_size 224 --patch_size 32 --train_batch 16 --test_batch_size 7 --dim 100 --kmeans=128 --data=zipper --with_mask True --dim_reduction PCA --type=good --epoch 80 &
# wait