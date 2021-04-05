#!/bin/bash

CUDA_VISIBLE_DEVICES=2,3 python model_weightSample.py --image_size 1024 --patch_size 64 --train_batch 8 --test_batch_size 32 --dim 128 --kmeans=128 --data=capsule --with_mask True --dim_reduction PCA --type=good --epoch 60 &
CUDA_VISIBLE_DEVICES=0,1 python model_weightSample.py --image_size 1024 --patch_size 64 --train_batch 8 --test_batch_size 32 --dim 128 --kmeans=128 --data=wood --with_mask True --dim_reduction PCA --type=good --epoch 60 &
wait
CUDA_VISIBLE_DEVICES=0,1 python model_weightSample.py --image_size 1024 --patch_size 64 --train_batch 8 --test_batch_size 32 --dim 128 --kmeans=128 --data=cable --with_mask True --dim_reduction PCA --type=good --epoch 60 &
CUDA_VISIBLE_DEVICES=0,1 python model_weightSample.py --image_size 1024 --patch_size 64 --train_batch 8 --test_batch_size 32 --dim 128 --kmeans=128 --data=carpet --with_mask True --dim_reduction PCA --type=good --epoch 60 &
wait
CUDA_VISIBLE_DEVICES=2,3 python model_weightSample.py --image_size 1024 --patch_size 64 --train_batch 8 --test_batch_size 32 --dim 128 --kmeans=128 --data=zipper --with_mask True --dim_reduction PCA --type=good --epoch 60 &
CUDA_VISIBLE_DEVICES=0,1 python model_weightSample.py --image_size 1024 --patch_size 64 --train_batch 8 --test_batch_size 32 --dim 128 --kmeans=128 --data=screw --with_mask True --dim_reduction PCA --type=good --epoch 60 &
wait
CUDA_VISIBLE_DEVICES=2,3 python model_weightSample.py --image_size 1024 --patch_size 64 --train_batch 8 --test_batch_size 32 --dim 128 --kmeans=128 --data=tile --with_mask True --dim_reduction PCA --type=good --epoch 60 &
CUDA_VISIBLE_DEVICES=0,1 python model_weightSample.py --image_size 1024 --patch_size 64 --train_batch 8 --test_batch_size 32 --dim 128 --kmeans=128 --data=toothbrush --with_mask True --dim_reduction PCA --type=good --epoch 60 &
wait
CUDA_VISIBLE_DEVICES=2,3 python model_weightSample.py --image_size 1024 --patch_size 64 --train_batch 8 --test_batch_size 32 --dim 128 --kmeans=128 --data=bottle --with_mask True --dim_reduction PCA --type=good --epoch 60 &
CUDA_VISIBLE_DEVICES=2,3 python model_weightSample.py --image_size 1024 --patch_size 64 --train_batch 8 --test_batch_size 32 --dim 128 --kmeans=128 --data=transistor --with_mask True --dim_reduction PCA --type=good --epoch 60 &
wait
CUDA_VISIBLE_DEVICES=2,3 python model_weightSample.py --image_size 1024 --patch_size 64 --train_batch 8 --test_batch_size 32 --dim 128 --kmeans=128 --data=grid --with_mask True --dim_reduction PCA --type=good --epoch 60 &
CUDA_VISIBLE_DEVICES=0,1 python model_weightSample.py --image_size 1024 --patch_size 64 --train_batch 8 --test_batch_size 32 --dim 128 --kmeans=128 --data=hazelnut --with_mask True --dim_reduction PCA --type=good --epoch 60 &
wait
CUDA_VISIBLE_DEVICES=2,3 python model_weightSample.py --image_size 1024 --patch_size 64 --train_batch 8 --test_batch_size 32 --dim 128 --kmeans=128 --data=leather --with_mask True --dim_reduction PCA --type=good --epoch 60 &
CUDA_VISIBLE_DEVICES=0,1 python model_weightSample.py --image_size 1024 --patch_size 64 --train_batch 8 --test_batch_size 32 --dim 128 --kmeans=128 --data=metal_nut --with_mask True --dim_reduction PCA --type=good --epoch 60 &
wait
# CUDA_VISIBLE_DEVICES=2,3 python model_weightSample.py --image_size 1024 --patch_size 64 --train_batch 8 --test_batch_size 32 --dim 128 --kmeans=128 --data=pill --with_mask True --dim_reduction PCA --type=good --epoch 60 &
# wait