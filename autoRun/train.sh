#!/bin/bash
# count=1

# while [ `python checkPreprocessData.py` == 0 ]
#     do
#         echo Round $count
#         python BoW_PCA.py --data toothbrush --kmeans 128 --dim 128 &
#         wait
#         echo BoW_PCA Done
#         CUDA_VISIBLE_DEVICES=0 python assign_idx.py --data toothbrush --dim 128 --patch_size 128 --kmeans 128 --type train &
#         CUDA_VISIBLE_DEVICES=1 python assign_idx.py --data toothbrush --dim 128 --patch_size 128 --kmeans 128 --type test &
#         CUDA_VISIBLE_DEVICES=2 python assign_idx.py --data toothbrush --dim 128 --patch_size 128 --kmeans 128 --type all &
#         wait
#         echo assign_idx Done
#         python dataloaders.py --data toothbrush --kmeans 128 --patch_size 128 &
#         wait
#         echo dataloaders Done
#         python getCenterFeature.py --data toothbrush --kmeans 128 --patch_size 128 &
#         wait
#         echo getCenterFeature Done
#         count=$(($count + 1))
#     done


<<<<<<< HEAD
CUDA_VISIBLE_DEVICES=0,1 python model_weightSample.py --train_batch 16 --kmeans=128 --data=capsule --with_mask True --type=good --epoch 30 &
CUDA_VISIBLE_DEVICES=2,3 python model_weightSample.py --train_batch 16 --kmeans=128 --data=bottle --with_mask True --type=good --epoch 30 &
=======
CUDA_VISIBLE_DEVICES=0,1 python model_weightSample.py --patch_size 128 --train_batch 64 --kmeans=128 --data=capsule --with_mask True --type=good --epoch 30 &
CUDA_VISIBLE_DEVICES=2,3 python model_weightSample.py --patch_size 128 --train_batch 64 --kmeans=128 --data=bottle --with_mask True --type=good --epoch 30 &

wait
>>>>>>> 83d5ae1c5fc97ad6fb94c4beea9352c6d30b3def
