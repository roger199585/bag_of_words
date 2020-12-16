#!/bin/bash
# count=1

# while [ `python checkPreprocessData.py` == 0 ]
#     do
#         echo Round $count
#         python BoW_PCA.py --data tile --kmeans 64 --dim 128 &
#         wait
#         echo BoW_PCA Done
#         CUDA_VISIBLE_DEVICES=0 python assign_idx.py --data tile --dim 128 --kmeans 64 --type train &
#         CUDA_VISIBLE_DEVICES=1 python assign_idx.py --data tile --dim 128 --kmeans 64 --type test &
#         CUDA_VISIBLE_DEVICES=2 python assign_idx.py --data tile --dim 128 --kmeans 64 --type all &
#         wait
#         echo assign_idx Done
#         python dataloaders.py --data tile --kmeans 64 &
#         wait
#         echo dataloaders Done
#         python getCenterFeature.py --data tile --kmeans 64 &
#         wait
#         echo getCenterFeature Done
#         count=$(($count + 1))
#     done

cd $PWD
source ../BoW_env/bin/activate

CUDA_VISIBLE_DEVICES=0,1 python model_weightSample.py --patch_size 128 --train_batch 64 --kmeans=128 --data=capsule --with_mask True --type=good --epoch 30 &
CUDA_VISIBLE_DEVICES=2,3 python model_weightSample.py --patch_size 128 --train_batch 64 --kmeans=128 --data=bottle --with_mask True --type=good --epoch 30 &

wait