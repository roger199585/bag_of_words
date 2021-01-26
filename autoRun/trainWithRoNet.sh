# CUDA_VISIBLE_DEVICES=0,1 python model_RoNet_weightSample.py --data bottle --kmeans 128 --type good --train_batch 16 --with_mask True --patch_size 64 --epoch 40 &
# CUDA_VISIBLE_DEVICES=2,3 python model_RoNet_weightSample.py --data cable --kmeans 128 --type good --train_batch 16 --with_mask True --patch_size 64 --epoch 40 &
# wait
# CUDA_VISIBLE_DEVICES=0,1 python model_RoNet_weightSample.py --data capsule --kmeans 128 --type good --train_batch 16 --with_mask True --patch_size 64 --epoch 40 &
# CUDA_VISIBLE_DEVICES=2,3 python model_RoNet_weightSample.py --data carpet --kmeans 128 --type good --train_batch 16 --with_mask True --patch_size 64 --epoch 40 &
# wait
# CUDA_VISIBLE_DEVICES=0,1 python model_RoNet_weightSample.py --data grid --kmeans 128 --type good --train_batch 16 --with_mask True --patch_size 64 --epoch 40 &
# CUDA_VISIBLE_DEVICES=2,3 python model_RoNet_weightSample.py --data hazelnut --kmeans 128 --type good --train_batch 16 --with_mask True --patch_size 64 --epoch 40 &
# wait
# CUDA_VISIBLE_DEVICES=0,1 python model_RoNet_weightSample.py --data leather --kmeans 128 --type good --train_batch 16 --with_mask True --patch_size 64 --epoch 40 &
# CUDA_VISIBLE_DEVICES=2,3 python model_RoNet_weightSample.py --data metal_nut --kmeans 128 --type good --train_batch 16 --with_mask True --patch_size 64 --epoch 40 &
# wait
CUDA_VISIBLE_DEVICES=0,1 python model_RoNet_weightSample.py --data pill --kmeans 128 --type good --train_batch 16 --with_mask True --patch_size 64 --epoch 40 &
CUDA_VISIBLE_DEVICES=2,3 python model_RoNet_weightSample.py --data screw --kmeans 128 --type good --train_batch 16 --with_mask True --patch_size 64 --epoch 40 &
wait
CUDA_VISIBLE_DEVICES=0,1 python model_RoNet_weightSample.py --data tile --kmeans 128 --type good --train_batch 16 --with_mask True --patch_size 64 --epoch 40 &
CUDA_VISIBLE_DEVICES=2,3 python model_RoNet_weightSample.py --data toothbrush --kmeans 128 --type good --train_batch 16 --with_mask True --patch_size 64 --epoch 40 &
wait
CUDA_VISIBLE_DEVICES=0,1 python model_RoNet_weightSample.py --data transistor --kmeans 128 --type good --train_batch 16 --with_mask True --patch_size 64 --epoch 40 &
CUDA_VISIBLE_DEVICES=2,3 python model_RoNet_weightSample.py --data wood --kmeans 128 --type good --train_batch 16 --with_mask True --patch_size 64 --epoch 40 &
wait
CUDA_VISIBLE_DEVICES=0,1 python model_RoNet_weightSample.py --data zipper --kmeans 128 --type good --train_batch 16 --with_mask True --patch_size 64 --epoch 40 &
wait
