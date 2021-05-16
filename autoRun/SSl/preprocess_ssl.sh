# CUDA_VISIBLE_DEVICES=2 python preprocess_SSL/pretrain.py 
# CUDA_VISIBLE_DEVICES=2 python preprocess_SSL/BoW_PCA.py

wait
CUDA_VISIBLE_DEVICES=2 python preprocess_SSL/assign_idx.py --dim 128 --type train &
CUDA_VISIBLE_DEVICES=2 python preprocess_SSL/assign_idx.py --dim 128 --type test &
CUDA_VISIBLE_DEVICES=3 python preprocess_SSL/assign_idx.py --dim 128 --type all &
wait

python dataloaders.py --kmeans 128 --model ssl
python preprocess_SSL/getCenterFeature.py --data bottle
python preprocess_SSL/checkPreprocessData.py 