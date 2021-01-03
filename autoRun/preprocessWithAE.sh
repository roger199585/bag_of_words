# python preprocessAE/pretrain_autoencoder.py --data bottle --resolution 4 &
python preprocessAE/pretrain_autoencoder.py --data bottle --resolution 8 &

wait

# python preprocessAE/BoW.py --data bottle --kmeans 128 --resolution 4 &
python preprocessAE/BoW.py --data bottle --kmeans 128 --resolution 8 &

wait 

# python preprocessAE/assign_idx.py --data bottle --type train --kmeans 128 --resolution 4 &
# python preprocessAE/assign_idx.py --data bottle --type test --kmeans 128 --resolution 4 &
# python preprocessAE/assign_idx.py --data bottle --type all --kmeans 128 --resolution 4 &

# wait

python preprocessAE/assign_idx.py --data bottle --type train --kmeans 128 --resolution 8 &
python preprocessAE/assign_idx.py --data bottle --type test --kmeans 128 --resolution 8 &
python preprocessAE/assign_idx.py --data bottle --type all --kmeans 128 --resolution 8 &

wait

# python preprocessAE/getCenterFeature.py --data bottle --kmeans 128 --resolution 4 &
python preprocessAE/getCenterFeature.py --data bottle --kmeans 128 --resolution 8 &

wait 

# Train
# CUDA_VISIBLE_DEVICES=0.1 python model_ae_weightSample.py --data bottle --kmeans 128 --type good --with_mask True --resolution 4 --epoch 40 &

# wait