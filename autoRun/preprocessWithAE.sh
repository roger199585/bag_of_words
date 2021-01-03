python preprocessAE/pretrain_autoencoder.py --data tile --resolution 4 &
# python preprocessAE/pretrain_autoencoder.py --data tole --resolution 8 &

wait

python preprocessAE/BoW.py --data tile --kmeans 128 --resolution 4 &
# python preprocessAE/BoW.py --data tile --kmeans 128 --resolution 8 &

wait 

python preprocessAE/assign_idx.py --data tile --type train --kmeans 128 --resolution 4 &
python preprocessAE/assign_idx.py --data tile --type test --kmeans 128 --resolution 4 &
python preprocessAE/assign_idx.py --data tile --type all --kmeans 128 --resolution 4 &

# wait

# python preprocessAE/assign_idx.py --data tile --type train --kmeans 128 --resolution 8 &
# python preprocessAE/assign_idx.py --data tile --type test --kmeans 128 --resolution 8 &
# python preprocessAE/assign_idx.py --data tile --type all --kmeans 128 --resolution 8 &

wait

python preprocessAE/getCenterFeature.py --data tile --kmeans 128 --resolution 4 &
# python preprocessAE/getCenterFeature.py --data tile --kmeans 128 --resolution 8 &

wait 
