cd $PWD

CUDA_VISIBLE_DEVICES=2,3 python evalAE/eval_aucroc.py --patch_size 64 --kmeans 128 --data tile --resolution 4 --index 38 &
CUDA_VISIBLE_DEVICES=0,1 python evalAE/eval_aucroc.py --patch_size 64 --kmeans 128 --data tile --resolution 8 --index 16 &

wait

# CUDA_VISIBLE_DEVICES=2,3 python evalAE/multi_map.py --patch_size 64 --kmeans 128 --data tile --resolution 4 --index 38
# CUDA_VISIBLE_DEVICES=2,3 python evalAE/multi_map.py --patch_size 64 --kmeans 128 --data tile --resolution 8 --index 16

# wait

# python evalAE/draw_multiMap.py --patch_size 64 --kmeans 128 --data tile --resolution 4 --index 38 &
# python evalAE/draw_multiMap.py --patch_size 64 --kmeans 128 --data tile --resolution 8 --index 16 &

# wait
