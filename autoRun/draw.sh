cd $PWD

# CUDA_VIDIBLE_DEVICES=0,1 python eval_aucroc.py --patch_size 128 --kmeans 128 --data bottle --index 19 &
# CUDA_VIDIBLE_DEVICES=2,3 python eval_aucroc.py --patch_size 128 --kmeans 128 --data capsule --index 18 &
# wait

CUDA_VIDIBLE_DEVICES=0,1 python multi_map.py --patch_size 128 --kmeans 128 --data bottle --index 19 &
CUDA_VIDIBLE_DEVICES=2,3 python multi_map.py --patch_size 128 --kmeans 128 --data capsule --index 18 &
wait

python draw_multiMap.py --kmeans 128 --data bottle --index 19 &
python draw_multiMap.py --kmeans 128 --data capsule --index 18 &
wait
