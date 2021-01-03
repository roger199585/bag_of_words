cd $PWD

# CUDA_VISIBLE_DEVICES=0,1 python eval_aucroc.py --patch_size 128 --kmeans 128 --data bottle --index 19 &
# CUDA_VISIBLE_DEVICES=2,3 python eval_aucroc.py --patch_size 128 --kmeans 128 --data capsule --index 18 &
# wait

# CUDA_VISIBLE_DEVICES=0,1 python multi_map.py --patch_size 128 --kmeans 128 --data bottle --index 19 &
CUDA_VISIBLE_DEVICES=2,3 python eval/multi_map.py --patch_size 128 --kmeans 128 --data bottle --index 35 &
wait

python draw_multiMap.py --kmeans 128 --data bottle --index 19 &
python draw_multiMap.py --kmeans 128 --data capsule --index 18 &
wait
