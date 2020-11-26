cd $PWD

# CUDA_VISIBLE_DEVICES=0,1 python multi_map.py --kmeans 128 --data bottle --index 7 &
# CUDA_VISIBLE_DEVICES=2,3 python multi_map.py --kmeans 128 --data grid --index 27 &
# CUDA_VISIBLE_DEVICES=2,3 python multi_map.py --kmeans 128 --data metal_nut --index 9 &

CUDA_VISIBLE_DEVICES=0,1 python multi_map.py --kmeans 128 --data leather --index 3 &
CUDA_VISIBLE_DEVICES=2,3 python multi_map.py --kmeans 128 --data capsule --index 1 &
wait

# python draw_multiMap.py --kmeans 128 --data bottle --index 7 &
# python draw_multiMap.py --kmeans 128 --data grid --index 27 &
# python draw_multiMap.py --kmeans 128 --data metal_nut --index 9 &
python draw_multiMap.py --kmeans 128 --data leather --index 3 &
python draw_multiMap.py --kmeans 128 --data capsule --index 1 &

# wait