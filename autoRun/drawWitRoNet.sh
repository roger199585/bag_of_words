cd $PWD

# # Single Map
# CUDA_VISIBLE_DEVICES=0,1 python evalRoNet/eval_aucroc.py --patch_size 64 --kmeans 128 --data pill --index 38 &
# CUDA_VISIBLE_DEVICES=2,3 python evalRoNet/eval_aucroc.py --patch_size 64 --kmeans 128 --data screw --index 31 &
# wait
# CUDA_VISIBLE_DEVICES=0,1 python evalRoNet/eval_aucroc.py --patch_size 64 --kmeans 128 --data tile --index 6 &
# CUDA_VISIBLE_DEVICES=2,3 python evalRoNet/eval_aucroc.py --patch_size 64 --kmeans 128 --data toothbrush --index 39 &
# wait
# CUDA_VISIBLE_DEVICES=0,1 python evalRoNet/eval_aucroc.py --patch_size 64 --kmeans 128 --data transistor --index 17 &
# CUDA_VISIBLE_DEVICES=2,3 python evalRoNet/eval_aucroc.py --patch_size 64 --kmeans 128 --data wood --index 13 &
# wait
# CUDA_VISIBLE_DEVICES=2,3 python evalRoNet/eval_aucroc.py --patch_size 64 --kmeans 128 --data zipper --index 2 &
# wait

# # Multi Map
# CUDA_VISIBLE_DEVICES=0,1 python evalRoNet/multi_map.py --patch_size 64 --kmeans 128 --data pill --index 38 &
# CUDA_VISIBLE_DEVICES=2,3 python evalRoNet/multi_map.py --patch_size 64 --kmeans 128 --data screw --index 32 &
# wait
# CUDA_VISIBLE_DEVICES=0,1 python evalRoNet/multi_map.py --patch_size 64 --kmeans 128 --data tile --index 6 &
# CUDA_VISIBLE_DEVICES=2,3 python evalRoNet/multi_map.py --patch_size 64 --kmeans 128 --data toothbrush --index 29 &
# wait
# CUDA_VISIBLE_DEVICES=0,1 python evalRoNet/multi_map.py --patch_size 64 --kmeans 128 --data transistor --index 17 &
# CUDA_VISIBLE_DEVICES=2,3 python evalRoNet/multi_map.py --patch_size 64 --kmeans 128 --data wood --index 13 &
# wait
# CUDA_VISIBLE_DEVICES=2,3 python evalRoNet/multi_map.py --patch_size 64 --kmeans 128 --data zipper --index 2 &
# wait

# # draw Multi Map
# CUDA_VISIBLE_DEVICES=0,1 python evalRoNet/draw_multiMap.py --patch_size 64 --kmeans 128 --data pill --index 38 &
# CUDA_VISIBLE_DEVICES=2,3 python evalRoNet/draw_multiMap.py --patch_size 64 --kmeans 128 --data screw --index 32 &
# wait
# CUDA_VISIBLE_DEVICES=0,1 python evalRoNet/draw_multiMap.py --patch_size 64 --kmeans 128 --data tile --index 6 &
# CUDA_VISIBLE_DEVICES=2,3 python evalRoNet/draw_multiMap.py --patch_size 64 --kmeans 128 --data toothbrush --index 29 &
# wait
# CUDA_VISIBLE_DEVICES=0,1 python evalRoNet/draw_multiMap.py --patch_size 64 --kmeans 128 --data transistor --index 17 &
# CUDA_VISIBLE_DEVICES=2,3 python evalRoNet/draw_multiMap.py --patch_size 64 --kmeans 128 --data wood --index 13 &
# wait
# CUDA_VISIBLE_DEVICES=2,3 python evalRoNet/draw_multiMap.py --patch_size 64 --kmeans 128 --data zipper --index 2 &
# wait


# Single Map
CUDA_VISIBLE_DEVICES=0,1 python evalRoNet/eval_aucroc.py --patch_size 64 --kmeans 128 --data tile --index 33 &
CUDA_VISIBLE_DEVICES=2,3 python evalRoNet/eval_aucroc.py --patch_size 64 --kmeans 128 --data zipper --index 30 &
wait

# Multi Map
CUDA_VISIBLE_DEVICES=0,1 python evalRoNet/multi_map.py --patch_size 64 --kmeans 128 --data tile --index 33 &
CUDA_VISIBLE_DEVICES=2,3 python evalRoNet/multi_map.py --patch_size 64 --kmeans 128 --data zipper --index 30 &
wait

# draw Multi Map
CUDA_VISIBLE_DEVICES=0,1 python evalRoNet/draw_multiMap.py --patch_size 64 --kmeans 128 --data tile --index 33 &
CUDA_VISIBLE_DEVICES=2,3 python evalRoNet/draw_multiMap.py --patch_size 64 --kmeans 128 --data zipper --index 30 &
wait
