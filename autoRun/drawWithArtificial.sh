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
CUDA_VISIBLE_DEVICES=2,3 python evalArtificial/eval_aucroc.py --patch_size 64 --kmeans 128 --data cable --index 12
CUDA_VISIBLE_DEVICES=2,3 python evalArtificial/eval_aucroc.py --patch_size 64 --kmeans 128 --data capsule --index 36
CUDA_VISIBLE_DEVICES=2,3 python evalArtificial/eval_aucroc.py --patch_size 64 --kmeans 128 --data carpet --index 12
CUDA_VISIBLE_DEVICES=2,3 python evalArtificial/eval_aucroc.py --patch_size 64 --kmeans 128 --data grid --index 25
CUDA_VISIBLE_DEVICES=2,3 python evalArtificial/eval_aucroc.py --patch_size 64 --kmeans 128 --data hazelnut --index 32
CUDA_VISIBLE_DEVICES=2,3 python evalArtificial/eval_aucroc.py --patch_size 64 --kmeans 128 --data pill --index 7
CUDA_VISIBLE_DEVICES=2,3 python evalArtificial/eval_aucroc.py --patch_size 64 --kmeans 128 --data screw --index 38
CUDA_VISIBLE_DEVICES=2,3 python evalArtificial/eval_aucroc.py --patch_size 64 --kmeans 128 --data tile --index 35
CUDA_VISIBLE_DEVICES=2,3 python evalArtificial/eval_aucroc.py --patch_size 64 --kmeans 128 --data toothbrush --index 4
CUDA_VISIBLE_DEVICES=2,3 python evalArtificial/eval_aucroc.py --patch_size 64 --kmeans 128 --data transistor --index 2
CUDA_VISIBLE_DEVICES=2,3 python evalArtificial/eval_aucroc.py --patch_size 64 --kmeans 128 --data wood --index 3


# Multi Map
CUDA_VISIBLE_DEVICES=2,3 python evalArtificial/multi_map.py --patch_size 64 --kmeans 128 --data cable --index 12
CUDA_VISIBLE_DEVICES=2,3 python evalArtificial/multi_map.py --patch_size 64 --kmeans 128 --data capsule --index 36
CUDA_VISIBLE_DEVICES=2,3 python evalArtificial/multi_map.py --patch_size 64 --kmeans 128 --data carpet --index 12
CUDA_VISIBLE_DEVICES=2,3 python evalArtificial/multi_map.py --patch_size 64 --kmeans 128 --data grid --index 25
CUDA_VISIBLE_DEVICES=2,3 python evalArtificial/multi_map.py --patch_size 64 --kmeans 128 --data hazelnut --index 32
CUDA_VISIBLE_DEVICES=2,3 python evalArtificial/multi_map.py --patch_size 64 --kmeans 128 --data pill --index 7
CUDA_VISIBLE_DEVICES=2,3 python evalArtificial/multi_map.py --patch_size 64 --kmeans 128 --data screw --index 38
CUDA_VISIBLE_DEVICES=2,3 python evalArtificial/multi_map.py --patch_size 64 --kmeans 128 --data tile --index 35
CUDA_VISIBLE_DEVICES=2,3 python evalArtificial/multi_map.py --patch_size 64 --kmeans 128 --data toothbrush --index 4
CUDA_VISIBLE_DEVICES=2,3 python evalArtificial/multi_map.py --patch_size 64 --kmeans 128 --data transistor --index 2
CUDA_VISIBLE_DEVICES=2,3 python evalArtificial/multi_map.py --patch_size 64 --kmeans 128 --data wood --index 3

# draw Multi Map
CUDA_VISIBLE_DEVICES=2,3 python evalArtificial/draw_multiMap.py --patch_size 64 --kmeans 128 --data cable --index 12 &
CUDA_VISIBLE_DEVICES=2,3 python evalArtificial/draw_multiMap.py --patch_size 64 --kmeans 128 --data capsule --index 36 &
CUDA_VISIBLE_DEVICES=2,3 python evalArtificial/draw_multiMap.py --patch_size 64 --kmeans 128 --data carpet --index 12 &
CUDA_VISIBLE_DEVICES=2,3 python evalArtificial/draw_multiMap.py --patch_size 64 --kmeans 128 --data grid --index 25 &
CUDA_VISIBLE_DEVICES=2,3 python evalArtificial/draw_multiMap.py --patch_size 64 --kmeans 128 --data hazelnut --index 32 &
CUDA_VISIBLE_DEVICES=2,3 python evalArtificial/draw_multiMap.py --patch_size 64 --kmeans 128 --data pill --index 7 &
CUDA_VISIBLE_DEVICES=2,3 python evalArtificial/draw_multiMap.py --patch_size 64 --kmeans 128 --data screw --index 38 &
CUDA_VISIBLE_DEVICES=2,3 python evalArtificial/draw_multiMap.py --patch_size 64 --kmeans 128 --data tile --index 35 &
CUDA_VISIBLE_DEVICES=2,3 python evalArtificial/draw_multiMap.py --patch_size 64 --kmeans 128 --data toothbrush --index 4 &
CUDA_VISIBLE_DEVICES=2,3 python evalArtificial/draw_multiMap.py --patch_size 64 --kmeans 128 --data transistor --index 2 &
CUDA_VISIBLE_DEVICES=2,3 python evalArtificial/draw_multiMap.py --patch_size 64 --kmeans 128 --data wood --index 3 &
wait
