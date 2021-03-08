cd $PWD

# Single Map
# CUDA_VISIBLE_DEVICES=2,3 python evalArtificial/eval_aucroc.py --patch_size 64 --kmeans 128 --data cable --index 12
# CUDA_VISIBLE_DEVICES=2,3 python evalArtificial/eval_aucroc.py --patch_size 64 --kmeans 128 --data capsule --index 36
# CUDA_VISIBLE_DEVICES=2,3 python evalArtificial/eval_aucroc.py --patch_size 64 --kmeans 128 --data carpet --index 12
# CUDA_VISIBLE_DEVICES=2,3 python evalArtificial/eval_aucroc.py --patch_size 64 --kmeans 128 --data grid --index 25
# CUDA_VISIBLE_DEVICES=2,3 python evalArtificial/eval_aucroc.py --patch_size 64 --kmeans 128 --data hazelnut --index 32
# CUDA_VISIBLE_DEVICES=2,3 python evalArtificial/eval_aucroc.py --patch_size 64 --kmeans 128 --data pill --index 7
# CUDA_VISIBLE_DEVICES=2,3 python evalArtificial/eval_aucroc.py --patch_size 64 --kmeans 128 --data screw --index 38
CUDA_VISIBLE_DEVICES=2,3 python evalArtificial/eval_aucroc.py --patch_size 64 --kmeans 128 --data tile --index 40 &
# CUDA_VISIBLE_DEVICES=2,3 python evalArtificial/eval_aucroc.py --patch_size 64 --kmeans 128 --data toothbrush --index 4
# CUDA_VISIBLE_DEVICES=2,3 python evalArtificial/eval_aucroc.py --patch_size 64 --kmeans 128 --data transistor --index 2
# CUDA_VISIBLE_DEVICES=2,3 python evalArtificial/eval_aucroc.py --patch_size 64 --kmeans 128 --data wood --index 3


# Multi Map
# CUDA_VISIBLE_DEVICES=2,3 python evalArtificial/multi_map.py --patch_size 64 --kmeans 128 --data cable --index 12
# CUDA_VISIBLE_DEVICES=2,3 python evalArtificial/multi_map.py --patch_size 64 --kmeans 128 --data capsule --index 36
# CUDA_VISIBLE_DEVICES=2,3 python evalArtificial/multi_map.py --patch_size 64 --kmeans 128 --data carpet --index 12
# CUDA_VISIBLE_DEVICES=2,3 python evalArtificial/multi_map.py --patch_size 64 --kmeans 128 --data grid --index 25
# CUDA_VISIBLE_DEVICES=2,3 python evalArtificial/multi_map.py --patch_size 64 --kmeans 128 --data hazelnut --index 32
# CUDA_VISIBLE_DEVICES=2,3 python evalArtificial/multi_map.py --patch_size 64 --kmeans 128 --data pill --index 7
# CUDA_VISIBLE_DEVICES=2,3 python evalArtificial/multi_map.py --patch_size 64 --kmeans 128 --data screw --index 38
CUDA_VISIBLE_DEVICES=0,1 python evalArtificial/multi_map.py --patch_size 64 --kmeans 128 --data tile --index 40 &
# CUDA_VISIBLE_DEVICES=2,3 python evalArtificial/multi_map.py --patch_size 64 --kmeans 128 --data toothbrush --index 4
# CUDA_VISIBLE_DEVICES=2,3 python evalArtificial/multi_map.py --patch_size 64 --kmeans 128 --data transistor --index 2
# CUDA_VISIBLE_DEVICES=2,3 python evalArtificial/multi_map.py --patch_size 64 --kmeans 128 --data wood --index 3
wait 
# draw Multi Map
# CUDA_VISIBLE_DEVICES=2,3 python evalArtificial/draw_multiMap.py --patch_size 64 --kmeans 128 --data cable --index 12 &
# CUDA_VISIBLE_DEVICES=2,3 python evalArtificial/draw_multiMap.py --patch_size 64 --kmeans 128 --data capsule --index 36 &
# CUDA_VISIBLE_DEVICES=2,3 python evalArtificial/draw_multiMap.py --patch_size 64 --kmeans 128 --data carpet --index 12 &
# CUDA_VISIBLE_DEVICES=2,3 python evalArtificial/draw_multiMap.py --patch_size 64 --kmeans 128 --data grid --index 25 &
# CUDA_VISIBLE_DEVICES=2,3 python evalArtificial/draw_multiMap.py --patch_size 64 --kmeans 128 --data hazelnut --index 32 &
# CUDA_VISIBLE_DEVICES=2,3 python evalArtificial/draw_multiMap.py --patch_size 64 --kmeans 128 --data pill --index 7 &
# CUDA_VISIBLE_DEVICES=2,3 python evalArtificial/draw_multiMap.py --patch_size 64 --kmeans 128 --data screw --index 38 &
CUDA_VISIBLE_DEVICES=2,3 python evalArtificial/draw_multiMap.py --patch_size 64 --kmeans 128 --data tile --index 40 &
CUDA_VISIBLE_DEVICES=0,1 python evalArtificial/PRO.py --data tile --kmeans 128 --index 40 &
# CUDA_VISIBLE_DEVICES=2,3 python evalArtificial/draw_multiMap.py --patch_size 64 --kmeans 128 --data toothbrush --index 4 &
# CUDA_VISIBLE_DEVICES=2,3 python evalArtificial/draw_multiMap.py --patch_size 64 --kmeans 128 --data transistor --index 2 &
# CUDA_VISIBLE_DEVICES=2,3 python evalArtificial/draw_multiMap.py --patch_size 64 --kmeans 128 --data wood --index 3 &
wait
