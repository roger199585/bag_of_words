cd $PWD

# CUDA_VISIBLE_DEVICES=0,1 python eval/eval_aucroc.py --dim_reduction PCA --patch_size 64 --kmeans 128 --data bottle --index 7 &
# CUDA_VISIBLE_DEVICES=2,3 python eval/eval_aucroc.py --dim_reduction PCA --patch_size 64 --kmeans 128 --data cable --index 2 &
# wait
# CUDA_VISIBLE_DEVICES=0,1 python eval/eval_aucroc.py --dim_reduction PCA --patch_size 64 --kmeans 128 --data capsule --index 1 &
# CUDA_VISIBLE_DEVICES=2,3 python eval/eval_aucroc.py --dim_reduction PCA --patch_size 64 --kmeans 128 --data carpet --index 1 &
# wait
# CUDA_VISIBLE_DEVICES=0,1 python eval/eval_aucroc.py --dim_reduction PCA --patch_size 64 --kmeans 128 --data grid --index 27 &
# CUDA_VISIBLE_DEVICES=2,3 python eval/eval_aucroc.py --dim_reduction PCA --patch_size 64 --kmeans 128 --data hazelnut --index 15 &
# wait
# CUDA_VISIBLE_DEVICES=0,1 python eval/eval_aucroc.py --dim_reduction PCA --patch_size 64 --kmeans 128 --data leather --index 3 &
# CUDA_VISIBLE_DEVICES=2,3 python eval/eval_aucroc.py --dim_reduction PCA --patch_size 64 --kmeans 128 --data metal_nut --index 9 &
# wait
# CUDA_VISIBLE_DEVICES=0,1 python eval/eval_aucroc.py --dim_reduction PCA --patch_size 64 --kmeans 128 --data pill --index 13 &
CUDA_VISIBLE_DEVICES=2,3 python eval/eval_aucroc.py --dim_reduction PCA --patch_size 64 --kmeans 128 --data screw --index 26 &
# wait
# CUDA_VISIBLE_DEVICES=0,1 python eval/eval_aucroc.py --dim_reduction PCA --patch_size 64 --kmeans 128 --data tile --index 20 &
# CUDA_VISIBLE_DEVICES=2,3 python eval/eval_aucroc.py --dim_reduction PCA --patch_size 64 --kmeans 128 --data toothbrush --index 11 &
# wait
CUDA_VISIBLE_DEVICES=0,1 python eval/eval_aucroc.py --dim_reduction PCA --patch_size 64 --kmeans 128 --data transistor --index 3 &
# CUDA_VISIBLE_DEVICES=2,3 python eval/eval_aucroc.py --dim_reduction PCA --patch_size 64 --kmeans 128 --data wood --index 7 &
wait
# CUDA_VISIBLE_DEVICES=0,1 python eval/eval_aucroc.py --dim_reduction PCA --patch_size 64 --kmeans 128 --data zipper --index 16 &

# ====
# CUDA_VISIBLE_DEVICES=0,1 python eval/multi_map.py --dim_reduction PCA --patch_size 64 --kmeans 128 --data bottle --index 7 &
# wait
# CUDA_VISIBLE_DEVICES=2,3 python eval/multi_map.py --dim_reduction PCA --patch_size 64 --kmeans 128 --data cable --index 2 &
# CUDA_VISIBLE_DEVICES=0,1 python eval/multi_map.py --dim_reduction PCA --patch_size 64 --kmeans 128 --data capsule --index 1 &
# wait
# CUDA_VISIBLE_DEVICES=2,3 python eval/multi_map.py --dim_reduction PCA --patch_size 64 --kmeans 128 --data carpet --index 1 &
# CUDA_VISIBLE_DEVICES=0,1 python eval/multi_map.py --dim_reduction PCA --patch_size 64 --kmeans 128 --data grid --index 27 &
# wait
# CUDA_VISIBLE_DEVICES=2,3 python eval/multi_map.py --dim_reduction PCA --patch_size 64 --kmeans 128 --data hazelnut --index 15 &
# CUDA_VISIBLE_DEVICES=0,1 python eval/multi_map.py --dim_reduction PCA --patch_size 64 --kmeans 128 --data leather --index 3 &
# wait
# CUDA_VISIBLE_DEVICES=2,3 python eval/multi_map.py --dim_reduction PCA --patch_size 64 --kmeans 128 --data metal_nut --index 9 &
# CUDA_VISIBLE_DEVICES=0,1 python eval/multi_map.py --dim_reduction PCA --patch_size 64 --kmeans 128 --data pill --index 13 &
# wait
CUDA_VISIBLE_DEVICES=2,3 python eval/multi_map.py --dim_reduction PCA --patch_size 64 --kmeans 128 --data screw --index 26 &
# CUDA_VISIBLE_DEVICES=0,1 python eval/multi_map.py --dim_reduction PCA --patch_size 64 --kmeans 128 --data tile --index 20 &
# wait
# CUDA_VISIBLE_DEVICES=2,3 python eval/multi_map.py --dim_reduction PCA --patch_size 64 --kmeans 128 --data toothbrush --index 11 &
CUDA_VISIBLE_DEVICES=0,1 python eval/multi_map.py --dim_reduction PCA --patch_size 64 --kmeans 128 --data transistor --index 3 &
wait
# CUDA_VISIBLE_DEVICES=2,3 python eval/multi_map.py --dim_reduction PCA --patch_size 64 --kmeans 128 --data wood --index 7 &
# CUDA_VISIBLE_DEVICES=0,1 python eval/multi_map.py --dim_reduction PCA --patch_size 64 --kmeans 128 --data zipper --index 16 &
# wait

# CUDA_VISIBLE_DEVICES=0,1 python eval/draw_multiMap.py --patch_size 64 --kmeans 128 --data bottle --index 7 &
# CUDA_VISIBLE_DEVICES=2,3 python eval/draw_multiMap.py --patch_size 64 --kmeans 128 --data cable --index 2 &
# wait
# CUDA_VISIBLE_DEVICES=0,1 python eval/draw_multiMap.py --patch_size 64 --kmeans 128 --data capsule --index 1 &
# CUDA_VISIBLE_DEVICES=2,3 python eval/draw_multiMap.py --patch_size 64 --kmeans 128 --data carpet --index 1 &
# wait
# CUDA_VISIBLE_DEVICES=0,1 python eval/draw_multiMap.py --patch_size 64 --kmeans 128 --data grid --index 27 &
# CUDA_VISIBLE_DEVICES=2,3 python eval/draw_multiMap.py --patch_size 64 --kmeans 128 --data hazelnut --index 15 &
# wait
# CUDA_VISIBLE_DEVICES=0,1 python eval/draw_multiMap.py --patch_size 64 --kmeans 128 --data leather --index 3 &
# CUDA_VISIBLE_DEVICES=2,3 python eval/draw_multiMap.py --patch_size 64 --kmeans 128 --data metal_nut --index 9 &
# wait
# CUDA_VISIBLE_DEVICES=0,1 python eval/draw_multiMap.py --patch_size 64 --kmeans 128 --data pill --index 13 &
CUDA_VISIBLE_DEVICES=2,3 python eval/draw_multiMap.py --patch_size 64 --kmeans 128 --data screw --index 26 &
# wait
# CUDA_VISIBLE_DEVICES=0,1 python eval/draw_multiMap.py --patch_size 64 --kmeans 128 --data tile --index 20 &
# CUDA_VISIBLE_DEVICES=2,3 python eval/draw_multiMap.py --patch_size 64 --kmeans 128 --data toothbrush --index 11 &
# wait
CUDA_VISIBLE_DEVICES=0,1 python eval/draw_multiMap.py --patch_size 64 --kmeans 128 --data transistor --index 3 &
# CUDA_VISIBLE_DEVICES=2,3 python eval/draw_multiMap.py --patch_size 64 --kmeans 128 --data wood --index 7 &
# wait
# CUDA_VISIBLE_DEVICES=0,1 python eval/draw_multiMap.py --patch_size 64 --kmeans 128 --data zipper --index 16 &
wait
