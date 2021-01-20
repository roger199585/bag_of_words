cd $PWD

CUDA_VISIBLE_DEVICES=2,3 python eval/eval_aucroc.py --dim_reduction PCA --patch_size 64 --kmeans 128 --data pill --index 13 &
CUDA_VISIBLE_DEVICES=2,3 python eval/multi_map.py --dim_reduction PCA --patch_size 64 --kmeans 128 --data pill --index 13 &
wait

CUDA_VISIBLE_DEVICES=0,1 python eval/draw_multiMap.py --patch_size 64 --kmeans 128 --data pill --index 13 &
wait
