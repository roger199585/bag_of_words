cd $PWD

CUDA_VISIBLE_DEVICES=2,3 python eval/eval_aucroc.py --patch_size 64 --kmeans 128 --data tile --index 26 &
CUDA_VISIBLE_DEVICES=0,1 python eval/multi_map.py --patch_size 64 --kmeans 128 --data tile --index 26 &

wait

python eval/draw_multiMap.py --patch_size 64 --kmeans 128 --data tile --index 26 &

wait
