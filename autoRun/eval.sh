CUDA_VISIBLE_DEVICES=0 python eval/eval_aucroc.py --data capsule --kmeans 128 --index 50 &
CUDA_VISIBLE_DEVICES=1 python eval/eval_aucroc.py --data wood --kmeans 128 --index 27 &
wait

CUDA_VISIBLE_DEVICES=0 python eval/multi_map.py --data capsule --kmeans 128 --index 50 &
CUDA_VISIBLE_DEVICES=1 python eval/multi_map.py --data wood --kmeans 128 --index 27 &
wait

CUDA_VISIBLE_DEVICES=0 python eval/draw_multiMap.py --data capsule --kmeans 128 --index 50 &
CUDA_VISIBLE_DEVICES=1 python eval/draw_multiMap.py --data wood --kmeans 128 --index 27 &
wait