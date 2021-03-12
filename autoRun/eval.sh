CUDA_VISIBLE_DEVICES=0 python eval_feature_first/eval_aucroc.py --data pill --kmeans 32 --index 33 &
CUDA_VISIBLE_DEVICES=1 python eval_feature_first/eval_aucroc.py --data screw --kmeans 32 --index 53 &
CUDA_VISIBLE_DEVICES=2 python eval_feature_first/eval_aucroc.py --data tile --kmeans 32 --index 8 &
CUDA_VISIBLE_DEVICES=3 python eval_feature_first/eval_aucroc.py --data toothbrush --kmeans 32 --index 16 &
CUDA_VISIBLE_DEVICES=0 python eval_feature_first/eval_aucroc.py --data transistor --kmeans 32 --index 20 &
CUDA_VISIBLE_DEVICES=1 python eval_feature_first/eval_aucroc.py --data wood --kmeans 32 --index 36 &
CUDA_VISIBLE_DEVICES=2 python eval_feature_first/eval_aucroc.py --data zipper --kmeans 32 --index 23 &
wait