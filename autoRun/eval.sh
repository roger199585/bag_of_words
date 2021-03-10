CUDA_VISIBLE_DEVICES=0 python eval_feature_first/eval_aucroc.py --data bottle --kmeans 128 --index 48 &
CUDA_VISIBLE_DEVICES=1 python eval_feature_first/eval_aucroc.py --data cable --kmeans 128 --index 31 &
CUDA_VISIBLE_DEVICES=2 python eval_feature_first/eval_aucroc.py --data capsule --kmeans 128 --index 51 &
CUDA_VISIBLE_DEVICES=3 python eval_feature_first/eval_aucroc.py --data carpet --kmeans 128 --index 34 &
CUDA_VISIBLE_DEVICES=0 python eval_feature_first/eval_aucroc.py --data grid --kmeans 128 --index 70 &
CUDA_VISIBLE_DEVICES=1 python eval_feature_first/eval_aucroc.py --data hazelnut --kmeans 128 --index 53 &
CUDA_VISIBLE_DEVICES=2 python eval_feature_first/eval_aucroc.py --data leather --kmeans 128 --index 54 &
CUDA_VISIBLE_DEVICES=3 python eval_feature_first/eval_aucroc.py --data metal_nut --kmeans 128 --index 75 &
wait