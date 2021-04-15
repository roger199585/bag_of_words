# Calculate auc roc score
CUDA_VISIBLE_DEVICES=2 python eval_feature_first/eval_aucroc.py --data cable --kmeans 64 --index 25 &
CUDA_VISIBLE_DEVICES=2 python eval_feature_first/eval_aucroc.py --data bottle --kmeans 64 --index 12 &
wait

# Calculate multimap auc roc score
# CUDA_VISIBLE_DEVICES=0 python eval/multi_map.py --data tile --kmeans 128 --index 31 &
# wait

# Draw image and calculate PRO score
# python eval/draw_multiMap.py --data tile --kmeans 128 --index 31 &
# python eval/PRO.py --kmeans 128 --data tile --index 31 &
# wait