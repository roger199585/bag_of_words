# Calculate auc roc score
CUDA_VISIBLE_DEVICES=2 python eval_feature_first/eval_aucroc.py --data bottle --kmeans 128 --index 24 --fine_tune_epoch 15 &
CUDA_VISIBLE_DEVICES=3 python eval_feature_first/eval_aucroc.py --data bottle_1 --kmeans 128 --index 36 --fine_tune_epoch 0 &
wait

# Calculate multimap auc roc score
# CUDA_VISIBLE_DEVICES=0 python eval/multi_map.py --data tile --kmeans 128 --index 31 &
# wait

# Draw image and calculate PRO score
# python eval/draw_multiMap.py --data tile --kmeans 128 --index 31 &
# python eval/PRO.py --kmeans 128 --data tile --index 31 &
# wait