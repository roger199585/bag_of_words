# Calculate auc roc score
CUDA_VISIBLE_DEVICES=2 python eval/eval_aucroc.py --data bottle --kmeans 128 --index 13 --fine_tune_epoch 15 &
wait

# Calculate multimap auc roc score
CUDA_VISIBLE_DEVICES=2 python eval/multi_map.py --data bottle --kmeans 128 --index 13 --fine_tune_epoch 15 &
wait

# Draw image and calculate PRO score
python eval/draw_multiMap.py --data bottle --kmeans 128 --index 13 &
python eval/PRO.py --data bottle --kmeans 128 --index 13 &
wait