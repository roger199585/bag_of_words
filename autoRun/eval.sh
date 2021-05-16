# Calculate auc roc score
CUDA_VISIBLE_DEVICES=0,1 python eval/eval_aucroc.py --data bottle --kmeans 128 --index 15 --fine_tune_epoch 60 &
wait

# Calculate multimap auc roc score
CUDA_VISIBLE_DEVICES=0,1 python eval/multi_map.py --data bottle --kmeans 128 --index 15 --fine_tune_epoch 60 &
wait

# Draw image and calculate PRO score
python eval/draw_multiMap.py --data bottle --kmeans 128 --index 15 &
python eval/PRO.py --data bottle --kmeans 128 --index 15 &
wait