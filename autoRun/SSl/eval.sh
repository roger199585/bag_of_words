# Calculate auc roc score
CUDA_VISIBLE_DEVICES=2,3 python eval_SSL/eval_aucroc.py --data bottle --kmeans 128 --index 42 &
# Calculate multimap auc roc score
CUDA_VISIBLE_DEVICES=2,3 python eval_SSL/multi_map.py --data bottle --kmeans 128 --index 42 &
wait

# Draw image and calculate PRO score
python eval_SSL/draw_multiMap.py --data bottle --kmeans 128 --index 42 &
python eval_SSL/PRO.py --data bottle --kmeans 128 --index 42 &
wait