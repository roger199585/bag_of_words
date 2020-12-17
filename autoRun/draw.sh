cd $PWD

CUDA_VISIBLE_DEVICES=0,1 python eval_aucroc.py --data cable --index 17 &
CUDA_VISIBLE_DEVICES=0,1 python multi_map.py --data cable --index 17 &
wait
python draw_multiMap.py --data cable --index 17
