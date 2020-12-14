cd $PWD

python eval_aucroc.py --data screw --index 1
python multi_map.py --data screw --index 1 && python draw_multiMap.py --data screw --index 1
