cd $PWD

# python eval_aucroc.py --data cable --index 2
# python multi_map.py --data cable --index 2 && python draw_multiMap.py --data cable --index 2

python eval_aucroc.py --data capsule --index 25
python multi_map.py --data capsule --index 25 && python draw_multiMap.py --data capsule --index 25

python eval_aucroc.py --data carpet --index 1
python multi_map.py --data carpet --index 1 && python draw_multiMap.py --data carpet --index 1

python eval_aucroc.py --data hazelnut --index 15
python multi_map.py --data hazelnut --index 15 && python draw_multiMap.py --data hazelnut --index 15
