cd $PWD

python eval_aucroc.py --data grid --index 27
python multi_map.py --data grid --index 27 && python draw_multiMap.py --data grid --index 27

# python eval_aucroc.py --data leather --index 3
# python multi_map.py --data leather --index 3 && python draw_multiMap.py --data leather --index 3

# python eval_aucroc.py --data metal_nut --index 9
# python multi_map.py --data metal_nut --index 9 && python draw_multiMap.py --data metal_nut --index 9

# python eval_aucroc.py --data carpet --index 1
# python multi_map.py --data carpet --index 1 && python draw_multiMap.py --data carpet --index 1
