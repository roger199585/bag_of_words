cd $PWD

python draw_multiMap.py --kmeans 128 --data grid --index 27 &
wait
python draw_multiMap.py --kmeans 128 --data metal_nut --index 9 &
wait
