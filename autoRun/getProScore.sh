cd $PWD

python PRO.py --kmeans 128 --data grid --index 27 &
python PRO.py --kmeans 128 --data leather --index 3 &
python PRO.py --kmeans 128 --data capsule --index 1 &
python PRO.py --kmeans 128 --data bottle --index 7 &
python PRO.py --kmeans 128 --data metal_nut --index 9 &

wait
