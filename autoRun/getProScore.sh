cd $PWD

CUDA_VISIBLE_DEVICES=0,1 python evalRoNet/PRO.py --kmeans 128 --data bottle --index 2 &
CUDA_VISIBLE_DEVICES=2,3 python evalRoNet/PRO.py --kmeans 128 --data cable --index 7 &
CUDA_VISIBLE_DEVICES=0,1 python evalRoNet/PRO.py --kmeans 128 --data capsule --index 5 &
CUDA_VISIBLE_DEVICES=2,3 python evalRoNet/PRO.py --kmeans 128 --data carpet --index 25 &
CUDA_VISIBLE_DEVICES=0,1 python evalRoNet/PRO.py --kmeans 128 --data grid --index 34 &
CUDA_VISIBLE_DEVICES=2,3 python evalRoNet/PRO.py --kmeans 128 --data hazelnut --index 7 &
CUDA_VISIBLE_DEVICES=0,1 python evalRoNet/PRO.py --kmeans 128 --data leather --index 18 &
CUDA_VISIBLE_DEVICES=2,3 python evalRoNet/PRO.py --kmeans 128 --data metal_nut --index 29 &
wait