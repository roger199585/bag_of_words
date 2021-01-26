cd $PWD

CUDA_VISIBLE_DEVICES=0,1 python evalRoNet/PRO.py --kmeans 128 --data pill --index 38 &
CUDA_VISIBLE_DEVICES=2,3 python evalRoNet/PRO.py --kmeans 128 --data screw --index 32 &
CUDA_VISIBLE_DEVICES=0,1 python evalRoNet/PRO.py --kmeans 128 --data tile --index 6 &
CUDA_VISIBLE_DEVICES=2,3 python evalRoNet/PRO.py --kmeans 128 --data toothbrush --index 29 &
CUDA_VISIBLE_DEVICES=0,1 python evalRoNet/PRO.py --kmeans 128 --data transistor --index 17 &
CUDA_VISIBLE_DEVICES=2,3 python evalRoNet/PRO.py --kmeans 128 --data wood --index 13 &
CUDA_VISIBLE_DEVICES=0,1 python evalRoNet/PRO.py --kmeans 128 --data zipper --index 2 &
wait