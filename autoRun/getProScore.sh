cd $PWD

CUDA_VISIBLE_DEVICES=0,1 python eval/PRO.py --kmeans 128 --data bottle --index 7 &
CUDA_VISIBLE_DEVICES=2,3 python eval/PRO.py --kmeans 128 --data cable --index 2 &
CUDA_VISIBLE_DEVICES=0,1 python eval/PRO.py --kmeans 128 --data capsule --index 1 &
CUDA_VISIBLE_DEVICES=2,3 python eval/PRO.py --kmeans 128 --data carpet --index 1 &
CUDA_VISIBLE_DEVICES=0,1 python eval/PRO.py --kmeans 128 --data grid --index 27 &
CUDA_VISIBLE_DEVICES=2,3 python eval/PRO.py --kmeans 128 --data hazelnut --index 15 &
CUDA_VISIBLE_DEVICES=0,1 python eval/PRO.py --kmeans 128 --data leather --index 3 &
CUDA_VISIBLE_DEVICES=2,3 python eval/PRO.py --kmeans 128 --data metal_nut --index 9 &
CUDA_VISIBLE_DEVICES=0,1 python eval/PRO.py --kmeans 128 --data pill --index 13 &
CUDA_VISIBLE_DEVICES=2,3 python eval/PRO.py --kmeans 128 --data screw --index 26 &
CUDA_VISIBLE_DEVICES=0,1 python eval/PRO.py --kmeans 128 --data tile --index 20 &
CUDA_VISIBLE_DEVICES=2,3 python eval/PRO.py --kmeans 128 --data toothbrush --index 11 &
CUDA_VISIBLE_DEVICES=0,1 python eval/PRO.py --kmeans 128 --data transistor --index 3 &
CUDA_VISIBLE_DEVICES=2,3 python eval/PRO.py --kmeans 128 --data wood --index 7 &
CUDA_VISIBLE_DEVICES=0,1 python eval/PRO.py --kmeans 128 --data zipper --index 16 &
wait