CUDA_VISIBLE_DEVICES=0 python feature_extractor.py --data hazelnut --batch 32 --epochs 50000 --resolution 4 &
CUDA_VISIBLE_DEVICES=1 python feature_extractor.py --data hazelnut --batch 32 --epochs 50000 --resolution 8 &
CUDA_VISIBLE_DEVICES=2 python feature_extractor.py --data leather --batch 32 --epochs 50000 --resolution 4 &
CUDA_VISIBLE_DEVICES=3 python feature_extractor.py --data leather --batch 32 --epochs 50000 --resolution 8 &

CUDA_VISIBLE_DEVICES=0 python feature_extractor.py --data metal_nut --batch 32 --epochs 50000 --resolution 4 &
CUDA_VISIBLE_DEVICES=1 python feature_extractor.py --data metal_nut --batch 32 --epochs 50000 --resolution 8 &
CUDA_VISIBLE_DEVICES=2 python feature_extractor.py --data pill --batch 32 --epochs 50000 --resolution 4 &
CUDA_VISIBLE_DEVICES=3 python feature_extractor.py --data pill --batch 32 --epochs 50000 --resolution 8 &

wait
