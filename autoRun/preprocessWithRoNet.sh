# 切 patch
CUDA_VISIBLE_DEVICES=0,1 python preprocess_Artificial/artificial_feature.py --data bottle &
# CUDA_VISIBLE_DEVICES=2,3 python preprocess_Artificial/artificial_feature.py --data cable &
# CUDA_VISIBLE_DEVICES=0,1 python preprocess_Artificial/artificial_feature.py --data capsule &
# CUDA_VISIBLE_DEVICES=2,3 python preprocess_Artificial/artificial_feature.py --data carpet &
# CUDA_VISIBLE_DEVICES=0,1 python preprocess_Artificial/artificial_feature.py --data grid &
# CUDA_VISIBLE_DEVICES=2,3 python preprocess_Artificial/artificial_feature.py --data hazelnut &
# CUDA_VISIBLE_DEVICES=0,1 python preprocess_Artificial/artificial_feature.py --data leather &
# CUDA_VISIBLE_DEVICES=2,3 python preprocess_Artificial/artificial_feature.py --data metal_nut &
# CUDA_VISIBLE_DEVICES=0,1 python preprocess_Artificial/artificial_feature.py --data pill &
# CUDA_VISIBLE_DEVICES=2,3 python preprocess_Artificial/artificial_feature.py --data screw &
# CUDA_VISIBLE_DEVICES=0,1 python preprocess_Artificial/artificial_feature.py --data tile &
# CUDA_VISIBLE_DEVICES=2,3 python preprocess_Artificial/artificial_feature.py --data toothbrush &
# CUDA_VISIBLE_DEVICES=0,1 python preprocess_Artificial/artificial_feature.py --data transistor &
# CUDA_VISIBLE_DEVICES=2,3 python preprocess_Artificial/artificial_feature.py --data wood &
CUDA_VISIBLE_DEVICES=0,1 python preprocess_Artificial/artificial_feature.py --data zipper &
wait
# CUDA_VISIBLE_DEVICES=0,1 python preprocess_RoNet/pretrain_RoNet.py --data tile &
# CUDA_VISIBLE_DEVICES=2,3 python preprocess_RoNet/pretrain_RoNet.py --data cable &
# wait
# CUDA_VISIBLE_DEVICES=0,1 python preprocess_RoNet/pretrain_RoNet.py --data capsule &
# CUDA_VISIBLE_DEVICES=2,3 python preprocess_RoNet/pretrain_RoNet.py --data carpet &
# wait
# CUDA_VISIBLE_DEVICES=0,1 python preprocess_RoNet/pretrain_RoNet.py --data grid &
# CUDA_VISIBLE_DEVICES=2,3 python preprocess_RoNet/pretrain_RoNet.py --data hazelnut &
# wait
# CUDA_VISIBLE_DEVICES=0,1 python preprocess_RoNet/pretrain_RoNet.py --data leather &
# CUDA_VISIBLE_DEVICES=2,3 python preprocess_RoNet/pretrain_RoNet.py --data metal_nut &
# wait
# CUDA_VISIBLE_DEVICES=0,1 python preprocess_RoNet/pretrain_RoNet.py --data pill &
# CUDA_VISIBLE_DEVICES=2,3 python preprocess_RoNet/pretrain_RoNet.py --data screw &
# wait
# CUDA_VISIBLE_DEVICES=0,1 python preprocess_RoNet/pretrain_RoNet.py --data tile &
# CUDA_VISIBLE_DEVICES=2,3 python preprocess_RoNet/pretrain_RoNet.py --data toothbrush &
# wait
# CUDA_VISIBLE_DEVICES=0,1 python preprocess_RoNet/pretrain_RoNet.py --data transistor &
# CUDA_VISIBLE_DEVICES=2,3 python preprocess_RoNet/pretrain_RoNet.py --data wood &
# wait
# CUDA_VISIBLE_DEVICES=0,1 python preprocess_RoNet/pretrain_RoNet.py --data zipper &

# 分群生成 Bow
# CUDA_VISIBLE_DEVICES=2,3 python preprocess_Artificial/BoW.py --data tile --kmeans 128 &
# wait 
# CUDA_VISIBLE_DEVICES=0,1 python preprocess_RoNet/BoW.py --data cable --kmeans 128 &
# CUDA_VISIBLE_DEVICES=2,3 python preprocess_RoNet/BoW.py --data capsule --kmeans 128 &
# wait 
# CUDA_VISIBLE_DEVICES=0,1 python preprocess_RoNet/BoW.py --data carpet --kmeans 128 &
# CUDA_VISIBLE_DEVICES=2,3 python preprocess_RoNet/BoW.py --data grid --kmeans 128 &
# wait 
# CUDA_VISIBLE_DEVICES=0,1 python preprocess_RoNet/BoW.py --data hazelnut --kmeans 128 &
# CUDA_VISIBLE_DEVICES=2,3 python preprocess_RoNet/BoW.py --data leather --kmeans 128 &
# wait 
# CUDA_VISIBLE_DEVICES=0,1 python preprocess_RoNet/BoW.py --data metal_nut --kmeans 128 &
# CUDA_VISIBLE_DEVICES=2,3 python preprocess_RoNet/BoW.py --data pill --kmeans 128 &
# wait 
# CUDA_VISIBLE_DEVICES=0,1 python preprocess_RoNet/BoW.py --data screw --kmeans 128 &
# CUDA_VISIBLE_DEVICES=2,3 python preprocess_RoNet/BoW.py --data tile --kmeans 128 &
# wait 
# CUDA_VISIBLE_DEVICES=0,1 python preprocess_RoNet/BoW.py --data toothbrush --kmeans 128 &
# CUDA_VISIBLE_DEVICES=2,3 python preprocess_RoNet/BoW.py --data transistor --kmeans 128 &
# wait 
# CUDA_VISIBLE_DEVICES=0,1 python preprocess_RoNet/BoW.py --data wood --kmeans 128 &
# CUDA_VISIBLE_DEVICES=2,3 python preprocess_RoNet/BoW.py --data zipper --kmeans 128 &
# wait

CUDA_VISIBLE_DEVICES=2,3 python preprocess_Artificial/BoW.py --data bottle --kmeans 128 &
# CUDA_VISIBLE_DEVICES=0,1 python preprocess_Artificial/BoW.py --data cable --kmeans 128 &
# CUDA_VISIBLE_DEVICES=2,3 python preprocess_Artificial/BoW.py --data capsule --kmeans 128 &
# CUDA_VISIBLE_DEVICES=0,1 python preprocess_Artificial/BoW.py --data carpet --kmeans 128 &
# CUDA_VISIBLE_DEVICES=2,3 python preprocess_Artificial/BoW.py --data grid --kmeans 128 &
# CUDA_VISIBLE_DEVICES=0,1 python preprocess_Artificial/BoW.py --data hazelnut --kmeans 128 &
# CUDA_VISIBLE_DEVICES=2,3 python preprocess_Artificial/BoW.py --data leather --kmeans 128 &
# CUDA_VISIBLE_DEVICES=0,1 python preprocess_Artificial/BoW.py --data metal_nut --kmeans 128 &
# CUDA_VISIBLE_DEVICES=2,3 python preprocess_Artificial/BoW.py --data pill --kmeans 128 &
# CUDA_VISIBLE_DEVICES=0,1 python preprocess_Artificial/BoW.py --data screw --kmeans 128 &
# CUDA_VISIBLE_DEVICES=2,3 python preprocess_Artificial/BoW.py --data tile --kmeans 128 &
# CUDA_VISIBLE_DEVICES=0,1 python preprocess_Artificial/BoW.py --data toothbrush --kmeans 128 &
# CUDA_VISIBLE_DEVICES=2,3 python preprocess_Artificial/BoW.py --data transistor --kmeans 128 &
# CUDA_VISIBLE_DEVICES=0,1 python preprocess_Artificial/BoW.py --data wood --kmeans 128 &
CUDA_VISIBLE_DEVICES=2,3 python preprocess_Artificial/BoW.py --data zipper --kmeans 128 &
wait

# 生成 GT
# CUDA_VISIBLE_DEVICES=0,1 python preprocess_RoNet/assign_idx.py --data tile --type train --kmeans 128 &
# CUDA_VISIBLE_DEVICES=0,1 python preprocess_RoNet/assign_idx.py --data tile --type test --kmeans 128 &
# CUDA_VISIBLE_DEVICES=0,1 python preprocess_RoNet/assign_idx.py --data tile --type all --kmeans 128 &
# CUDA_VISIBLE_DEVICES=2,3 python preprocess_RoNet/assign_idx.py --data cable --type train --kmeans 128 &
# CUDA_VISIBLE_DEVICES=2,3 python preprocess_RoNet/assign_idx.py --data cable --type test --kmeans 128 &
# CUDA_VISIBLE_DEVICES=2,3 python preprocess_RoNet/assign_idx.py --data cable --type all --kmeans 128 &
# wait
# CUDA_VISIBLE_DEVICES=0,1 python preprocess_RoNet/assign_idx.py --data capsule --type train --kmeans 128 &
# CUDA_VISIBLE_DEVICES=0,1 python preprocess_RoNet/assign_idx.py --data capsule --type test --kmeans 128 &
# CUDA_VISIBLE_DEVICES=0,1 python preprocess_RoNet/assign_idx.py --data capsule --type all --kmeans 128 &
# CUDA_VISIBLE_DEVICES=2,3 python preprocess_RoNet/assign_idx.py --data carpet --type train --kmeans 128 &
# CUDA_VISIBLE_DEVICES=2,3 python preprocess_RoNet/assign_idx.py --data carpet --type test --kmeans 128 &
# CUDA_VISIBLE_DEVICES=2,3 python preprocess_RoNet/assign_idx.py --data carpet --type all --kmeans 128 &
# wait
# CUDA_VISIBLE_DEVICES=0,1 python preprocess_RoNet/assign_idx.py --data grid --type train --kmeans 128 &
# CUDA_VISIBLE_DEVICES=0,1 python preprocess_RoNet/assign_idx.py --data grid --type test --kmeans 128 &
# CUDA_VISIBLE_DEVICES=0,1 python preprocess_RoNet/assign_idx.py --data grid --type all --kmeans 128 &
# CUDA_VISIBLE_DEVICES=2,3 python preprocess_RoNet/assign_idx.py --data hazelnut --type train --kmeans 128 &
# CUDA_VISIBLE_DEVICES=2,3 python preprocess_RoNet/assign_idx.py --data hazelnut --type test --kmeans 128 &
# CUDA_VISIBLE_DEVICES=2,3 python preprocess_RoNet/assign_idx.py --data hazelnut --type all --kmeans 128 &
# wait
# CUDA_VISIBLE_DEVICES=0,1 python preprocess_RoNet/assign_idx.py --data leather --type train --kmeans 128 &
# CUDA_VISIBLE_DEVICES=0,1 python preprocess_RoNet/assign_idx.py --data leather --type test --kmeans 128 &
# CUDA_VISIBLE_DEVICES=0,1 python preprocess_RoNet/assign_idx.py --data leather --type all --kmeans 128 &
# CUDA_VISIBLE_DEVICES=2,3 python preprocess_RoNet/assign_idx.py --data metal_nut --type train --kmeans 128 &
# CUDA_VISIBLE_DEVICES=2,3 python preprocess_RoNet/assign_idx.py --data metal_nut --type test --kmeans 128 &
# CUDA_VISIBLE_DEVICES=2,3 python preprocess_RoNet/assign_idx.py --data metal_nut --type all --kmeans 128 &
# wait
# CUDA_VISIBLE_DEVICES=0,1 python preprocess_RoNet/assign_idx.py --data pill --type train --kmeans 128 &
# CUDA_VISIBLE_DEVICES=0,1 python preprocess_RoNet/assign_idx.py --data pill --type test --kmeans 128 &
# CUDA_VISIBLE_DEVICES=0,1 python preprocess_RoNet/assign_idx.py --data pill --type all --kmeans 128 &
# CUDA_VISIBLE_DEVICES=2,3 python preprocess_RoNet/assign_idx.py --data screw --type train --kmeans 128 &
# CUDA_VISIBLE_DEVICES=2,3 python preprocess_RoNet/assign_idx.py --data screw --type test --kmeans 128 &
# CUDA_VISIBLE_DEVICES=2,3 python preprocess_RoNet/assign_idx.py --data screw --type all --kmeans 128 &
# wait
# CUDA_VISIBLE_DEVICES=0,1 python preprocess_RoNet/assign_idx.py --data tile --type train --kmeans 128 &
# CUDA_VISIBLE_DEVICES=0,1 python preprocess_RoNet/assign_idx.py --data tile --type test --kmeans 128 &
# CUDA_VISIBLE_DEVICES=0,1 python preprocess_RoNet/assign_idx.py --data tile --type all --kmeans 128 &
# CUDA_VISIBLE_DEVICES=2,3 python preprocess_RoNet/assign_idx.py --data toothbrush --type train --kmeans 128 &
# CUDA_VISIBLE_DEVICES=2,3 python preprocess_RoNet/assign_idx.py --data toothbrush --type test --kmeans 128 &
# CUDA_VISIBLE_DEVICES=2,3 python preprocess_RoNet/assign_idx.py --data toothbrush --type all --kmeans 128 &
# wait
# CUDA_VISIBLE_DEVICES=0,1 python preprocess_RoNet/assign_idx.py --data transistor --type train --kmeans 128 &
# CUDA_VISIBLE_DEVICES=0,1 python preprocess_RoNet/assign_idx.py --data transistor --type test --kmeans 128 &
# CUDA_VISIBLE_DEVICES=0,1 python preprocess_RoNet/assign_idx.py --data transistor --type all --kmeans 128 &
# CUDA_VISIBLE_DEVICES=2,3 python preprocess_RoNet/assign_idx.py --data wood --type train --kmeans 128 &
# CUDA_VISIBLE_DEVICES=2,3 python preprocess_RoNet/assign_idx.py --data wood --type test --kmeans 128 &
# CUDA_VISIBLE_DEVICES=2,3 python preprocess_RoNet/assign_idx.py --data wood --type all --kmeans 128 &
# wait
# CUDA_VISIBLE_DEVICES=0,1 python preprocess_RoNet/assign_idx.py --data zipper --type train --kmeans 128 &
# CUDA_VISIBLE_DEVICES=0,1 python preprocess_RoNet/assign_idx.py --data zipper --type test --kmeans 128 &
# CUDA_VISIBLE_DEVICES=0,1 python preprocess_RoNet/assign_idx.py --data zipper --type all --kmeans 128 &
# wait


CUDA_VISIBLE_DEVICES=0,1 python preprocess_Artificial/assign_idx.py --data bottle --type train --kmeans 128 &
CUDA_VISIBLE_DEVICES=0,1 python preprocess_Artificial/assign_idx.py --data bottle --type test --kmeans 128 &
CUDA_VISIBLE_DEVICES=0,1 python preprocess_Artificial/assign_idx.py --data bottle --type all --kmeans 128 &
# CUDA_VISIBLE_DEVICES=2,3 python preprocess_Artificial/assign_idx.py --data cable --type train --kmeans 128 &
# CUDA_VISIBLE_DEVICES=2,3 python preprocess_Artificial/assign_idx.py --data cable --type test --kmeans 128 &
# CUDA_VISIBLE_DEVICES=2,3 python preprocess_Artificial/assign_idx.py --data cable --type all --kmeans 128 &
# wait
# CUDA_VISIBLE_DEVICES=0,1 python preprocess_Artificial/assign_idx.py --data capsule --type train --kmeans 128 &
# CUDA_VISIBLE_DEVICES=0,1 python preprocess_Artificial/assign_idx.py --data capsule --type test --kmeans 128 &
# CUDA_VISIBLE_DEVICES=0,1 python preprocess_Artificial/assign_idx.py --data capsule --type all --kmeans 128 &
# CUDA_VISIBLE_DEVICES=2,3 python preprocess_Artificial/assign_idx.py --data carpet --type train --kmeans 128 &
# CUDA_VISIBLE_DEVICES=2,3 python preprocess_Artificial/assign_idx.py --data carpet --type test --kmeans 128 &
# CUDA_VISIBLE_DEVICES=2,3 python preprocess_Artificial/assign_idx.py --data carpet --type all --kmeans 128 &
# wait
# CUDA_VISIBLE_DEVICES=0,1 python preprocess_Artificial/assign_idx.py --data grid --type train --kmeans 128 &
# CUDA_VISIBLE_DEVICES=0,1 python preprocess_Artificial/assign_idx.py --data grid --type test --kmeans 128 &
# CUDA_VISIBLE_DEVICES=0,1 python preprocess_Artificial/assign_idx.py --data grid --type all --kmeans 128 &
# CUDA_VISIBLE_DEVICES=2,3 python preprocess_Artificial/assign_idx.py --data hazelnut --type train --kmeans 128 &
# CUDA_VISIBLE_DEVICES=2,3 python preprocess_Artificial/assign_idx.py --data hazelnut --type test --kmeans 128 &
# CUDA_VISIBLE_DEVICES=2,3 python preprocess_Artificial/assign_idx.py --data hazelnut --type all --kmeans 128 &
# wait
# CUDA_VISIBLE_DEVICES=0,1 python preprocess_Artificial/assign_idx.py --data leather --type train --kmeans 128 &
# CUDA_VISIBLE_DEVICES=0,1 python preprocess_Artificial/assign_idx.py --data leather --type test --kmeans 128 &
# CUDA_VISIBLE_DEVICES=0,1 python preprocess_Artificial/assign_idx.py --data leather --type all --kmeans 128 &
# CUDA_VISIBLE_DEVICES=2,3 python preprocess_Artificial/assign_idx.py --data metal_nut --type train --kmeans 128 &
# CUDA_VISIBLE_DEVICES=2,3 python preprocess_Artificial/assign_idx.py --data metal_nut --type test --kmeans 128 &
# CUDA_VISIBLE_DEVICES=2,3 python preprocess_Artificial/assign_idx.py --data metal_nut --type all --kmeans 128 &
# wait
# CUDA_VISIBLE_DEVICES=0,1 python preprocess_Artificial/assign_idx.py --data pill --type train --kmeans 128 &
# CUDA_VISIBLE_DEVICES=0,1 python preprocess_Artificial/assign_idx.py --data pill --type test --kmeans 128 &
# CUDA_VISIBLE_DEVICES=0,1 python preprocess_Artificial/assign_idx.py --data pill --type all --kmeans 128 &
# CUDA_VISIBLE_DEVICES=2,3 python preprocess_Artificial/assign_idx.py --data screw --type train --kmeans 128 &
# CUDA_VISIBLE_DEVICES=2,3 python preprocess_Artificial/assign_idx.py --data screw --type test --kmeans 128 &
# CUDA_VISIBLE_DEVICES=2,3 python preprocess_Artificial/assign_idx.py --data screw --type all --kmeans 128 &
# wait
# CUDA_VISIBLE_DEVICES=0,1 python preprocess_Artificial/assign_idx.py --data tile --type train --kmeans 128 &
# CUDA_VISIBLE_DEVICES=0,1 python preprocess_Artificial/assign_idx.py --data tile --type test --kmeans 128 &
# CUDA_VISIBLE_DEVICES=0,1 python preprocess_Artificial/assign_idx.py --data tile --type all --kmeans 128 &
# CUDA_VISIBLE_DEVICES=2,3 python preprocess_Artificial/assign_idx.py --data toothbrush --type train --kmeans 128 &
# CUDA_VISIBLE_DEVICES=2,3 python preprocess_Artificial/assign_idx.py --data toothbrush --type test --kmeans 128 &
# CUDA_VISIBLE_DEVICES=2,3 python preprocess_Artificial/assign_idx.py --data toothbrush --type all --kmeans 128 &
# wait
# CUDA_VISIBLE_DEVICES=0,1 python preprocess_Artificial/assign_idx.py --data transistor --type train --kmeans 128 &
# CUDA_VISIBLE_DEVICES=0,1 python preprocess_Artificial/assign_idx.py --data transistor --type test --kmeans 128 &
# CUDA_VISIBLE_DEVICES=0,1 python preprocess_Artificial/assign_idx.py --data transistor --type all --kmeans 128 &
# CUDA_VISIBLE_DEVICES=2,3 python preprocess_Artificial/assign_idx.py --data wood --type train --kmeans 128 &
# CUDA_VISIBLE_DEVICES=2,3 python preprocess_Artificial/assign_idx.py --data wood --type test --kmeans 128 &
# CUDA_VISIBLE_DEVICES=2,3 python preprocess_Artificial/assign_idx.py --data wood --type all --kmeans 128 &
# wait
CUDA_VISIBLE_DEVICES=0,1 python preprocess_Artificial/assign_idx.py --data zipper --type train --kmeans 128 &
CUDA_VISIBLE_DEVICES=0,1 python preprocess_Artificial/assign_idx.py --data zipper --type test --kmeans 128 &
CUDA_VISIBLE_DEVICES=0,1 python preprocess_Artificial/assign_idx.py --data zipper --type all --kmeans 128 &
wait

# python preprocess_Artificial/getCenterFeature.py --data tile --kmeans 128 &
# python preprocess_RoNet/getCenterFeature.py --data cable --kmeans 128 &
# python preprocess_RoNet/getCenterFeature.py --data capsule --kmeans 128 &
# python preprocess_RoNet/getCenterFeature.py --data carpet --kmeans 128 &
# python preprocess_RoNet/getCenterFeature.py --data grid --kmeans 128 &
# python preprocess_RoNet/getCenterFeature.py --data hazelnut --kmeans 128 &
# python preprocess_RoNet/getCenterFeature.py --data leather --kmeans 128 &
# python preprocess_RoNet/getCenterFeature.py --data metal_nut --kmeans 128 &
# python preprocess_RoNet/getCenterFeature.py --data pill --kmeans 128 &
# python preprocess_RoNet/getCenterFeature.py --data screw --kmeans 128 &
# python preprocess_RoNet/getCenterFeature.py --data tile --kmeans 128 &
# python preprocess_RoNet/getCenterFeature.py --data toothbrush --kmeans 128 &
# python preprocess_RoNet/getCenterFeature.py --data transistor --kmeans 128 &
# python preprocess_RoNet/getCenterFeature.py --data wood --kmeans 128 &
# python preprocess_RoNet/getCenterFeature.py --data zipper --kmeans 128 &
# wait

python preprocess_Artificial/getCenterFeature.py --data bottle --kmeans 128 &
# python preprocess_Artificial/getCenterFeature.py --data cable --kmeans 128 &
# python preprocess_Artificial/getCenterFeature.py --data capsule --kmeans 128 &
# python preprocess_Artificial/getCenterFeature.py --data carpet --kmeans 128 &
# python preprocess_Artificial/getCenterFeature.py --data grid --kmeans 128 &
# python preprocess_Artificial/getCenterFeature.py --data hazelnut --kmeans 128 &
# python preprocess_Artificial/getCenterFeature.py --data leather --kmeans 128 &
# python preprocess_Artificial/getCenterFeature.py --data metal_nut --kmeans 128 &
# python preprocess_Artificial/getCenterFeature.py --data pill --kmeans 128 &
# python preprocess_Artificial/getCenterFeature.py --data screw --kmeans 128 &
# python preprocess_Artificial/getCenterFeature.py --data tile --kmeans 128 &
# python preprocess_Artificial/getCenterFeature.py --data toothbrush --kmeans 128 &
# python preprocess_Artificial/getCenterFeature.py --data transistor --kmeans 128 &
# python preprocess_Artificial/getCenterFeature.py --data wood --kmeans 128 &
python preprocess_Artificial/getCenterFeature.py --data zipper --kmeans 128 &
wait 

python preprocess_Artificial/checkPreprocessData.py