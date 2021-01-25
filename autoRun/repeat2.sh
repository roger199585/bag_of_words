type="zipper"

while [ "python preprocess_Artificial/checkPreprocessData.py --data $type" ]
do
    # 切 patch
    CUDA_VISIBLE_DEVICES=0,1 python preprocess_Artificial/artificial_feature.py --data $type &
    wait
    CUDA_VISIBLE_DEVICES=0,1 python preprocess_Artificial/BoW.py --data $type --kmeans 128 &
    wait
    CUDA_VISIBLE_DEVICES=0,1 python preprocess_Artificial/assign_idx.py --data $type --type train --kmeans 128 &
    CUDA_VISIBLE_DEVICES=0,1 python preprocess_Artificial/assign_idx.py --data $type --type test --kmeans 128 &
    CUDA_VISIBLE_DEVICES=0,1 python preprocess_Artificial/assign_idx.py --data $type --type all --kmeans 128 &
    wait
done
echo "$type finish"