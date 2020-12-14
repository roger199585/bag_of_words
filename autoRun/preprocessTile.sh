#!/bin/bash
count=1

while [ `python checkPreprocessData.py` == 0 ]
    do
        echo Round $count
        python BoW_PCA.py --data tile --kmeans 64 --dim 128 &
        wait
        echo BoW_PCA Done
        python assign_idx.py --data tile --dim 128 --kmeans 64 --type train &
        python assign_idx.py --data tile --dim 128 --kmeans 64 --type test &
        python assign_idx.py --data tile --dim 128 --kmeans 64 --type all &
        wait
        echo assign_idx Done
        python dataloaders.py --data tile --kmeans 64 &
        wait
        echo dataloaders Done
        python getCenterFeature.py --data tile --kmeans 64 &
        wait
        echo getCenterFeature Done
        count=$(($count + 1))
    done