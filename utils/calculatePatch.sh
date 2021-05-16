#!/bin/bash

for i in $(seq 0 53503) ;
do
    python utils/pickPatches.py --root '/mnt/train-data1/corn/bottle/' --start $i &
    if [ `expr $i % 70` -eq 0 ]; then
        wait
    fi
done