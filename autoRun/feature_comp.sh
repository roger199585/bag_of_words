#!/bin/bash

for i in {0..6};
do
    for j in {0..6};
    do
        read -r mean_0 max_0 <<< "$(python eval_fine-tune.py --data bottle_1 --pos_x $i --pos_y $j --fine_tune_epoch 0)"
        read -r mean_5 max_5 <<< "$(python eval_fine-tune.py --data bottle_2 --pos_x $i --pos_y $j --fine_tune_epoch 5)"
        read -r mean_10 max_10 <<< "$(python eval_fine-tune.py --data bottle_3 --pos_x $i --pos_y $j --fine_tune_epoch 10)"
        read -r mean_15 max_15 <<< "$(python eval_fine-tune.py --data bottle --pos_x $i --pos_y $j --fine_tune_epoch 15)"

        read -r mean_diff <<< $(echo "scale=4; (($mean_15 - $mean_0) / $mean_0) * 100" | bc)
        read -r max_diff <<< $(echo "scale=4; (($max_15 - $max_0) / $max_0) * 100" | bc)
        echo "Pos ($i, $j) [ $mean_diff% / $max_diff% ]"

        echo -e "\t- tune 0"
        echo -e "\t\t- Mean: $mean_0"
        echo -e "\t\t- Max: $max_0"
        
        echo -e "\t - tune 5"
        echo -e "\t\t- Mean: $mean_5"
        echo -e "\t\t- Max: $max_5"

        echo -e "\t - tune 10"
        echo -e "\t\t- Mean: $mean_10"
        echo -e "\t\t- Max: $max_10"

        echo -e "\t - tune 15"
        echo -e "\t\t- Mean: $mean_15"
        echo -e "\t\t- Max: $max_15"
    done
done