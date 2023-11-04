#! /bin/bash

#methods=("active_query" "test_query")

methods=("Core_set" "OpenMax")
structures=('resnet18')

# shellcheck disable=SC2068
for method in ${methods[@]};
do
    for structure in ${structures[@]};
    do
        for j in 600 800
        do
            for i in 10
            do
                CUDA_VISIBLE_DEVICES=3 python NEAT_main.py --gpu 3 --k $i --save-dir log_AL/ --query-strategy $method --init-percent 8 --known-class 40 --query-batch $j --seed 1 --model $structure --dataset Tiny-Imagenet &
                CUDA_VISIBLE_DEVICES=2 python NEAT_main.py --gpu 2 --k $i --save-dir log_AL/ --query-strategy $method --init-percent 8 --known-class 40 --query-batch $j --seed 2 --model $structure --dataset Tiny-Imagenet &
                CUDA_VISIBLE_DEVICES=2 python NEAT_main.py --gpu 1 --k $i --save-dir log_AL/ --query-strategy $method --init-percent 8 --known-class 40 --query-batch $j --seed 3 --model $structure --dataset Tiny-Imagenet &
                wait
            done
        done
    done
done



