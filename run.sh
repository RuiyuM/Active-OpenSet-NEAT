#! /bin/bash

#methods=("active_query" "test_query")

methods=("BGADL" "Core_set" "uncertainty")
#methods=("BADGE_sampling")

for method in ${methods[@]}; 
do
	for j in 400
	do
		for i in 10
		do
			CUDA_VISIBLE_DEVICES=0 python AL_center_temperature.py --gpu 0 --k $i --save-dir log_AL/ --weight-cent 0 --query-strategy $method --init-percent 8 --known-class 40 --query-batch $j --seed 1 --model resnet18 --known-T 0.5 --unknown-T 0.5 --modelB-T 1 --dataset Tiny-Imagenet &
			CUDA_VISIBLE_DEVICES=1 python AL_center_temperature.py --gpu 1 --k $i --save-dir log_AL/ --weight-cent 0 --query-strategy $method --init-percent 8 --known-class 40 --query-batch $j --seed 2 --model resnet18 --known-T 0.5 --unknown-T 0.5 --modelB-T 1 --dataset Tiny-Imagenet &
			CUDA_VISIBLE_DEVICES=2 python AL_center_temperature.py --gpu 2 --k $i --save-dir log_AL/ --weight-cent 0 --query-strategy $method --init-percent 8 --known-class 40 --query-batch $j --seed 3 --model resnet18 --known-T 0.5 --unknown-T 0.5 --modelB-T 1 --dataset Tiny-Imagenet &

			#CUDA_VISIBLE_DEVICES=0 python NEAT_main.py --gpu 0 --k $i --save-dir log_AL/ --weight-cent 0 --query-strategy $method --init-percent 8 --known-class 20 --query-batch $j --seed 1 --model resnet18 --known-T 0.5 --unknown-T 0.5 --modelB-T 1 --dataset cifar100 &
			#CUDA_VISIBLE_DEVICES=0 python NEAT_main.py --gpu 0 --k $i --save-dir log_AL/ --weight-cent 0 --query-strategy $method --init-percent 8 --known-class 20 --query-batch $j --seed 2 --model resnet18 --known-T 0.5 --unknown-T 0.5 --modelB-T 1 --dataset cifar100 &
			#CUDA_VISIBLE_DEVICES=1 python NEAT_main.py --gpu 1 --k $i --save-dir log_AL/ --weight-cent 0 --query-strategy $method --init-percent 8 --known-class 20 --query-batch $j --seed 3 --model resnet18 --known-T 0.5 --unknown-T 0.5 --modelB-T 1 --dataset cifar100 &

			#CUDA_VISIBLE_DEVICES=1 python NEAT_main.py --gpu 1 --k $i --save-dir log_AL/ --weight-cent 0 --query-strategy $method --init-percent 1 --known-class 2 --query-batch $j --seed 1 --model resnet18 --known-T 0.5 --unknown-T 0.5 --modelB-T 1 --dataset cifar10 &
			#CUDA_VISIBLE_DEVICES=2 python NEAT_main.py --gpu 2 --k $i --save-dir log_AL/ --weight-cent 0 --query-strategy $method --init-percent 1 --known-class 2 --query-batch $j --seed 2 --model resnet18 --known-T 0.5 --unknown-T 0.5 --modelB-T 1 --dataset cifar10 &
			#CUDA_VISIBLE_DEVICES=2 python NEAT_main.py --gpu 2 --k $i --save-dir log_AL/ --weight-cent 0 --query-strategy $method --init-percent 1 --known-class 2 --query-batch $j --seed 3 --model resnet18 --known-T 0.5 --unknown-T 0.5 --modelB-T 1 --dataset cifar10 &
			wait
		done
	done
done


