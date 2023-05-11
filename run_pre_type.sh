#! /bin/bash

pre_types=("image18" "image34" "image50")

methods=("active_query")

for method in ${methods[@]}; 
do
	for pre_type in ${pre_types[@]}; 
	do
		for j in 400
		do
			#CUDA_VISIBLE_DEVICES=0 python AL_center_temperature.py --gpu 0 --save-dir log_AL/ --weight-cent 0 --query-strategy $method --init-percent 8 --known-class 40 --query-batch $j --seed 1 --model resnet18 --known-T 0.5 --unknown-T 0.5 --modelB-T 1 --dataset Tiny-Imagenet --pre-type $pre_type &
			#CUDA_VISIBLE_DEVICES=1 python AL_center_temperature.py --gpu 1 --save-dir log_AL/ --weight-cent 0 --query-strategy $method --init-percent 8 --known-class 40 --query-batch $j --seed 2 --model resnet18 --known-T 0.5 --unknown-T 0.5 --modelB-T 1 --dataset Tiny-Imagenet --pre-type $pre_type &
			#CUDA_VISIBLE_DEVICES=2 python AL_center_temperature.py --gpu 2 --save-dir log_AL/ --weight-cent 0 --query-strategy $method --init-percent 8 --known-class 40 --query-batch $j --seed 3 --model resnet18 --known-T 0.5 --unknown-T 0.5 --modelB-T 1 --dataset Tiny-Imagenet --pre-type $pre_type &

			CUDA_VISIBLE_DEVICES=6 python AL_center_temperature.py --gpu 6 --save-dir log_AL/ --weight-cent 0 --query-strategy $method --init-percent 8 --known-class 20 --query-batch $j --seed 1 --model resnet18 --known-T 0.5 --unknown-T 0.5 --modelB-T 1 --dataset cifar100 --pre-type $pre_type &
			CUDA_VISIBLE_DEVICES=7 python AL_center_temperature.py --gpu 7 --save-dir log_AL/ --weight-cent 0 --query-strategy $method --init-percent 8 --known-class 20 --query-batch $j --seed 2 --model resnet18 --known-T 0.5 --unknown-T 0.5 --modelB-T 1 --dataset cifar100 --pre-type $pre_type &
			CUDA_VISIBLE_DEVICES=8 python AL_center_temperature.py --gpu 8 --save-dir log_AL/ --weight-cent 0 --query-strategy $method --init-percent 8 --known-class 20 --query-batch $j --seed 3 --model resnet18 --known-T 0.5 --unknown-T 0.5 --modelB-T 1 --dataset cifar100 --pre-type $pre_type &

			#CUDA_VISIBLE_DEVICES=3 python AL_center_temperature.py --gpu 3 --save-dir log_AL/ --weight-cent 0 --query-strategy $method --init-percent 1 --known-class 2 --query-batch $j --seed 1 --model resnet18 --known-T 0.5 --unknown-T 0.5 --modelB-T 1 --dataset cifar10 --pre-type $pre_type &
			#CUDA_VISIBLE_DEVICES=4 python AL_center_temperature.py --gpu 4 --save-dir log_AL/ --weight-cent 0 --query-strategy $method --init-percent 1 --known-class 2 --query-batch $j --seed 2 --model resnet18 --known-T 0.5 --unknown-T 0.5 --modelB-T 1 --dataset cifar10 --pre-type $pre_type &
			#CUDA_VISIBLE_DEVICES=5 python AL_center_temperature.py --gpu 5 --save-dir log_AL/ --weight-cent 0 --query-strategy $method --init-percent 1 --known-class 2 --query-batch $j --seed 3 --model resnet18 --known-T 0.5 --unknown-T 0.5 --modelB-T 1 --dataset cifar10 --pre-type $pre_type &
			wait
		done
	done
done



