#! /bin/bash

#methods=("active_query" "test_query")

methods=("active_query")

for method in ${methods[@]}; 
do
	for j in 400
	do
		for i in 5 10 15 20
		do
			CUDA_VISIBLE_DEVICES=0 python AL_center_temperature.py --gpu 0 --k $i --save-dir log_AL/ --weight-cent 0 --query-strategy $method --init-percent 8 --known-class 40 --query-batch $j --seed 1 --model resnet18 --known-T 0.5 --unknown-T 0.5 --modelB-T 1 --dataset Tiny-Imagenet &
			CUDA_VISIBLE_DEVICES=1 python AL_center_temperature.py --gpu 1 --k $i --save-dir log_AL/ --weight-cent 0 --query-strategy $method --init-percent 8 --known-class 40 --query-batch $j --seed 2 --model resnet18 --known-T 0.5 --unknown-T 0.5 --modelB-T 1 --dataset Tiny-Imagenet &
			CUDA_VISIBLE_DEVICES=2 python AL_center_temperature.py --gpu 2 --k $i --save-dir log_AL/ --weight-cent 0 --query-strategy $method --init-percent 8 --known-class 40 --query-batch $j --seed 3 --model resnet18 --known-T 0.5 --unknown-T 0.5 --modelB-T 1 --dataset Tiny-Imagenet &

			CUDA_VISIBLE_DEVICES=6 python AL_center_temperature.py --gpu 6 --k $i --save-dir log_AL/ --weight-cent 0 --query-strategy $method --init-percent 8 --known-class 20 --query-batch $j --seed 1 --model resnet18 --known-T 0.5 --unknown-T 0.5 --modelB-T 1 --dataset cifar100 &
			CUDA_VISIBLE_DEVICES=7 python AL_center_temperature.py --gpu 7 --k $i --save-dir log_AL/ --weight-cent 0 --query-strategy $method --init-percent 8 --known-class 20 --query-batch $j --seed 2 --model resnet18 --known-T 0.5 --unknown-T 0.5 --modelB-T 1 --dataset cifar100 &
			CUDA_VISIBLE_DEVICES=8 python AL_center_temperature.py --gpu 8 --k $i --save-dir log_AL/ --weight-cent 0 --query-strategy $method --init-percent 8 --known-class 20 --query-batch $j --seed 3 --model resnet18 --known-T 0.5 --unknown-T 0.5 --modelB-T 1 --dataset cifar100 &

			CUDA_VISIBLE_DEVICES=3 python AL_center_temperature.py --gpu 3 --k $i --save-dir log_AL/ --weight-cent 0 --query-strategy $method --init-percent 1 --known-class 2 --query-batch $j --seed 1 --model resnet18 --known-T 0.5 --unknown-T 0.5 --modelB-T 1 --dataset cifar10 &
			CUDA_VISIBLE_DEVICES=4 python AL_center_temperature.py --gpu 4 --k $i --save-dir log_AL/ --weight-cent 0 --query-strategy $method --init-percent 1 --known-class 2 --query-batch $j --seed 2 --model resnet18 --known-T 0.5 --unknown-T 0.5 --modelB-T 1 --dataset cifar10 &
			CUDA_VISIBLE_DEVICES=5 python AL_center_temperature.py --gpu 5 --k $i --save-dir log_AL/ --weight-cent 0 --query-strategy $method --init-percent 1 --known-class 2 --query-batch $j --seed 3 --model resnet18 --known-T 0.5 --unknown-T 0.5 --modelB-T 1 --dataset cifar10 &
			wait
		done
	done
done




#CUDA_VISIBLE_DEVICES=1 python AL_center_temperature.py --gpu 1 --save-dir log_AL/ --weight-cent 0 --query-strategy active_query --init-percent 8 --known-class 40 --query-batch 200 --seed 2 --model resnet18 --known-T 0.5 --unknown-T 0.5 --modelB-T 1 --dataset Tiny-Imagenet
