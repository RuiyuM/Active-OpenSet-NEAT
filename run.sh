



# active
CUDA_VISIBLE_DEVICES=0  nohup python AL_center_temperature.py --gpu 0 --save-dir ./log_AL/ --weight-cent 0 --query-strategy test_query --init-percent 8 --known-class 20 --query-batch 200 --seed 1 --model resnet18 --known-T 0.5 --unknown-T 0.5 --modelB-T 1 --dataset cifar100 --active --active_5 > ./log_AL/cifar100_active_5_200.txt &

CUDA_VISIBLE_DEVICES=1  nohup python AL_center_temperature.py --gpu 1 --save-dir ./log_AL/ --weight-cent 0 --query-strategy test_query --init-percent 8 --known-class 20 --query-batch 400 --seed 1 --model resnet18 --known-T 0.5 --unknown-T 0.5 --modelB-T 1 --dataset cifar100 --active --active_5 > ./log_AL/cifar100_active_5_400.txt &

CUDA_VISIBLE_DEVICES=2  nohup python AL_center_temperature.py --gpu 2 --save-dir ./log_AL/ --weight-cent 0 --query-strategy test_query --init-percent 8 --known-class 20 --query-batch 600 --seed 1 --model resnet18 --known-T 0.5 --unknown-T 0.5 --modelB-T 1 --dataset cifar100 --active --active_5 > ./log_AL/cifar100_active_5_600.txt &

CUDA_VISIBLE_DEVICES=3  nohup python AL_center_temperature.py --gpu 3 --save-dir ./log_AL/ --weight-cent 0 --query-strategy test_query --init-percent 8 --known-class 20 --query-batch 800 --seed 1 --model resnet18 --known-T 0.5 --unknown-T 0.5 --modelB-T 1 --dataset cifar100 --active --active_5 > ./log_AL/cifar100_active_5_800.txt &

CUDA_VISIBLE_DEVICES=4  nohup python AL_center_temperature.py --gpu 4 --save-dir ./log_AL/ --weight-cent 0 --query-strategy test_query --init-percent 8 --known-class 20 --query-batch 1500 --seed 1 --model resnet18 --known-T 0.5 --unknown-T 0.5 --modelB-T 1 --dataset cifar100 --active --active_5 > ./log_AL/cifar100_active_5_1500.txt &


# passive
CUDA_VISIBLE_DEVICES=5  nohup python AL_center_temperature.py --gpu 5 --save-dir ./log_AL/ --weight-cent 0 --query-strategy test_query --init-percent 8 --known-class 20 --query-batch 200 --seed 1 --model resnet18 --known-T 0.5 --unknown-T 0.5 --modelB-T 1 --dataset cifar100 > ./log_AL/cifar100_passive_200.txt &

CUDA_VISIBLE_DEVICES=6  nohup python AL_center_temperature.py --gpu 6 --save-dir ./log_AL/ --weight-cent 0 --query-strategy test_query --init-percent 8 --known-class 20 --query-batch 400 --seed 1 --model resnet18 --known-T 0.5 --unknown-T 0.5 --modelB-T 1 --dataset cifar100 > ./log_AL/cifar100_passive_400.txt &

CUDA_VISIBLE_DEVICES=7  nohup python AL_center_temperature.py --gpu 7 --save-dir ./log_AL/ --weight-cent 0 --query-strategy test_query --init-percent 8 --known-class 20 --query-batch 600 --seed 1 --model resnet18 --known-T 0.5 --unknown-T 0.5 --modelB-T 1 --dataset cifar100 > ./log_AL/cifar100_passive_600.txt &

CUDA_VISIBLE_DEVICES=8  nohup python AL_center_temperature.py --gpu 8 --save-dir ./log_AL/ --weight-cent 0 --query-strategy test_query --init-percent 8 --known-class 20 --query-batch 800 --seed 1 --model resnet18 --known-T 0.5 --unknown-T 0.5 --modelB-T 1 --dataset cifar100 > ./log_AL/cifar100_passive_800.txt &

CUDA_VISIBLE_DEVICES=9  nohup python AL_center_temperature.py --gpu 9 --save-dir ./log_AL/ --weight-cent 0 --query-strategy test_query --init-percent 8 --known-class 20 --query-batch 1500 --seed 1 --model resnet18 --known-T 0.5 --unknown-T 0.5 --modelB-T 1 --dataset cifar100 > ./log_AL/cifar100_passive_1500.txt &


# active_5_reverse

#CUDA_VISIBLE_DEVICES=5  nohup python AL_center_temperature.py --gpu 0 --save-dir ./log_AL/ --weight-cent 0 --query-strategy test_query --init-percent 8 --known-class 20 --query-batch 200 --seed 1 --model resnet18 --known-T 0.5 --unknown-T 0.5 --modelB-T 1 --dataset cifar100 --active --active_5_reverse > ./log_AL/cifar100_active_reverse_5_200.txt &

#CUDA_VISIBLE_DEVICES=6  nohup python AL_center_temperature.py --gpu 1 --save-dir ./log_AL/ --weight-cent 0 --query-strategy test_query --init-percent 8 --known-class 20 --query-batch 400 --seed 1 --model resnet18 --known-T 0.5 --unknown-T 0.5 --modelB-T 1 --dataset cifar100 --active --active_5_reverse > ./log_AL/cifar100_active_reverse_5_400.txt &

#CUDA_VISIBLE_DEVICES=7  nohup python AL_center_temperature.py --gpu 2 --save-dir ./log_AL/ --weight-cent 0 --query-strategy test_query --init-percent 8 --known-class 20 --query-batch 600 --seed 1 --model resnet18 --known-T 0.5 --unknown-T 0.5 --modelB-T 1 --dataset cifar100 --active --active_5_reverse > ./log_AL/cifar100_active_reverse_5_600.txt &

#CUDA_VISIBLE_DEVICES=8  nohup python AL_center_temperature.py --gpu 3 --save-dir ./log_AL/ --weight-cent 0 --query-strategy test_query --init-percent 8 --known-class 20 --query-batch 800 --seed 1 --model resnet18 --known-T 0.5 --unknown-T 0.5 --modelB-T 1 --dataset cifar100 --active --active_5_reverse > ./log_AL/cifar100_active_reverse_5_800.txt &

#CUDA_VISIBLE_DEVICES=9  nohup python AL_center_temperature.py --gpu 4 --save-dir ./log_AL/ --weight-cent 0 --query-strategy test_query --init-percent 8 --known-class 20 --query-batch 1500 --seed 1 --model resnet18 --known-T 0.5 --unknown-T 0.5 --modelB-T 1 --dataset cifar100 --active --active_5_reverse > ./log_AL/cifar100_active_reverse_5_1500.txt &


CUDA_VISIBLE_DEVICES=4   python AL_center_temperature.py --gpu 4 --save-dir ./log_AL/ --weight-cent 0 --query-strategy test_query --init-percent 8 --known-class 20 --query-batch 200 --seed 1 --model resnet18 --known-T 0.5 --unknown-T 0.5 --modelB-T 1 --dataset cifar100 --active --active_5
