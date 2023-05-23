# Inconsistency-Based Data-Centric Active Open-Set Annotation
An implementation of NEAT batch active learning algorithm for Open-Set Annotation.
Details are provided in our paper[***]



## 1. Requirements
### Environments
Currently, requires following packages. (We are using CUDA == 11.6, 
python == 3.9,
pytorch == 1.13.0, torchvision == 0.14.0, scikit-learn == 1.2.2, matplotlib == 3.7.1, numpy == 1.23.5)

- CUDA 10.1+
- python 3.7.9+
- pytorch 1.7.1+
- torchvision 0.8.2+
- scikit-learn 0.24.0+
- matplotlib 3.3.3+
- numpy 1.19.2+


### Datasets 
For CIFAR10 and CIFAR100, we provide a function to automatically download and preprocess the data, you can also download the datasets from the link, and please download it to `~/data` folder.
* [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz)
* [CIFAR100](https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz)
* [TinyImagenet](http://cs231n.stanford.edu/tiny-imagenet-200.zip)

## 2. Get started
```bash
$ cd resnet_CLIP
```
Although We have provided scripts to automatically download CIFAR10 and CIFAR100 dataset but for Tiny-Imagenet
you are supposed to download yourself utilizing the following command or using the link to download manually.
```bash
$ mkdir data
$ cd data
$ wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
$ unzip tiny-imagenet-200.zip
```
After you have all the dataset available you also need to run the extract_features.py to extract features using
CLIP for the dataset you want to use to train.
```bash
$ mkdir features
$ python extract_features.py --dataset cifar10
```

## 3. Training all the active learning strategies mentioned in our paper
* run the following command in the terminal(example).
* You have the freedom to adjust the arguments according to your interest in modifying the command, depending on the "Option" provided below.
```bash
$ python NEAT_main.py --gpu 0 --save-dir log_AL/ --weight-cent 0 --query-strategy NEAT --init-percent 8 --known-class 20 --query-batch 400 --seed 1 --model resnet18 --dataset cifar100
```
* **Option** 
* --datatset: cifar10, cifar100 and Tiny-Imagenet.
* In our experiment, we set --init-percent 8 in CIFAR100, TinyImagenet and --init-percent 1 in CIFAR10. 
* We set --query-batch 400 and --model resnet18.
* We set --known-class = 2, 20, 40 for CIFAR10, CIFAR100, and TinyImagenet respectively. And we set --seed = 1, 2, 3.



## 4. Evaluation
To evaluate the performance of NEAT, we provide a set of plotting python scripts.
* **Option** 
* --datatset: cifar10, cifar100 and Tiny-Imagenet.
* In our experiment, we set --init-percent 8 in CIFAR100, TinyImagenet and --init-percent 1 in CIFAR10. 
* We set --query-batch 400 and --model resnet18.
* We set --known-class = 2, 20, 40 for CIFAR10, CIFAR100, and TinyImagenet respectively. And we set --seed = 1, 2, 3.

```bash
$ python plot.py
```
