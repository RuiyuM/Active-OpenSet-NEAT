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
CLIP for all dataset if you want to specify certain dataset please change the script.
```bash
$ mkdir features
$ python extract_features.py
```

## 3. Training all the active learning strategies mentioned in our paper
* run the following command in the terminal(example).
* You have the freedom to adjust the arguments according to your interest in modifying the command, depending on the "Option" provided below.
```bash
$ python NEAT_main.py --gpu 1 --k 10 --save-dir log_AL/ --query-strategy NEAT --init-percent 1 --known-class 2 --query-batch 400 --seed 2 --model resnet18 --dataset cifar10
```
* **Option** 
* --datatset: cifar10, cifar100, and Tiny-Imagenet.
* --known-class: 2, 20, and 40 for cifar10, cifar100, Tiny-Imagenet respectively in our experiments.
* --init-percent: 1, 8, 8 for cifar10, cifar100, Tiny-Imagenet respectively in our experiments.
* --query-batch 400 in our experiments
* --model: 'resnet18', 'resnet34', 'resnet50', and 'vgg16'
* --query-strategy: 'random', 'uncertainty',
                             'AV_temperature', 'NEAT_passive', 'NEAT',
                             "BGADL", "OpenMax", "Core_set", 'BADGE_sampling', "certainty", "hybrid-BGADL",
                             "hybrid-OpenMax", "hybrid-Core_set", "hybrid-BADGE_sampling", "hybrid-uncertainty"
* --workers: default 4 in our setup if you only have one gpu please set to 0.
* --max-epoch: 100 in our experiments.
* --max-query: 10 in our experiments.
* --k: 10 number of neighbors you can change this number based on your research requirements.
* --pre-type: default is clip, and you can introduce your pre-trained model base on your interest.


## 4. Evaluation
To evaluate the performance of NEAT, we provide a set of plotting python scripts.
* **Option** 
* 

```bash
$ python plot.py
```
