import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

import clip
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from collections import Counter

from torch.utils.data import SubsetRandomSampler
import pickle 
import os


import resnet_image as res_image



def set_model_min(pre_type):
    # use resnet18 (pretrained with CIFAR-10). Only for the minimum implementation of HOC
    print(f'Use model {pre_type}')
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if pre_type == 'CLIP':
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = clip.load('ViT-B/32', device, jit=False)  # RN50, RN101, RN50x4, ViT-B/32
        return model, preprocess

    else:

        if  pre_type == 'image18':
            model = res_image.resnet18(pretrained=True)
        elif pre_type == 'image34':
            model = res_image.resnet34(pretrained=True)
        elif pre_type == 'image50':
            model = res_image.resnet50(pretrained=True)
        else:
            RuntimeError('Undefined pretrained model.')
        
        for param in model.parameters():

            param.requires_grad = False
        
        model.to(device)
        
        return model, None



class CustomCIFAR10Dataset_train(Dataset):
    cifar100_dataset = None
    targets = None

    @classmethod
    def load_dataset(cls, root="./data/cifar10", train=True, download=True, transform=None):
        cls.cifar10_dataset = datasets.CIFAR10(root, train=train, download=download, transform=transform)
        cls.targets = cls.cifar10_dataset.targets
    def __init__(self):
        if CustomCIFAR10Dataset_train.cifar10_dataset is None:
            raise RuntimeError("Dataset not loaded. Call load_dataset() before creating instances of this class.")

    def __getitem__(self, index):
        data_point, label = CustomCIFAR10Dataset_train.cifar10_dataset[index]
        return index, (data_point, label)

    def __len__(self):
        return len(CustomCIFAR10Dataset_train.cifar10_dataset)



class CustomCIFAR100Dataset_train(Dataset):
    cifar100_dataset = None
    targets = None

    @classmethod
    def load_dataset(cls, root="./data/cifar100", train=True, download=True, transform=None):
        cls.cifar100_dataset = datasets.CIFAR100(root, train=train, download=download, transform=transform)
        cls.targets = cls.cifar100_dataset.targets
    def __init__(self):
        if CustomCIFAR100Dataset_train.cifar100_dataset is None:
            raise RuntimeError("Dataset not loaded. Call load_dataset() before creating instances of this class.")

    def __getitem__(self, index):
        data_point, label = CustomCIFAR100Dataset_train.cifar100_dataset[index]
        return index, (data_point, label)

    def __len__(self):
        return len(CustomCIFAR100Dataset_train.cifar100_dataset)



class CustomTinyImageNetDataset_train(Dataset):
    def __init__(self, root='./data/tiny-imagenet-200', train=True, download=True,target_train=None, transform=None, invalidList=None):
       

        with open('target_train.pkl', 'rb') as f:
            target_train = pickle.load(f)


        self.tiny_imagenet_dataset = datasets.ImageFolder(os.path.join(root, 'train' if train else 'val'),
                                                          transform=transform)
        self.targets = target_train


    def __getitem__(self, index):
        data_point, _ = self.tiny_imagenet_dataset[index]
        # data_point, _ = self.tiny_imagenet_dataset[index]
        # label = self.targets[index]
        label = self.targets[index]

        return index, (data_point, label)

    def __len__(self):
        return len(self.tiny_imagenet_dataset)



def cosDistance_two(unlabeled_features, labeled_features):
    # features: N*M matrix. N features, each features is M-dimension.
    unlabeled_features = F.normalize(unlabeled_features, dim=1)  # each feature's l2-norm should be 1

    labeled_features   = F.normalize(labeled_features, dim=1)  # each feature's l2-norm should be 1

    similarity_matrix  = torch.matmul(unlabeled_features, labeled_features.T)

    distance_matrix = 1.0 - similarity_matrix
    
    return distance_matrix



def CIFAR100_EXTRACT_ALL(pre_type, dataset, model, preprocess):
    
    model.eval()

    #################################################
    crop = transforms.RandomCrop(32, padding=4)
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    
    if pre_type == "CLIP":
        preprocess_rand = transforms.Compose([crop,
                                              transforms.RandomHorizontalFlip(),
                                              preprocess])
    elif 'image' in pre_type:
        
        preprocess_rand = transforms.Compose([crop,
                                              transforms.RandomHorizontalFlip(),
                                              transforms.Resize(224),
                                              transforms.ToTensor(),
                                              normalize])
    #################################################


    if dataset == "cifar10":

        CustomCIFAR10Dataset_train.load_dataset(transform=preprocess_rand)
        train_data = CustomCIFAR10Dataset_train()

    else:

        CustomCIFAR100Dataset_train.load_dataset(transform=preprocess_rand)
        train_data = CustomCIFAR100Dataset_train()

    batch_size = 256
    print('Data Loader')
    data_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"


    features = []

    all_labels = []

    sel_idx = []

    for batch_idx, (index, (data, labels)) in enumerate(data_loader):
        print (batch_idx)
        data = data.to(device)
        labels = labels.to(device)
        
        with torch.no_grad():

            if pre_type == "CLIP":

                extracted_feature = model.encode_image(data)

            else:

                extracted_feature = model(data)

        
            features.append(extracted_feature)

            sel_idx.append(index)
            all_labels.append(labels)


    final_feat = torch.concat(features).to(device)
    sel_idx    = torch.concat(sel_idx).to(device)
    labels     = torch.concat(all_labels).to(device)

    ###########################################################
    feature_sel_idx = sel_idx.unsqueeze(1).repeat(1, final_feat.size()[1]).to(device)

    ordered_feature = torch.gather(final_feat, 0, feature_sel_idx)

    ordered_label   = torch.gather(labels, 0, sel_idx)

    ###########################################################

    index_to_label = {}
    for idx, index in enumerate(sel_idx):

        index_to_label[index] = labels[idx]




    save_folder = "./features/" +  pre_type

    isExist = os.path.exists(save_folder)
    if not isExist:

       # Create a new directory because it does not exist
       os.makedirs(save_folder) 



    if dataset == "cifar10":

        torch.save(ordered_feature, save_folder + '/cifar10_features.pt')
        torch.save(ordered_label,   save_folder + '/cifar10_labels.pt')
        torch.save(index_to_label,  save_folder + '/cifar10_index_to_label.pt')
    
    else:
    
        torch.save(ordered_feature, save_folder + '/cifar100_features.pt')
        torch.save(ordered_label,   save_folder + '/cifar100_labels.pt')
        torch.save(index_to_label,  save_folder + '/cifar100_index_to_label.pt')




def ImageNet_EXTRACT_ALL(pre_type, model, preprocess):
    
    model.eval()

    crop = transforms.RandomCrop(64, padding=4, padding_mode='reflect')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


    if pre_type == "CLIP":
        preprocess_rand = transforms.Compose([crop,
                                              transforms.RandomHorizontalFlip(),
                                              preprocess])
    elif 'image' in pre_type:

        preprocess_rand = transforms.Compose([crop,
                                              transforms.RandomHorizontalFlip(),
                                              transforms.Resize(224),
                                              transforms.ToTensor(),
                                              normalize])


    train_data = CustomTinyImageNetDataset_train(transform=preprocess_rand)


    batch_size = 256
    print('Data Loader')
    data_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"


    features = []

    all_labels = []

    sel_idx = []

    for batch_idx, (index, (data, labels)) in enumerate(data_loader):
        print (batch_idx)
        data = data.to(device)
        labels = labels.to(device)
        
        with torch.no_grad():

            if pre_type == "CLIP":

                extracted_feature = model.encode_image(data)

            else:

                extracted_feature = model(data)
        
            features.append(extracted_feature)

            sel_idx.append(index)
            all_labels.append(labels)


    final_feat = torch.concat(features).to(device)
    sel_idx    = torch.concat(sel_idx).to(device)
    labels     = torch.concat(all_labels).to(device)

    ###########################################################
    feature_sel_idx = sel_idx.unsqueeze(1).repeat(1, final_feat.size()[1]).to(device)

    ordered_feature = torch.gather(final_feat, 0, feature_sel_idx)

    ordered_label   = torch.gather(labels, 0, sel_idx)

    ###########################################################

    index_to_label = {}
    for idx, index in enumerate(sel_idx):

        index_to_label[index] = labels[idx]


    save_folder = "./features/" +  pre_type

    isExist = os.path.exists(save_folder)
    if not isExist:

       # Create a new directory because it does not exist
       os.makedirs(save_folder) 



    torch.save(ordered_feature, save_folder + '/Tiny-Imagenet_features.pt')
    torch.save(ordered_label,   save_folder +  '/Tiny-Imagenet_labels.pt')
    torch.save(index_to_label,  save_folder + '/Tiny-Imagenet_index_to_label.pt')



def extracted_feature(pre_type, dataset):


    model, preprocess = set_model_min(pre_type)


    if dataset == "Tiny-Imagenet":

        ImageNet_EXTRACT_ALL(pre_type, model, preprocess)

    else:

        CIFAR100_EXTRACT_ALL(pre_type, dataset, model, preprocess)



def CIFAR100_LOAD_ALL(dataset="cifar100", pre_type="clip"):

    device = "cuda" if torch.cuda.is_available() else "cpu"


    save_folder = "./features/" +  pre_type


    if dataset == "cifar10":
        
        index_to_label = torch.load(save_folder + '/cifar10_index_to_label.pt')
        ordered_feature = torch.load(save_folder + '/cifar10_features.pt')
        ordered_label = torch.load(save_folder + '/cifar10_labels.pt')

    elif dataset == "Tiny-Imagenet":

        index_to_label = torch.load(save_folder + '/Tiny-Imagenet_index_to_label.pt')
        ordered_feature = torch.load(save_folder + '/Tiny-Imagenet_features.pt')
        ordered_label = torch.load(save_folder + '/Tiny-Imagenet_labels.pt')

    elif dataset == "cifar100":

        index_to_label = torch.load(save_folder + '/cifar100_index_to_label.pt')
        ordered_feature = torch.load(save_folder + '/cifar100_features.pt')
        ordered_label = torch.load(save_folder + './cifar100_labels.pt')



    print ("finish loading " + dataset )

    new_dict = {}
    for k, v in index_to_label.items():
        new_dict[k.item()] = v.item()

    return ordered_feature, ordered_label, new_dict


def get_features(ordered_feature, ordered_label, indices):

    features = ordered_feature[indices, :]

    labels = ordered_label[indices]

    return features, labels

def CIFAR100_EXTRACT_FEATURE_CLIP_new(labeled_index, unlabeled_index, args, ordered_feature, ordered_label):

    ###########################################################

    labeled_final_feat, labled_labels     = get_features(ordered_feature, ordered_label, labeled_index)

    unlabeled_final_feat, unlabled_labels = get_features(ordered_feature, ordered_label, unlabeled_index)
    ###########################################################

    order_to_index = {}

    index_to_order = {}
    for i in range(len(labeled_index)):

        order_to_index[i] = labeled_index[i]

        index_to_order[labeled_index[i]] = i

    ###################################################

    Dist = cosDistance_two(unlabeled_final_feat, labeled_final_feat)


    values, indices = torch.topk(Dist, k=args.k, dim=1, largest=False, sorted=True)

    for k in range(indices.size()[0]):
        for j in range(indices.size()[1]):

            indices[k][j] = order_to_index[indices[k][j].item()]

    ###################################################

    index_knn = {}

    for i in range(len(unlabeled_index)):

        index_knn[unlabeled_index[i]] = ( indices[i, :].cpu().numpy(), values[i, :] )


    return index_knn


if __name__ == "__main__":

    #dataset = "Tiny-Imagenet"

    #CIFAR100_EXTRACT_ALL(dataset=dataset)

    for dataset in ["cifar10", "cifar100", "Tiny-Imagenet"]:
        print (dataset)
        for pre_type in ["image18", "image34", "image50"]:


            extracted_feature(pre_type, dataset)