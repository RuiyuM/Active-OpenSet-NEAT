import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.autograd import Variable
import clip
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


class CustomCIFAR100Dataset(Dataset):
    def __init__(self, root="./data/cifar100", train=True, download=True, transform=None):
        self.cifar100_dataset = datasets.CIFAR100(root, train=train, download=download, transform=transform)

    def __getitem__(self, index):
        data_point, label = self.cifar100_dataset[index]
        return index, (data_point, label)

    def __len__(self):
        return len(self.cifar100_dataset)



def CIFAR100_EXTRACT_FEATURE_CLIP():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    crop = transforms.RandomCrop(32, padding=4)
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

    preprocess_rand = transforms.Compose([crop, transforms.RandomHorizontalFlip(), preprocess])

    # train_data = datasets.CIFAR100('./data/', train=True, download=True, transform=preprocess_rand)
    train_data = CustomCIFAR100Dataset(train=True, download=True, transform=preprocess_rand)
    record = [[] for _ in range(100)]

    batch_size = 256
    print('Data Loader')
    data_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for batch_idx, (index, (data, labels)) in enumerate(data_loader):
        data = data.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            extracted_feature = model.encode_image(data)
        for i in range(extracted_feature.shape[0]):
            record[labels[i]].append({'feature': extracted_feature[i].detach().cpu(), 'index': index[i]})


    total_len = sum([len(a) for a in record])
    origin_trans = torch.zeros(total_len, record[0][0]['feature'].shape[0])
    origin_label = torch.zeros(total_len).long()
    index_rec = np.zeros(total_len, dtype=int)
    cnt, lb = 0, 0
    for item in record:
        for i in item:
            # if i['index'] not in sel_noisy:
            origin_trans[cnt] = i['feature']
            origin_label[cnt] = lb
            index_rec[cnt] = i['index']
            cnt += 1
            # print(cnt)
        lb += 1
    data_set = {'feature': origin_trans[:cnt], 'label': origin_label[:cnt], 'index': index_rec[:cnt]}

    KINDS = 100
    all_point_cnt = data_set['feature'].shape[0]

    sample = np.random.choice(np.arange(data_set['feature'].shape[0]), all_point_cnt, replace=False)
    # final_feat, noisy_label = get_feat_clusters(data_set, sample)
    final_feat = data_set['feature'][sample]
    sel_idx = data_set['index'][sample]
    Dist = cosDistance(final_feat)
    # min_similarity = 0.0
    values, indices = Dist.topk(k=10, dim=1, largest=False, sorted=True)

    return indices, sel_idx


def cosDistance(features):
    # features: N*M matrix. N features, each features is M-dimension.
    features = F.normalize(features, dim=1)  # each feature's l2-norm should be 1
    similarity_matrix = torch.matmul(features, features.T)
    distance_matrix = 1.0 - similarity_matrix
    return distance_matrix

    # extracted_feature = model.encode_image(data)
    # features.append(extracted_feature.cpu().numpy())
    # data = Variable(data)
    # labels.append(labels.cpu().numpy())
    # indices.extend(index)

# Install necessary packages
# !pip install torch torchvision openai-clip

# # Create a custom dataset class to return the index along with the data and label
# batch_size = 256
# use_gpu = True
# num_workers = 4
# SEED = 1
#
# torch.manual_seed(SEED)
# np.random.seed(SEED)
# random.seed(SEED)
#
# # Load CIFAR-100 dataset
# transform_train = transforms.Compose([
# transforms.RandomCrop(32, padding=4),
# transforms.RandomHorizontalFlip(),
# transforms.RandomRotation(15),
# transforms.ToTensor(),
#         # This results in an image tensor with values centered
#         # around zero and scaled to a range of approximately -1 to 1.
# transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])
# pin_memory = True if use_gpu else False
# Cifar100_dataset = torchvision.datasets.CIFAR100("./data/cifar100", train=True, download=True, transform=transform_train)
# trainloader = DataLoader(dataset=Cifar100_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
# for batch_idx, (index, (data, labels)) in enumerate(trainloader):
#     x = index
#     break

# Load the CLIP model
# device = "cuda" if torch.cuda.is_available() else "cpu"
# model, preprocess = clip.load("ViT-B/32", device=device)


# Extract features from the images using the CLIP model
# x = CIFAR100_EXTRACT_FEATURE(256, True, 4, 1)
# print(x)
# mnistdata = torchvision.datasets.CIFAR100("./data/cifar100", train=True, download=True, transform=transform_train)
# print('number of image : ',len(mnistdata))
#
# batch_size = 10
# print('Data Loader')
# data_loader = torch.utils.data.DataLoader(dataset=mnistdata, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
#
# count = 0
# for batch_idx, (data, targets) in enumerate(data_loader):
#
#     count += batch_size
#     print('batch :', batch_idx + 1,'    ', count, '/', len(mnistdata),
#           'image:', data.shape, 'target : ', targets)
