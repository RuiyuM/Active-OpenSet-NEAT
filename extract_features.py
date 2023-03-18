import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.autograd import Variable
import clip

def CIFAR100_EXTRACT_FEATURE_CLIP():

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    crop = transforms.RandomCrop(32, padding=4)
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

    preprocess_rand = transforms.Compose([crop, transforms.RandomHorizontalFlip(), preprocess])

    train_data = datasets.CIFAR100('./data/', train=True, download=True, transform=preprocess_rand)
    record = [[] for _ in range(100)]

    batch_size = 10
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
    return record


x = CIFAR100_EXTRACT_FEATURE_CLIP()
print(x)

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