import random

import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE

seed = 10
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
import csv
#
# with open('C:\\Users\\maoda\\OneDrive\\Desktop\\OSA\\数据分析\\features_for_TSNE_train_A_1.csv', 'r') as file:
#     reader = csv.reader(file)
#     csv_data = []
#     for row in reader:
#         int_row = [float(val) for val in row]  # convert the row to integers
#         csv_data.append(int_row)
# csv_data = csv_data.data.numpy()
# print(csv_data[0])
#
# tsne = TSNE(n_components=2).fit_transform(csv_data)
# # print(tsne)
import numpy as np

# Load CSV file
csv_data = np.loadtxt('C:\\Users\\maoda\\OneDrive\\Desktop\\OSA\\LfOSA\\A_features_epoch99_query_9.csv', delimiter=',')

# Convert to NumPy array
data_array = np.array(csv_data)

labels = np.loadtxt('C:\\Users\\maoda\\OneDrive\\Desktop\\OSA\\LfOSA\\A_labels_epoch99_query_9.csv', delimiter=',')

# Convert to NumPy array
labels = np.array(labels)

# Print shape of array
tsne = TSNE(n_components=2).fit_transform(csv_data)
print(tsne)


def scale_to_01_range(x):
    # compute the distribution range
    value_range = (np.max(x) - np.min(x))

    # move the distribution so that it starts from zero
    # by extracting the minimal value from all its values
    starts_from_zero = x - np.min(x)

    # make the distribution fit [0; 1] by dividing by its range
    return (starts_from_zero / value_range) * 100


# extract x and y coordinates representing the positions of the images on T-SNE plot
tx = tsne[:, 0]
ty = tsne[:, 1]

tx = scale_to_01_range(tx)
ty = scale_to_01_range(ty)

fig, ax = plt.subplots(figsize=(10, 10))

# for every class, we'll add a scatter plot separately


class_name = ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle', 'bowl',
              'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle', 'unknown']
colors = plt.cm.rainbow(np.linspace(0, 1, len(class_name)))
num_classes = 21
# area = np.ones((128,), dtype=int)
for i in range(num_classes):
    indices = np.where(labels == i)[0]
    # colors = np.array(i, dtype=np.float) / 255
    if i == 20:
        ax.scatter(tx[indices], ty[indices], c=colors[i], label=f" {class_name[i]}")
    else:

        ax.scatter(tx[indices], ty[indices], s=10 * 10, c=colors[i], alpha=0.5, label=f" {class_name[i]}")

# build a legend using the labels we set previously
ax.legend(loc='best')

# finally, show the plot
# plt.show()
plt.savefig('plot_A_query_9——new_2.png')