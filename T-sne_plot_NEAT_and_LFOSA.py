import random
import pickle
import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from matplotlib.lines import Line2D
from matplotlib.ticker import NullLocator

def load_files(feature_dir, label_dir):
    feature = torch.load(feature_dir)

    labels = torch.load(label_dir)

    return feature, labels


# %%
feature_dir = "features/clip/cifar10_features.pt"

label_dir = "features/clip/cifar10_labels.pt"

feature, labels = load_files(feature_dir, label_dir)

feature = feature.cpu().numpy()
labels = labels.cpu().numpy()



# %%
def load_index(pkl):
    with open(pkl, "rb") as input_file:
        e = pickle.load(input_file)
    return e


baseline_ = "./log_AL/temperature_resnet18_cifar10_known2_init1_batch400_seed2_AV_temperature_unknown_T0.5_known_T0.5_modelB_T1.0_per_round_query_index.pkl"

active_ = "./log_AL/temperature_resnet18_cifar10_known2_init1_batch400_seed2_active_query_unknown_T0.5_known_T0.5_modelB_T1.0_per_round_query_index.pkl"

# active_ = "./log_AL/temperature_resnet18_Tiny-Imagenet_known40_init8_batch400_seed1_active_query_unknown_T0.5_known_T0.5_modelB_T1.0_pretrained_model_clip_per_round_query_index.pkl"
# baseline_ = "./log_AL/temperature_resnet18_Tiny-Imagenet_known40_init8_batch400_seed1_AV_temperature_unknown_T0.5_known_T0.5_modelB_T1.0_pretrained_model_clip_per_round_query_index.pkl"

baseline = load_index(baseline_)

active = load_index(active_)

fig, ax = plt.subplots(figsize=(10, 10))

# for every class, we'll add a scatter plot separately


# Define colors
colors = plt.cm.tab20(np.linspace(0, 1, 21))

class_name = ['known class 1', 'known class 2', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle', 'bowl',
              'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle', 'unknown']

# perplexities = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
# early_exaggerations = [4.0, 8.0, 12.0]
# learning_rates = [100.0, 'auto', 250.0, 500.0, 750.0, 1000.0]
# n_iters = [250, 500, 750, 1000]
# query_round = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

perplexities = [15]
early_exaggerations = [4.0]
learning_rates = [100.0]
n_iters = [250]
query_round = [2]
for perplexity in perplexities:
    for early_exaggeration in early_exaggerations:
        for learning_rate in learning_rates:
            for n_iter in n_iters:
                tsne = TSNE(n_components=2,
                            perplexity=perplexity,
                            early_exaggeration=early_exaggeration,
                            learning_rate=learning_rate,
                            n_iter=n_iter).fit_transform(feature)

                for round in query_round:
                    b = baseline[round]

                    a = active[round]

                    active_index = tsne[a]
                    baseline_index = tsne[b]

                    tx = tsne[:, 0]
                    ty = tsne[:, 1]

                    fig, ax = plt.subplots(figsize=(10, 10))

                    unknown_added = False
                    for i in range(10):
                        indices = np.where(labels == i)[0]
                        if i < 2:
                            if i == 0:
                                ax.scatter(tx[indices], ty[indices], s=7, color=colors[3], alpha=0.95, label=f" {class_name[i]}")
                            if i == 1:
                                ax.scatter(tx[indices], ty[indices], s=7, color=colors[5], alpha=0.95,
                                           label=f" {class_name[i]}")
                        else:
                            if not unknown_added:
                                ax.scatter(tx[indices], ty[indices], s=1, color='gray', alpha=0.3, label=f" {class_name[20]}")
                                unknown_added = True
                            else:
                                ax.scatter(tx[indices], ty[indices], s=1, color='gray', alpha=0.3)


                    ax.scatter(baseline_index[:, 0], baseline_index[:, 1], s=75, c='b', alpha=0.9, label='LfOSA',
                               edgecolor='white')
                    ax.scatter(active_index[:, 0], active_index[:, 1], s=70, c='r', alpha=0.9, label='NEAT',
                               edgecolor='white')

                    # create custom legend handles
                    legend_elements = [Line2D([0], [0], marker='o', color='w', label='Known Class 1',
                                              markerfacecolor=colors[3], markersize=15, markeredgecolor='white'),
                                       Line2D([0], [0], marker='o', color='w', label='Known Class 2',
                                              markerfacecolor=colors[5], markersize=15, markeredgecolor='white'),
                                       Line2D([0], [0], marker='o', color='w', label='Unknown Class',
                                              markerfacecolor='gray', markersize=15, markeredgecolor='white'),
                                       Line2D([0], [0], marker='o', color='w', label='NEAT',
                                              markerfacecolor='r', markersize=15, markeredgecolor='white'),
                                       Line2D([0], [0], marker='o', color='w', label='LfOSA',
                                              markerfacecolor='b', markersize=15, markeredgecolor='white')]

                    ax.legend(handles=legend_elements, loc='best', prop={'size': 16})
                    # Remove tick marks, labels, and axis lines
                    ax.xaxis.set_major_locator(NullLocator())
                    ax.yaxis.set_major_locator(NullLocator())
                    ax.set_frame_on(False)
                    # plt.savefig(f'image/feature_plot.png', format='png', dpi=300)
                    # ax.set_title("t-SNE Visualization of CLIP Extracted CIFAR-10 Features", fontsize=20)
                    plt.show()
