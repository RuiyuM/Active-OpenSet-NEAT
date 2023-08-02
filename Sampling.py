import numpy as np
from sklearn.mixture import GaussianMixture
import torch
from scipy.spatial.distance import euclidean
import torch.nn.functional as F
from collections import Counter
import pdb
from scipy import stats
from extract_features import CIFAR100_EXTRACT_FEATURE_CLIP_new
from copy import deepcopy
from torch.distributions import Categorical
import datasets
from torch.nn.functional import softmax
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from scipy.spatial import distance
from scipy.optimize import minimize
# OpenMax calculation
def compute_openmax_scores(activations, mavs, weibull_models, labels, known_class):
    openmax_scores = []
    for activation, label in zip(activations, labels):
        if label < known_class:  # For known classes, compute openmax score
            clipped_activation = np.clip(activation - mavs[label], a_min=0, a_max=None)
            params = weibull_models[label]["params"]
            w_score = 1 - np.exp(-(clipped_activation ** params[1]))
            score = w_score / (w_score + (1 - w_score + 1e-7) / known_class)
        else:  # For unknown classes, score is 0
            score = 0
        openmax_scores.append(np.max(score))
    return openmax_scores



# Weibull CDF calculation
def weibull_cdf(x, params):
    return 1 - np.exp(-((x/params[0])**params[1]))

def new_open_max(args, unlabeledloader, trainloader_B, Len_labeled_ind_train, Len_unlabeled_ind_train, model, use_gpu):
    model.eval()
    # Setup
    classes = tuple(range(args.known_class))  # Known classes
    features_dict = {c: [] for c in classes}

    for batch_idx, (index, (data, label)) in enumerate(trainloader_B):
        if use_gpu:
            data, label = data.cuda(), label.cuda()
        with torch.no_grad():
            batch_features, outputs = model(data)
        for c in classes:
            mask = (label == c)
            if mask.any():
                features_c = batch_features[mask].detach().cpu().numpy()
                features_dict[c].extend(features_c)

    # Calculate MAVs
    mavs = {c: np.mean(features, axis=0) for c, features in features_dict.items()}

    # Weibull fitting parameters setup
    weibull_models = {c: {"distances": [], "params": []} for c in classes}

    # Calculate MAVs
    for c in classes:
        mavs[c] = np.mean(features_dict[c], axis=0)



    # Calculating distances from MAV and fitting Weibull distribution
    for c in classes:
        distances = [distance.euclidean(f, mavs[c]) for f in features_dict[c]]
        weibull_models[c]["distances"] = distances

        # Normalize distances using mean and std, store normalization constants
        mean_distance = np.mean(distances)
        std_distance = np.std(distances)
        distances_normalized = (distances - mean_distance) / std_distance

        weibull_models[c]["mean_distance"] = mean_distance
        weibull_models[c]["inv_std_distance"] = 1 / std_distance

        # Weibull fitting using MLE
        def weibull_pdf(x, shape, scale):
            return (shape / scale) * (x / scale) ** (shape - 1) * np.exp(- (x / scale) ** shape)

        def neg_log_likelihood(params):
            return -np.sum(np.log(weibull_pdf(distances_normalized, *params)))

        initial_guess = [1, 1]
        bounds = [(0.1, None), (0.1, None)]
        result = minimize(neg_log_likelihood, initial_guess, bounds=bounds)

        weibull_models[c]["params"] = result.x


    already_selected = []
    n_obs = Len_unlabeled_ind_train
    features = []
    indices = []
    labels = []
    openmax_scores = []
    print(mavs[0].shape)
    # Extract features
    for batch_idx, (index, (data, label)) in enumerate(unlabeledloader):
        if use_gpu:
            data, label = data.cuda(), label.cuda()
        with torch.no_grad():
            batch_features, outputs = model(data)
        features.extend(batch_features.cpu().numpy())
        indices.extend(index)
        labels.extend(label.cpu().numpy())
        print(batch_idx)
        # Compute openmax scores
        openmax_scores.extend(compute_openmax_scores(batch_features.cpu().numpy(), mavs, weibull_models, labels, args.known_class))

    features = np.array(features)

    # Pair each score with its corresponding index in the original dataset
    score_index_pairs = list(zip(openmax_scores, indices))

    # Sort the pairs in descending order of scores
    sorted_pairs = sorted(score_index_pairs, key=lambda x: -x[0])

    # Separate the sorted scores and their corresponding indices
    sorted_scores, sorted_indices = zip(*sorted_pairs)

    # Select batch
    new_batch = list(sorted_indices[:args.query_batch])

    # Get labels for the selected batch
    query_labels = np.array(labels)[new_batch]

    precision = len(np.where(query_labels < args.known_class)[0]) / len(query_labels)
    recall = (len(np.where(query_labels < args.known_class)[0]) + Len_labeled_ind_train) / (
            len(np.where(np.array(labels) < args.known_class)[0]) + Len_labeled_ind_train)

    # Separate the selected indices into two lists based on the label
    selected_indices = np.array(indices)[new_batch]
    selected_known = selected_indices[np.where(query_labels < args.known_class)[0]]
    selected_unknown = selected_indices[np.where(query_labels >= args.known_class)[0]]

    return selected_known, selected_unknown, precision, recall

def new_core_set(args, unlabeledloader, Len_labeled_ind_train, Len_unlabeled_ind_train, model, use_gpu):
    model.eval()
    min_distances = None
    already_selected = []
    n_obs = Len_unlabeled_ind_train
    features = []
    indices = []
    labels = []

    # Extract features
    for batch_idx, (index, (data, label)) in enumerate(unlabeledloader):
        if use_gpu:
            data, label = data.cuda(), label.cuda()
        with torch.no_grad():
            batch_features, outputs = model(data)
        features.extend(batch_features.cpu().numpy())
        indices.extend(index)
        labels.extend(label.cpu().numpy())

    features = np.array(features)

    # Select batch
    new_batch = []
    for _ in range(args.query_batch):
        if not already_selected:
            # Initialize centers with a randomly selected datapoint
            ind = np.random.choice(np.arange(n_obs))
        else:
            ind = np.argmax(min_distances)
            assert ind not in already_selected

        # Update min_distances for all examples given new cluster center.
        dist = pairwise_distances(features, features[ind].reshape(1, -1))
        if min_distances is None:
            min_distances = dist
        else:
            min_distances = np.minimum(min_distances, dist)

        new_batch.append(ind)
        already_selected.append(ind)

    # Get labels for the selected batch
    query_labels = np.array(labels)[new_batch]

    # Calculate precision and recall
    precision = len(np.where(query_labels < args.known_class)[0]) / len(query_labels)
    recall = (len(np.where(query_labels < args.known_class)[0]) + Len_labeled_ind_train) / (
            len(np.where(np.array(labels) < args.known_class)[0]) + Len_labeled_ind_train)
    
    # Separate the selected indices into two lists based on the label
    selected_indices = np.array(indices)[new_batch]
    selected_known = selected_indices[np.where(query_labels < args.known_class)[0]]
    selected_unknown = selected_indices[np.where(query_labels >= args.known_class)[0]]

    return selected_known, selected_unknown, precision, recall


def random_sampling(args, unlabeledloader, Len_labeled_ind_train, model, use_gpu):
    model.eval()
    queryIndex = []
    labelArr = []
    precision, recall = 0, 0
    for batch_idx, (index, (_, labels)) in enumerate(unlabeledloader):
        queryIndex += index
        labelArr += list(np.array(labels.data))

    tmp_data = np.vstack((queryIndex, labelArr)).T
    np.random.shuffle(tmp_data)
    tmp_data = tmp_data.T
    queryIndex = tmp_data[0][:args.query_batch]
    labelArr = tmp_data[1]
    queryLabelArr = tmp_data[1][:args.query_batch]
    precision = len(np.where(queryLabelArr < args.known_class)[0]) / len(queryLabelArr)
    recall = (len(np.where(queryLabelArr < args.known_class)[0]) + Len_labeled_ind_train) / (
            len(np.where(labelArr < args.known_class)[0]) + Len_labeled_ind_train)
    return queryIndex[np.where(queryLabelArr < args.known_class)[0]], queryIndex[
        np.where(queryLabelArr >= args.known_class)[0]], precision, recall


def uncertainty_sampling_hybrid(args, unlabeledloader, Len_labeled_ind_train, model, use_gpu, labelArr_true):
    model.eval()
    queryIndex = []
    labelArr = []
    uncertaintyArr = []
    precision, recall = 0, 0
    for batch_idx, (index, (data, labels)) in enumerate(unlabeledloader):
        if use_gpu:
            data, labels = data.cuda(), labels.cuda()
        if args.dataset == 'mnist':
            data = data.repeat(1, 3, 1, 1)
        features, outputs = model(data)

        uncertaintyArr += list(
            np.array((-torch.softmax(outputs, 1) * torch.log(torch.softmax(outputs, 1))).sum(1).cpu().data))
        queryIndex += index
        labelArr += list(np.array(labels.cpu().data))

    tmp_data = np.vstack((uncertaintyArr, queryIndex, labelArr)).T
    tmp_data = tmp_data[np.argsort(tmp_data[:, 0])]
    tmp_data = tmp_data.T
    queryIndex = tmp_data[1][-args.query_batch:].astype(int)
    labelArr = tmp_data[2].astype(int)
    labelArr_true = np.array(labelArr_true)
    queryLabelArr = tmp_data[2][-args.query_batch:]

    precision = len(np.where(queryLabelArr < args.known_class)[0]) / args.query_batch

    recall = (len(np.where(queryLabelArr < args.known_class)[0]) + Len_labeled_ind_train) / (
            len(np.where(labelArr_true < args.known_class)[0]) + Len_labeled_ind_train)

    return queryIndex[np.where(queryLabelArr < args.known_class)[0]], queryIndex[
        np.where(queryLabelArr >= args.known_class)[0]], precision, recall

def uncertainty_sampling(args, unlabeledloader, Len_labeled_ind_train, model, use_gpu):
    model.eval()
    queryIndex = []
    labelArr = []
    uncertaintyArr = []
    precision, recall = 0, 0
    for batch_idx, (index, (data, labels)) in enumerate(unlabeledloader):
        if use_gpu:
            data, labels = data.cuda(), labels.cuda()
        if args.dataset == 'mnist':
            data = data.repeat(1, 3, 1, 1)
        features, outputs = model(data)

        uncertaintyArr += list(
            np.array((-torch.softmax(outputs, 1) * torch.log(torch.softmax(outputs, 1))).sum(1).cpu().data))
        queryIndex += index
        labelArr += list(np.array(labels.cpu().data))

    tmp_data = np.vstack((uncertaintyArr, queryIndex, labelArr)).T
    tmp_data = tmp_data[np.argsort(tmp_data[:, 0])]
    tmp_data = tmp_data.T
    queryIndex = tmp_data[1][-args.query_batch:].astype(int)
    labelArr = tmp_data[2].astype(int)
    queryLabelArr = tmp_data[2][-args.query_batch:]
    precision = len(np.where(queryLabelArr < args.known_class)[0]) / len(queryLabelArr)
    recall = (len(np.where(queryLabelArr < args.known_class)[0]) + Len_labeled_ind_train) / (
            len(np.where(labelArr < args.known_class)[0]) + Len_labeled_ind_train)
    return queryIndex[np.where(queryLabelArr < args.known_class)[0]], queryIndex[
        np.where(queryLabelArr >= args.known_class)[0]], precision, recall

# unlabeledloader is int 800
def AV_sampling_temperature(args, unlabeledloader, Len_labeled_ind_train, model, use_gpu):
    model.eval()
    queryIndex = []
    labelArr = []
    uncertaintyArr = []
    S_ij = {}
    for batch_idx, (index, (data, labels)) in enumerate(unlabeledloader):
        if use_gpu:
            data, labels = data.cuda(), labels.cuda()
        _, outputs = model(data)
        # 当前的index 128 个 进入queryIndex array
        queryIndex += index
        # my_test_for_outputs = outputs.cpu().data.numpy()
        # print(my_test_for_outputs)
        # 这句code的意思就是把GPU上的数据转移到CPU上面然后再把数据类型从tensor转变为python的数据类型
        labelArr += list(np.array(labels.cpu().data))
        # activation value based
        # 这个function会return 128行然后每行21列的数据，return分两个部分，一个部分是tensor的数据类型然后是每行最大的数据
        # 另一个return的东西也是tensor的数据类型然后是每行的最大的值具体在这一行的具体位置
        v_ij, predicted = outputs.max(1)
        for i in range(len(predicted.data)):
            tmp_class = np.array(predicted.data.cpu())[i]
            tmp_index = index[i]
            tmp_label = np.array(labels.data.cpu())[i]
            tmp_value = np.array(v_ij.data.cpu())[i]
            if tmp_class not in S_ij:
                S_ij[tmp_class] = []
            S_ij[tmp_class].append([tmp_value, tmp_index, tmp_label])
    # 上半部分的code就是把Resnet里面的输出做了一下简单的数据处理，把21长度的数据取最大值然后把这个值和其在数据集里面的index，label组成一个字典的value放到S——ij里面

    # fit a two-component GMM for each class
    tmp_data = []
    for tmp_class in S_ij:
        S_ij[tmp_class] = np.array(S_ij[tmp_class])
        activation_value = S_ij[tmp_class][:, 0]
        if len(activation_value) < 2:
            continue
        gmm = GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4)
        gmm.fit(np.array(activation_value).reshape(-1, 1))
        prob = gmm.predict_proba(np.array(activation_value).reshape(-1, 1))
        # 得到为known类别的概率
        prob = prob[:, gmm.means_.argmax()]
        # 如果为unknown类别直接为0
        if tmp_class == args.known_class:
            prob = [0] * len(prob)
            prob = np.array(prob)

        if len(tmp_data) == 0:
            # np.hstack 就是说把stack水平堆起来
            tmp_data = np.hstack((prob.reshape(-1, 1), S_ij[tmp_class]))
        else:
            # np。vstack 就是把stack竖直堆起来
            tmp_data = np.vstack((tmp_data, np.hstack((prob.reshape(-1, 1), S_ij[tmp_class]))))

    tmp_data = tmp_data[np.argsort(tmp_data[:, 0])]
    tmp_data = tmp_data.T

    # 取前1500个index
    queryIndex = tmp_data[2][-args.query_batch:].astype(int)
    labelArr = tmp_data[3].astype(int)

    queryLabelArr = tmp_data[3][-args.query_batch:]
    precision = len(np.where(queryLabelArr < args.known_class)[0]) / len(queryLabelArr)
    recall = (len(np.where(queryLabelArr < args.known_class)[0]) + Len_labeled_ind_train) / (
            len(np.where(labelArr < args.known_class)[0]) + Len_labeled_ind_train)
    return queryIndex[np.where(queryLabelArr < args.known_class)[0]], queryIndex[
        np.where(queryLabelArr >= args.known_class)[0]], precision, recall



def active_learning(index_knn, queryIndex, S_index):
    print("active learning")
    # S_index[n_index][0][1]

    for i in range(len(queryIndex)):

        neighbors_prediction = []

        # all the indices for neighbors
        neighbors = index_knn[queryIndex[i][0]]

        change = 0.0
        cur_predict = None

        for j in range(1, len(neighbors)):

            # current knn prediction
            k_predict = knn_prediction(neighbors[:j], S_index)

            if not cur_predict:
                cur_predict = k_predict

            else:
                if cur_predict != k_predict:
                    change += 1

                cur_predict = k_predict

        # how many changes
        score = change / len(neighbors)

        queryIndex[i][1] = np.append(queryIndex[i][1], score)

    return queryIndex


def knn_prediction(k_neighbors, S_index):
    neighbor_labels = []

    for k in k_neighbors:
        label = S_index[k][1][2]

        neighbor_labels.append(label)

    x = Counter(neighbor_labels)
    # print (x)
    top_1 = x.most_common(1)[0][0]

    return top_1


def active_learning_3(args, query, index_knn, queryIndex, S_index, labeled_index_to_label):
    print("active learning")
    # S_index[n_index][0][1]

    for i in range(len(queryIndex)):

        # all the indices for neighbors
        neighbors, values = index_knn[queryIndex[i][0]]

        if query == 0:

            score = -torch.mean(values)

        else:

            knn_labels_cnt = torch.zeros(args.known_class).cuda()

            for idx, neighbor in enumerate(neighbors):
                neighbor_labels = labeled_index_to_label[neighbor]

                test_variable_1 = 1.0 - values[idx]

                knn_labels_cnt[neighbor_labels] += test_variable_1

            knn_labels_prob = F.softmax(knn_labels_cnt, dim=0)

            score = Categorical(probs=knn_labels_prob).entropy()

        queryIndex[i].append(score.item())

    return queryIndex


def active_learning_2(args, index_knn, queryIndex, S_index, labeled_index_to_label):
    print("active learning")
    # S_index[n_index][0][1]

    for i in range(len(queryIndex)):

        neighbor_labels = []

        # all the indices for neighbors
        neighbors = index_knn[queryIndex[i][0]][0]

        for neighbor in neighbors:
            neighbor_labels.append(labeled_index_to_label[neighbor])

        x = Counter(neighbor_labels)

        top = x.most_common(2)

        if len(top) == 2:

            top_1 = x.most_common(2)[0][1]
            top_2 = x.most_common(2)[1][1]

            # how many changes
            score = - (top_1 + 0.0) / (top_2 + 0.0)

        else:

            score = -10

        queryIndex[i].append(score)

    return queryIndex


def active_learning_5(args, query, index_knn, queryIndex, S_index, labeled_index_to_label):
    print("active learning 5")
    # S_index[n_index][0][1]

    new_query_index = []

    for i in range(len(queryIndex)):

        # all the indices for neighbors
        neighbors, values = index_knn[queryIndex[i][0]]

        predicted_prob = F.softmax(S_index[queryIndex[i][0]][-1], dim=-1).cuda()

        predicted_label = S_index[queryIndex[i][0]][-3]

        knn_labels_cnt = torch.zeros(args.known_class).cuda()

        for idx, neighbor in enumerate(neighbors):

            neighbor_labels = labeled_index_to_label[neighbor]

            test_variable_1 = 1.0 - values[idx]

            if neighbor_labels < args.known_class:
                knn_labels_cnt[neighbor_labels] += 1.0

        score = F.cross_entropy(knn_labels_cnt.unsqueeze(0), predicted_prob.unsqueeze(0), reduction='mean')

        score_np = score.cpu().item()

        # entropy = Categorical(probs = predicted_prob ).entropy().cpu().item()

        new_query_index.append(queryIndex[i] + [score_np])

    new_query_index = sorted(new_query_index, key=lambda x: x[-1], reverse=True)

    return new_query_index


# unlabeledloader is int 800
def test_query_2(args, model, query, unlabeledloader, Len_labeled_ind_train, use_gpu, labeled_ind_train, invalidList,
                 unlabeled_ind_train, ordered_feature, ordered_label, labeled_index_to_label):
    index_knn = CIFAR100_EXTRACT_FEATURE_CLIP_new(labeled_ind_train + invalidList, unlabeled_ind_train, args,
                                                  ordered_feature, ordered_label)

    labelArr = []

    model.eval()
    #################################################################
    S_index = {}

    for batch_idx, (index, (data, labels)) in enumerate(unlabeledloader):

        if use_gpu:
            data, labels = data.cuda(), labels.cuda()

        _, outputs = model(data)

        v_ij, predicted = outputs.max(1)

        labelArr += list(np.array(labels.cpu().data))

        for i in range(len(data.data)):
            predict_class = predicted[i].detach()

            predict_value = np.array(v_ij.data.cpu())[i]

            predict_prob = outputs[i, :]

            tmp_index = index[i].item()

            true_label = np.array(labels.data.cpu())[i]

            S_index[tmp_index] = [true_label, predict_class, predict_value, predict_prob.detach().cpu()]

    #################################################################

    # 上半部分的code就是把Resnet里面的输出做了一下简单的数据处理，把21长度的数据取最大值然后把这个值和其在数据集里面的index，label组成一个字典的value放到S——ij里面

    # queryIndex 存放known class的地方
    queryIndex = []

    neighbor_unknown = {}

    detected_unknown = 0.0
    detected_known = 0.0

    for current_index in S_index:

        index_Neighbor, values = index_knn[current_index]

        true_label = S_index[current_index][0]

        count_known = 0.0
        count_unknown = 0.0

        for k in range(len(index_Neighbor)):

            n_index = index_Neighbor[k]

            if n_index in set(labeled_ind_train):
                count_known += 1

            elif n_index in set(invalidList):
                count_unknown += 1

        if count_unknown < count_known:

            queryIndex.append([current_index, count_known, true_label])

        else:
            detected_unknown += 1

    print("detected_unknown: ", detected_unknown)
    print("\n")

    queryIndex = sorted(queryIndex, key=lambda x: x[-2], reverse=True)

    print(queryIndex[:20])

    final_chosen_index = []
    invalid_index = []

    for item in queryIndex[:args.query_batch]:

        num = item[0]

        num3 = item[-1]

        if num3 < args.known_class:

            final_chosen_index.append(int(num))

        elif num3 >= args.known_class:

            invalid_index.append(int(num))

    #################################################################

    precision = len(final_chosen_index) / args.query_batch

    # recall = (len(final_chosen_index) + Len_labeled_ind_train) / (
    #        len([x for x in labelArr if args.known_class]) + Len_labeled_ind_train)

    recall = (len(final_chosen_index) + Len_labeled_ind_train) / (

            len(np.where(np.array(labelArr) < args.known_class)[0]) + Len_labeled_ind_train)

    return final_chosen_index, invalid_index, precision, recall


# unlabeledloader is int 800
def active_query(args, model, query, unlabeledloader, Len_labeled_ind_train, use_gpu, labeled_ind_train, invalidList,
                 unlabeled_ind_train, ordered_feature, ordered_label, labeled_index_to_label):
    index_knn = CIFAR100_EXTRACT_FEATURE_CLIP_new(labeled_ind_train + invalidList, unlabeled_ind_train, args,
                                                  ordered_feature, ordered_label)

    labelArr = []

    model.eval()
    #################################################################
    S_index = {}

    for batch_idx, (index, (data, labels)) in enumerate(unlabeledloader):

        if use_gpu:
            data, labels = data.cuda(), labels.cuda()

        _, outputs = model(data)

        v_ij, predicted = outputs.max(1)

        labelArr += list(np.array(labels.cpu().data))

        for i in range(len(data.data)):
            predict_class = predicted[i].detach()

            predict_value = np.array(v_ij.data.cpu())[i]

            predict_prob = outputs[i, :]

            tmp_index = index[i].item()

            true_label = np.array(labels.data.cpu())[i]

            S_index[tmp_index] = [true_label, predict_class, predict_value, predict_prob.detach().cpu()]

    #################################################################

    # 上半部分的code就是把Resnet里面的输出做了一下简单的数据处理，把21长度的数据取最大值然后把这个值和其在数据集里面的index，label组成一个字典的value放到S——ij里面

    # queryIndex 存放known class的地方
    queryIndex = []

    neighbor_unknown = {}

    detected_unknown = 0.0
    detected_known = 0.0

    for current_index in S_index:

        index_Neighbor, values = index_knn[current_index]

        true_label = S_index[current_index][0]

        count_known = 0.0
        count_unknown = 0.0

        for k in range(len(index_Neighbor)):

            n_index = index_Neighbor[k]

            if n_index in set(labeled_ind_train):
                count_known += 1

            elif n_index in set(invalidList):
                count_unknown += 1

        if count_unknown < count_known:

            queryIndex.append([current_index, count_known, true_label])

        else:
            detected_unknown += 1

    print("detected_unknown: ", detected_unknown)
    print("\n")

    queryIndex = sorted(queryIndex, key=lambda x: x[-2], reverse=True)

    #################################################################

    queryIndex = queryIndex[:2 * args.query_batch]

    #################################################################

    # if args.active_5 or args.active_5_reverse:

    queryIndex = active_learning_5(args, query, index_knn, queryIndex, S_index, labeled_index_to_label)

    # elif args.active_4:

    # queryIndex = active_learning_4(args, query, index_knn, queryIndex, S_index, labeled_index_to_label)

    #################################################################

    print(queryIndex[:20])

    final_chosen_index = []
    invalid_index = []

    for item in queryIndex[:args.query_batch]:

        num = item[0]

        num3 = item[-2]

        if num3 < args.known_class:

            final_chosen_index.append(int(num))

        elif num3 >= args.known_class:

            invalid_index.append(int(num))

    #################################################################

    precision = len(final_chosen_index) / args.query_batch

    # recall = (len(final_chosen_index) + Len_labeled_ind_train) / (
    #        len([x for x in labelArr if args.known_class]) + Len_labeled_ind_train)

    recall = (len(final_chosen_index) + Len_labeled_ind_train) / (

            len(np.where(np.array(labelArr) < args.known_class)[0]) + Len_labeled_ind_train)

    return final_chosen_index, invalid_index, precision, recall


def init_centers(X, K):
    embs = torch.Tensor(X)
    ind = torch.argmax(torch.norm(embs, 2, 1)).item()
    embs = embs.cuda()
    mu = [embs[ind]]
    indsAll = [ind]
    centInds = [0.] * len(embs)
    cent = 0
    # print('#Samps\tTotal Distance')
    while len(mu) < K:
        if len(mu) == 1:
            D2 = torch.cdist(mu[-1].view(1, -1), embs, 2)[0].cpu().numpy()
        else:
            newD = torch.cdist(mu[-1].view(1, -1), embs, 2)[0].cpu().numpy()
            for i in range(len(embs)):
                if D2[i] > newD[i]:
                    centInds[i] = cent
                    D2[i] = newD[i]
        print(str(len(mu)) + '\t' + str(sum(D2)), flush=True)
        if sum(D2) == 0.0: pdb.set_trace()
        D2 = D2.ravel().astype(float)
        Ddist = (D2 ** 2) / sum(D2 ** 2)
        customDist = stats.rv_discrete(name='custm', values=(np.arange(len(D2)), Ddist))
        ind = customDist.rvs(size=1)[0]
        while ind in indsAll: ind = customDist.rvs(size=1)[0]
        mu.append(embs[ind])
        indsAll.append(ind)
        cent += 1
    return indsAll

def badge_sampling(args, unlabeledloader, Len_labeled_ind_train,len_unlabeled_ind_train,labeled_ind_train,
                                                                            invalidList, model, use_gpu):
    model.eval()
    embDim = 512
    nLab = args.known_class
    embedding = np.zeros([len_unlabeled_ind_train + Len_labeled_ind_train + len(invalidList), embDim * nLab])

    queryIndex = []
    data_image = []
    labelArr = []
    uncertaintyArr = []
    S_ij = {}
    for batch_idx, (index, (data, labels)) in enumerate(unlabeledloader):
        if use_gpu:
            data, labels = data.cuda(), labels.cuda()
        # output = cout
        out, outputs = model(data)
        out = out.data.cpu().numpy()
        batchProbs = F.softmax(outputs, dim=1).data.cpu().numpy()
        maxInds = np.argmax(batchProbs, 1)
        for j in range(len(labels)):
            for c in range(nLab):
                if c == maxInds[j]:
                    embedding[index[j]][embDim * c: embDim * (c + 1)] = deepcopy(out[j]) * (1 - batchProbs[j][c])
                else:
                    embedding[index[j]][embDim * c: embDim * (c + 1)] = deepcopy(out[j]) * (-1 * batchProbs[j][c])
        # 当前的index 128 个 进入queryIndex array
        data_image += data
        queryIndex += index
        # my_test_for_outputs = outputs.cpu().data.numpy()
        # print(my_test_for_outputs)
        # 这句code的意思就是把GPU上的数据转移到CPU上面然后再把数据类型从tensor转变为python的数据类型
        labelArr += list(np.array(labels.cpu().data))
        # activation value based
        # 这个function会return 128行然后每行21列的数据，return分两个部分，一个部分是tensor的数据类型然后是每行最大的数据
        # 另一个return的东西也是tensor的数据类型然后是每行的最大的值具体在这一行的具体位置
        v_ij, predicted = outputs.max(1)
        for i in range(len(predicted.data)):
            tmp_class = np.array(predicted.data.cpu())[i]
            tmp_index = index[i].item()
            tmp_label = np.array(labels.data.cpu())[i]
            tmp_value = np.array(v_ij.data.cpu())[i]

            if tmp_index not in S_ij:
                S_ij[tmp_index] = []
            S_ij[tmp_index].append([tmp_class, tmp_value, tmp_label])

    embedding = torch.Tensor(embedding)
    chosen = init_centers(embedding, args.query_batch)
    queryIndex = chosen
    queryLabelArr = []
    # Assuming labeled_ind_train, invalidList and queryIndex are defined and are lists

    # Merging the lists labeled_ind_train and invalidList into a single list
    elements_to_remove = labeled_ind_train + invalidList

    # Using list comprehension to remove elements in queryIndex which are also found in elements_to_remove
    queryIndex = [element for element in queryIndex if element not in elements_to_remove]
    queryIndex = np.array(queryIndex)
    # Now, queryIndex contains only the elements not found in either labeled_ind_train or invalidList

    for i in range(len(queryIndex)):
        queryLabelArr.append(S_ij[queryIndex[i]][0][2])

    queryLabelArr = np.array(queryLabelArr)
    labelArr = np.array(labelArr)
    precision = len(np.where(queryLabelArr < args.known_class)[0]) / len(queryLabelArr)
    recall = (len(np.where(queryLabelArr < args.known_class)[0]) + Len_labeled_ind_train) / (
            len(np.where(labelArr < args.known_class)[0]) + Len_labeled_ind_train)
    return queryIndex[np.where(queryLabelArr < args.known_class)[0]], queryIndex[
        np.where(queryLabelArr >= args.known_class)[0]], precision, recall

def badge_sampling_hybrid(args, unlabeledloader, Len_labeled_ind_train, len_unlabeled_ind_train, labeled_ind_train,
                   invalidList, model, use_gpu, S_index, labelArr_true):
    model.eval()
    embDim = 512
    nLab = args.known_class
    embedding = np.zeros([len_unlabeled_ind_train + Len_labeled_ind_train + len(invalidList), embDim * nLab])

    queryIndex = []
    data_image = []
    labelArr = []
    uncertaintyArr = []
    S_ij = {}
    for batch_idx, (index, (data, labels)) in enumerate(unlabeledloader):
        if use_gpu:
            data, labels = data.cuda(), labels.cuda()
        # output = cout
        out, outputs = model(data)
        out = out.data.cpu().numpy()
        batchProbs = F.softmax(outputs, dim=1).data.cpu().numpy()
        maxInds = np.argmax(batchProbs, 1)
        for j in range(len(labels)):
            for c in range(nLab):
                if c == maxInds[j]:
                    embedding[index[j]][embDim * c: embDim * (c + 1)] = deepcopy(out[j]) * (1 - batchProbs[j][c])
                else:
                    embedding[index[j]][embDim * c: embDim * (c + 1)] = deepcopy(out[j]) * (-1 * batchProbs[j][c])
        # 当前的index 128 个 进入queryIndex array
        queryIndex += index

    print(f'number of image selected using KNN voting {len(queryIndex)}')
    embedding = torch.Tensor(embedding)
    chosen = init_centers(embedding, args.query_batch)
    queryIndex = chosen
    queryLabelArr = []
    # Assuming labeled_ind_train, invalidList and queryIndex are defined and are lists

    # Merging the lists labeled_ind_train and invalidList into a single list
    elements_to_remove = labeled_ind_train + invalidList

    # Using list comprehension to remove elements in queryIndex which are also found in elements_to_remove
    queryIndex = [element for element in queryIndex if element not in elements_to_remove]
    queryIndex = np.array(queryIndex)
    labelArr_true = np.array(labelArr_true)
    # Now, queryIndex contains only the elements not found in either labeled_ind_train or invalidList

    for i in range(len(queryIndex)):
        queryLabelArr.append(S_index[queryIndex[i]][0])

    queryLabelArr = np.array(queryLabelArr)
    labelArr = np.array(labelArr)
    precision = len(np.where(queryLabelArr < args.known_class)[0]) / args.query_batch
    recall = (len(np.where(queryLabelArr < args.known_class)[0]) + Len_labeled_ind_train) / (
            len(np.where(labelArr_true < args.known_class)[0]) + Len_labeled_ind_train)
    return queryIndex[np.where(queryLabelArr < args.known_class)[0]], queryIndex[
        np.where(queryLabelArr >= args.known_class)[0]], precision, recall


def openmax_sampling(args, unlabeledloader, Len_labeled_ind_train, model, use_gpu, openmax_beta=0.5):
    model.eval()
    queryIndex = []
    labelArr = []
    uncertainty_scores = []

    def compute_openmax_score(proba_out, mean_proba_known_classes, beta=openmax_beta):
        # Approximate distance by the difference between the probability and the mean probability
        distances = np.abs(proba_out - mean_proba_known_classes)
        scores = (1 - proba_out) * np.exp(-beta * distances)
        return np.sum(scores, axis=1)

    for batch_idx, (index, (data, labels)) in enumerate(unlabeledloader):
        if use_gpu:
            data, labels = data.cuda(), labels.cuda()
        if args.dataset == 'mnist':
            data = data.repeat(1, 3, 1, 1)
        with torch.no_grad():
            _, outputs = model(data)

        queryIndex += list(np.array(index.cpu().data))
        labelArr += list(np.array(labels.cpu().data))

        _, predicted = outputs.max(1)
        proba_out = softmax(outputs, dim=1)
        proba_out_known_classes = proba_out[:, :args.known_class]

        mean_proba_known_classes = proba_out_known_classes.mean(axis=1, keepdims=True)

        uncertainty = compute_openmax_score(proba_out_known_classes.cpu().numpy(), mean_proba_known_classes.cpu().numpy(), openmax_beta)
        uncertainty_scores += list(uncertainty)
        # if batch_idx > 10:
        #     break

    uncertainty_scores = np.array(uncertainty_scores)
    sorted_indices = np.argsort(-uncertainty_scores)
    selected_indices = sorted_indices[:args.query_batch]

    query_indices = np.array(queryIndex)[selected_indices]
    query_labels = np.array(labelArr)[selected_indices]

    known_indices = query_indices[query_labels < args.known_class]
    unknown_indices = query_indices[query_labels >= args.known_class]

    precision = len(known_indices) / len(query_indices)
    recall = (len(known_indices) + Len_labeled_ind_train) / (
            len(np.where(np.array(labelArr) < args.known_class)[0]) + Len_labeled_ind_train)

    return known_indices, unknown_indices, precision, recall

def openmax_sampling_hybrid(args, unlabeledloader, Len_labeled_ind_train, model, use_gpu, labelArr_true, openmax_beta=0.5):
    model.eval()
    queryIndex = []
    labelArr = []
    uncertainty_scores = []

    def compute_openmax_score(proba_out, mean_proba_known_classes, beta=openmax_beta):
        # Approximate distance by the difference between the probability and the mean probability
        distances = np.abs(proba_out - mean_proba_known_classes)
        scores = (1 - proba_out) * np.exp(-beta * distances)
        return np.sum(scores, axis=1)

    for batch_idx, (index, (data, labels)) in enumerate(unlabeledloader):
        if use_gpu:
            data, labels = data.cuda(), labels.cuda()
        if args.dataset == 'mnist':
            data = data.repeat(1, 3, 1, 1)
        with torch.no_grad():
            _, outputs = model(data)

        queryIndex += list(np.array(index.cpu().data))
        labelArr += list(np.array(labels.cpu().data))

        _, predicted = outputs.max(1)
        proba_out = softmax(outputs, dim=1)
        proba_out_known_classes = proba_out[:, :args.known_class]

        mean_proba_known_classes = proba_out_known_classes.mean(axis=1, keepdims=True)

        uncertainty = compute_openmax_score(proba_out_known_classes.cpu().numpy(),
                                            mean_proba_known_classes.cpu().numpy(), openmax_beta)
        uncertainty_scores += list(uncertainty)
        # if batch_idx > 10:
        #     break

    uncertainty_scores = np.array(uncertainty_scores)
    sorted_indices = np.argsort(-uncertainty_scores)
    selected_indices = sorted_indices[:args.query_batch]

    query_indices = np.array(queryIndex)[selected_indices]
    query_labels = np.array(labelArr)[selected_indices]

    known_indices = query_indices[query_labels < args.known_class]
    unknown_indices = query_indices[query_labels >= args.known_class]

    precision = len(known_indices) / args.query_batch
    recall = (len(known_indices) + Len_labeled_ind_train) / (
            len(np.where(np.array(labelArr_true) < args.known_class)[0]) + Len_labeled_ind_train)

    return known_indices, unknown_indices, precision, recall


def core_set(args, unlabeledloader, Len_labeled_ind_train, model, use_gpu):
    model.eval()
    queryIndex = []
    embedding_vectors = []
    S_per_class = {}
    S_index = {}
    labelArr = []
    for batch_idx, (index, (data, labels)) in enumerate(unlabeledloader):
        if use_gpu:
            data, labels = data.cuda(), labels.cuda()
        if args.dataset == 'mnist':
            data = data.repeat(1, 3, 1, 1)
        with torch.no_grad():
            embeddings, outputs = model(data)
        # 当前的index 128 个 进入queryIndex array
        queryIndex += list(np.array(index.cpu().data))
        # my_test_for_outputs = outputs.cpu().data.numpy()
        # print(my_test_for_outputs)
        # 这句code的意思就是把GPU上的数据转移到CPU上面然后再把数据类型从tensor转变为python的数据类型
        labelArr += list(np.array(labels.cpu().data))
        # activation value based
        # 这个function会return 128行然后每行21列的数据，return分两个部分，一个部分是tensor的数据类型然后是每行最大的数据
        # 另一个return的东西也是tensor的数据类型然后是每行的最大的值具体在这一行的具体位置
        v_ij, predicted = outputs.max(1)
        embedding_vectors += list(np.array(embeddings.cpu().data))
        # proba_out = torch.nn.functional.softmax(outputs, dim=1)

        # proba_out = torch.gather(proba_out, 1, predicted.unsqueeze(1))

        for i in range(len(predicted.data)):
            tmp_class = np.array(predicted.data.cpu())[i]
            tmp_index = index[i].item()

            tmp_label = np.array(labels.data.cpu())[i]
            tmp_value = np.array(v_ij.data.cpu())[i]

            if tmp_class not in S_per_class:
                S_per_class[tmp_class] = []

            S_per_class[tmp_class].append([tmp_value, tmp_class, tmp_index, tmp_label])

            if tmp_index not in S_index:
                S_index[tmp_index] = []

            S_index[tmp_index].append([tmp_value, tmp_class, tmp_label])
    kmeans = KMeans(n_clusters=args.query_batch)
    kmeans.fit(embedding_vectors)
    closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, embedding_vectors)

    selected_indices = [queryIndex[i] for i in closest]

    final_chosen_index = []
    invalid_index = []

    for i in range(len(selected_indices)):
        if S_index[selected_indices[i]][0][2] < args.known_class:
            final_chosen_index.append(selected_indices[i])
        elif S_index[selected_indices[i]][0][2] >= args.known_class:
            invalid_index.append(selected_indices[i])

    #
    precision = len(final_chosen_index) / args.query_batch
    # print(len(queryIndex_unknown))

    # recall = (len(final_chosen_index) + Len_labeled_ind_train) / (
    #        len([x for x in labelArr if args.known_class]) + Len_labeled_ind_train)

    recall = (len(final_chosen_index) + Len_labeled_ind_train) / (
            len(np.where(np.array(labelArr) < args.known_class)[0]) + Len_labeled_ind_train)

    return final_chosen_index, invalid_index, precision, recall

def core_set_hybrid(args, unlabeledloader, Len_labeled_ind_train, model, use_gpu, labelArr_true):
    model.eval()
    queryIndex = []
    embedding_vectors = []
    S_per_class = {}
    S_index = {}
    labelArr = []
    for batch_idx, (index, (data, labels)) in enumerate(unlabeledloader):
        if use_gpu:
            data, labels = data.cuda(), labels.cuda()
        if args.dataset == 'mnist':
            data = data.repeat(1, 3, 1, 1)
        with torch.no_grad():
            embeddings, outputs = model(data)
        # 当前的index 128 个 进入queryIndex array
        queryIndex += list(np.array(index.cpu().data))
        # my_test_for_outputs = outputs.cpu().data.numpy()
        # print(my_test_for_outputs)
        # 这句code的意思就是把GPU上的数据转移到CPU上面然后再把数据类型从tensor转变为python的数据类型
        labelArr += list(np.array(labels.cpu().data))
        # activation value based
        # 这个function会return 128行然后每行21列的数据，return分两个部分，一个部分是tensor的数据类型然后是每行最大的数据
        # 另一个return的东西也是tensor的数据类型然后是每行的最大的值具体在这一行的具体位置
        v_ij, predicted = outputs.max(1)
        embedding_vectors += list(np.array(embeddings.cpu().data))
        # proba_out = torch.nn.functional.softmax(outputs, dim=1)

        # proba_out = torch.gather(proba_out, 1, predicted.unsqueeze(1))

        for i in range(len(predicted.data)):
            tmp_class = np.array(predicted.data.cpu())[i]
            tmp_index = index[i].item()

            tmp_label = np.array(labels.data.cpu())[i]
            tmp_value = np.array(v_ij.data.cpu())[i]

            if tmp_class not in S_per_class:
                S_per_class[tmp_class] = []

            S_per_class[tmp_class].append([tmp_value, tmp_class, tmp_index, tmp_label])

            if tmp_index not in S_index:
                S_index[tmp_index] = []

            S_index[tmp_index].append([tmp_value, tmp_class, tmp_label])
    kmeans = KMeans(n_clusters=args.query_batch)
    kmeans.fit(embedding_vectors)
    closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, embedding_vectors)

    selected_indices = [queryIndex[i] for i in closest]

    final_chosen_index = []
    invalid_index = []

    for i in range(len(selected_indices)):
        if S_index[selected_indices[i]][0][2] < args.known_class:
            final_chosen_index.append(selected_indices[i])
        elif S_index[selected_indices[i]][0][2] >= args.known_class:
            invalid_index.append(selected_indices[i])

    #
    precision = len(final_chosen_index) / args.query_batch
    # print(len(queryIndex_unknown))

    # recall = (len(final_chosen_index) + Len_labeled_ind_train) / (
    #        len([x for x in labelArr if args.known_class]) + Len_labeled_ind_train)

    recall = (len(final_chosen_index) + Len_labeled_ind_train) / (
            len(np.where(np.array(labelArr_true) < args.known_class)[0]) + Len_labeled_ind_train)

    return final_chosen_index, invalid_index, precision, recall


def passive_and_implement_other_baseline(args, model, query, unlabeledloader, Len_labeled_ind_train,
                                         len_unlabeled_ind_train, use_gpu, labeled_ind_train, invalidList,
                                         unlabeled_ind_train, ordered_feature, ordered_label, labeled_index_to_label):
    index_knn = CIFAR100_EXTRACT_FEATURE_CLIP_new(labeled_ind_train + invalidList, unlabeled_ind_train, args,
                                                  ordered_feature, ordered_label)

    labelArr = []

    model.eval()
    #################################################################
    S_index = {}

    for batch_idx, (index, (data, labels)) in enumerate(unlabeledloader):

        if use_gpu:
            data, labels = data.cuda(), labels.cuda()

        _, outputs = model(data)

        v_ij, predicted = outputs.max(1)

        labelArr += list(np.array(labels.cpu().data))

        for i in range(len(data.data)):
            predict_class = predicted[i].detach()

            predict_value = np.array(v_ij.data.cpu())[i]

            predict_prob = outputs[i, :]

            tmp_index = index[i].item()

            true_label = np.array(labels.data.cpu())[i]

            S_index[tmp_index] = [true_label, predict_class, predict_value, predict_prob.detach().cpu()]

    #################################################################

    # 上半部分的code就是把Resnet里面的输出做了一下简单的数据处理，把21长度的数据取最大值然后把这个值和其在数据集里面的index，label组成一个字典的value放到S——ij里面

    # queryIndex 存放known class的地方
    queryIndex = []

    neighbor_unknown = {}

    detected_unknown = 0.0
    detected_known = 0.0

    for current_index in S_index:

        index_Neighbor, values = index_knn[current_index]

        true_label = S_index[current_index][0]

        count_known = 0.0
        count_unknown = 0.0

        for k in range(len(index_Neighbor)):

            n_index = index_Neighbor[k]

            if n_index in set(labeled_ind_train):
                count_known += 1

            elif n_index in set(invalidList):
                count_unknown += 1

        if count_unknown < count_known:

            queryIndex.append([current_index, count_known, true_label])

        else:
            detected_unknown += 1

    print("detected_unknown: ", detected_unknown)
    print("\n")

    queryIndex = sorted(queryIndex, key=lambda x: x[-2], reverse=True)

    #################################################################

    queryIndex = queryIndex[:2 * args.query_batch]
    newList = [sublist[0] for sublist in queryIndex]

    B_dataset = datasets.create(
        name=args.dataset, known_class_=args.known_class, init_percent_=args.init_percent,
        batch_size=args.batch_size, use_gpu=use_gpu,
        num_workers=args.workers, is_filter=args.is_filter, is_mini=args.is_mini, SEED=args.seed,
        unlabeled_ind_train=newList, labeled_ind_train=labeled_ind_train,
    )

    _, unlabeledloader = B_dataset.trainloader, B_dataset.unlabeledloader

    if args.query_strategy == "hybrid-BGADL":
        return bayesian_generative_active_learning_hybrid(args, unlabeledloader, Len_labeled_ind_train, model, use_gpu,
                                                   labelArr)
    if args.query_strategy == "hybrid-OpenMax":
        return openmax_sampling_hybrid(args, unlabeledloader, Len_labeled_ind_train, model, use_gpu, labelArr,
                                openmax_beta=0.5)
    if args.query_strategy == "hybrid-Core_set":
        return core_set_hybrid(args, unlabeledloader, Len_labeled_ind_train, model, use_gpu, labelArr)
    if args.query_strategy == "hybrid-BADGE_sampling":
        return badge_sampling_hybrid(args, unlabeledloader, Len_labeled_ind_train, len_unlabeled_ind_train, labeled_ind_train,
                              invalidList, model, use_gpu, S_index, labelArr)
    if args.query_strategy == "hybrid-uncertainty":
        return uncertainty_sampling_hybrid(args, unlabeledloader, Len_labeled_ind_train, model, use_gpu, labelArr)


def bayesian_generative_active_learning(args, unlabeledloader, Len_labeled_ind_train, model, use_gpu):
    model.eval()
    queryIndex = []
    labelArr = []
    uncertainty_scores = []

    for batch_idx, (index, (data, labels)) in enumerate(unlabeledloader):
        if use_gpu:
            data, labels = data.cuda(), labels.cuda()
        if args.dataset == 'mnist':
            data = data.repeat(1, 3, 1, 1)
        with torch.no_grad():
            _, outputs = model(data)


        queryIndex += list(np.array(index.cpu().data))
        labelArr += list(np.array(labels.cpu().data))

        _, predicted = outputs.max(1)
        proba_out = softmax(outputs, dim=1)
        proba_out = torch.gather(proba_out, 1, predicted.unsqueeze(1))

        uncertainty = 1 - proba_out.squeeze().cpu().numpy()
        uncertainty_scores += list(uncertainty)

    uncertainty_scores = np.array(uncertainty_scores)
    sorted_indices = np.argsort(-uncertainty_scores)
    selected_indices = sorted_indices[:args.query_batch]

    query_indices = np.array(queryIndex)[selected_indices]
    query_labels = np.array(labelArr)[selected_indices]

    known_indices = query_indices[query_labels < args.known_class]
    unknown_indices = query_indices[query_labels >= args.known_class]

    precision = len(known_indices) / len(query_indices)
    recall = (len(known_indices) + Len_labeled_ind_train) / (
            len(np.where(np.array(labelArr) < args.known_class)[0]) + Len_labeled_ind_train)

    return known_indices, unknown_indices, precision, recall


def bayesian_generative_active_learning_hybrid(args, unlabeledloader, Len_labeled_ind_train, model, use_gpu, labelArr_true):
    model.eval()
    queryIndex = []
    labelArr = []
    uncertainty_scores = []

    for batch_idx, (index, (data, labels)) in enumerate(unlabeledloader):
        if use_gpu:
            data, labels = data.cuda(), labels.cuda()
        if args.dataset == 'mnist':
            data = data.repeat(1, 3, 1, 1)
        with torch.no_grad():
            _, outputs = model(data)

        queryIndex += list(np.array(index.cpu().data))
        labelArr += list(np.array(labels.cpu().data))

        _, predicted = outputs.max(1)
        proba_out = softmax(outputs, dim=1)
        proba_out = torch.gather(proba_out, 1, predicted.unsqueeze(1))

        uncertainty = 1 - proba_out.squeeze().cpu().numpy()
        uncertainty_scores += list(uncertainty)

    uncertainty_scores = np.array(uncertainty_scores)
    sorted_indices = np.argsort(-uncertainty_scores)
    selected_indices = sorted_indices[:args.query_batch]

    query_indices = np.array(queryIndex)[selected_indices]
    query_labels = np.array(labelArr)[selected_indices]

    known_indices = query_indices[query_labels < args.known_class]
    unknown_indices = query_indices[query_labels >= args.known_class]

    precision = len(known_indices) / args.query_batch
    recall = (len(known_indices) + Len_labeled_ind_train) / (
            len(np.where(np.array(labelArr_true) < args.known_class)[0]) + Len_labeled_ind_train)

    return known_indices, unknown_indices, precision, recall


def certainty_sampling(args, unlabeledloader, Len_labeled_ind_train, model, use_gpu):
    model.eval()
    queryIndex = []
    labelArr = []
    certaintyArr = []
    precision, recall = 0, 0
    for batch_idx, (index, (data, labels)) in enumerate(unlabeledloader):
        if use_gpu:
            data, labels = data.cuda(), labels.cuda()
        if args.dataset == 'mnist':
            data = data.repeat(1, 3, 1, 1)
        features, outputs = model(data)

        certaintyArr += list(
            # Certainty(P) = max(P(x))
            np.array(torch.softmax(outputs, 1).max(1).values.cpu().data))
        queryIndex += index
        labelArr += list(np.array(labels.cpu().data))

    tmp_data = np.vstack((certaintyArr, queryIndex, labelArr)).T
    tmp_data = tmp_data[np.argsort(-tmp_data[:, 0])]  # Use negative sign to sort in descending order
    tmp_data = tmp_data.T
    queryIndex = tmp_data[1][-args.query_batch:].astype(int)
    labelArr = tmp_data[2].astype(int)
    queryLabelArr = tmp_data[2][-args.query_batch:]
    precision = len(np.where(queryLabelArr < args.known_class)[0]) / len(queryLabelArr)
    recall = (len(np.where(queryLabelArr < args.known_class)[0]) + Len_labeled_ind_train) / (
            len(np.where(labelArr < args.known_class)[0]) + Len_labeled_ind_train)
    return queryIndex[np.where(queryLabelArr < args.known_class)[0]], queryIndex[
        np.where(queryLabelArr >= args.known_class)[0]], precision, recall
