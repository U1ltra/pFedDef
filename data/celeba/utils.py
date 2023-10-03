import random
import time
import torch

import numpy as np
from tqdm import tqdm

attr_used = ["Eyeglasses", "Male", "Smiling", "Wearing_Hat"]
attr_to_idx = {
    "5_o_Clock_Shadow": 0,
    "Arched_Eyebrows": 1,
    "Attractive": 2,
    "Bags_Under_Eyes": 3,
    "Bald": 4,
    "Bangs": 5,
    "Big_Lips": 6,
    "Big_Nose": 7,
    "Black_Hair": 8,
    "Blond_Hair": 9,
    "Blurry": 10,
    "Brown_Hair": 11,
    "Bushy_Eyebrows": 12,
    "Chubby": 13,
    "Double_Chin": 14,
    "Eyeglasses": 15,
    "Goatee": 16,
    "Gray_Hair": 17,
    "Heavy_Makeup": 18,
    "High_Cheekbones": 19,
    "Male": 20,
    "Mouth_Slightly_Open": 21,
    "Mustache": 22,
    "Narrow_Eyes": 23,
    "No_Beard": 24,
    "Oval_Face": 25,
    "Pale_Skin": 26,
    "Pointy_Nose": 27,
    "Receding_Hairline": 28,
    "Rosy_Cheeks": 29,
    "Sideburns": 30,
    "Smiling": 31,
    "Straight_Hair": 32,
    "Wavy_Hair": 33,
    "Wearing_Earrings": 34,
    "Wearing_Hat": 35,
    "Wearing_Lipstick": 36,
    "Wearing_Necklace": 37,
    "Wearing_Necktie": 38,
    "Young": 39,
}

def iid_divide(l, g):
    """
    https://github.com/TalwalkarLab/leaf/blob/master/data/utils/sample.py
    divide list `l` among `g` groups
    each group has either `int(len(l)/g)` or `int(len(l)/g)+1` elements
    returns a list of groups
    """
    num_elems = len(l)
    group_size = int(len(l) / g)
    num_big_groups = num_elems - g * group_size
    num_small_groups = g - num_big_groups
    glist = []
    for i in range(num_small_groups):
        glist.append(l[group_size * i: group_size * (i + 1)])
    bi = group_size * num_small_groups
    group_size += 1
    for i in range(num_big_groups):
        glist.append(l[bi + group_size * i:bi + group_size * (i + 1)])
    return glist


def split_list_by_indices(l, indices):
    """
    divide list `l` given indices into `len(indices)` sub-lists
    sub-list `i` starts from `indices[i]` and stops at `indices[i+1]`
    returns a list of sub-lists
    """
    res = []
    current_index = 0
    for index in indices:
        res.append(l[current_index: index])
        current_index = index

    return res


def split_dataset_by_labels(dataset, n_classes, n_clients, n_clusters, alpha, frac, seed=1234):
    """
    split classification dataset among `n_clients`. The dataset is split as follow:
        1) classes are grouped into `n_clusters`
        2) for each cluster `c`, samples are partitioned across clients using dirichlet distribution

    Inspired by the split in "Federated Learning with Matched Averaging"__(https://arxiv.org/abs/2002.06440)

    :param dataset:
    :type dataset: torch.utils.Dataset
    :param n_classes: number of classes present in `dataset`
    :param n_clients: number of clients
    :param n_clusters: number of clusters to consider; if it is `-1`, then `n_clusters = n_classes`
    :param alpha: parameter controlling the diversity among clients
    :param frac: fraction of dataset to use
    :param seed:
    :return: list (size `n_clients`) of subgroups, each subgroup is a list of indices.
    """
    if n_clusters == -1:
        n_clusters = n_classes

    rng_seed = (seed if (seed is not None and seed >= 0) else int(time.time()))
    rng = random.Random(rng_seed)
    np.random.seed(rng_seed)

    all_labels = list(range(n_classes))
    rng.shuffle(all_labels)
    clusters_labels = iid_divide(all_labels, n_clusters)

    label2cluster = dict()  # maps label to its cluster
    for group_idx, labels in enumerate(clusters_labels):
        for label in labels:
            label2cluster[label] = group_idx
    
#     from IPython import embed
#     embed()
    # get subset
    n_samples = int(len(dataset) * frac)
    selected_indices = rng.sample(list(range(len(dataset))), n_samples)

    clusters_sizes = np.zeros(n_clusters, dtype=int)
    clusters = {k: [] for k in range(n_clusters)}

    print("Splitting dataset into {} clusters".format(n_clusters))
    for idx in tqdm(selected_indices):
        _, label = dataset[idx]
        group_id = label2cluster[label]
        clusters_sizes[group_id] += 1
        clusters[group_id].append(idx)

    print("Shuffling clusters")
    for _, cluster in tqdm(clusters.items()):
        rng.shuffle(cluster)

    clients_counts = np.zeros((n_clusters, n_clients), dtype=np.int64)  # number of samples by client from each cluster

    for cluster_id in range(n_clusters):
        weights = np.random.dirichlet(alpha=alpha * np.ones(n_clients))
        clients_counts[cluster_id] = np.random.multinomial(clusters_sizes[cluster_id], weights)

    clients_counts = np.cumsum(clients_counts, axis=1)

    clients_indices = [[] for _ in range(n_clients)]
    for cluster_id in range(n_clusters):
        cluster_split = split_list_by_indices(clusters[cluster_id], clients_counts[cluster_id])

        for client_id, indices in enumerate(cluster_split):
            clients_indices[client_id] += indices

    return clients_indices


def pathological_non_iid_split(dataset, n_classes, n_clients, n_classes_per_client, frac=1, seed=1234):
    """
    split classification dataset among `n_clients`. The dataset is split as follow:
        1) sort the data by label
        2) divide it into `n_clients * n_classes_per_client` shards, of equal size.
        3) assign each of the `n_clients` with `n_classes_per_client` shards

    Inspired by the split in
     "Communication-Efficient Learning of Deep Networks from Decentralized Data"__(https://arxiv.org/abs/1602.05629)

    :param dataset:
    :type dataset: torch.utils.Dataset
    :param n_classes: umber of classes present in `dataset`
    :param n_clients: number of clients
    :param n_classes_per_client:
    :param frac: fraction of dataset to use
    :param seed:
    :return: list (size `n_clients`) of subgroups, each subgroup is a list of indices.
    """
    rng_seed = (seed if (seed is not None and seed >= 0) else int(time.time()))
    rng = random.Random(rng_seed)
    np.random.seed(rng_seed)

    # get subset
    n_samples = int(len(dataset) * frac)
    selected_indices = rng.sample(list(range(len(dataset))), n_samples)

    label2index = {k: [] for k in range(n_classes)}
    for idx in selected_indices:
        _, label = dataset[idx]
        label2index[label].append(idx)

    sorted_indices = []
    for label in label2index:
        sorted_indices += label2index[label]

    n_shards = n_clients * n_classes_per_client
    shards = iid_divide(sorted_indices, n_shards)
    random.shuffle(shards)
    tasks_shards = iid_divide(shards, n_clients)

    clients_indices = [[] for _ in range(n_clients)]
    for client_id in range(n_clients):
        for shard in tasks_shards[client_id]:
            clients_indices[client_id] += shard

    return clients_indices
    
def filter_dataset(attr, keep_attr = attr_used):
    print("keep_attr: ", keep_attr)
    indices = None
    for attr_name in keep_attr:
        if indices is None:
            indices = (attr[:, attr_to_idx[attr_name]] == 1)
        else:
            indices = indices | (attr[:, attr_to_idx[attr_name]] == 1)
    print("indices: ", indices.shape)
    indices = torch.nonzero(indices).squeeze()
    print("indices: ", indices.shape)

    return indices.numpy()