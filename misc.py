import os
import torch
import numpy as np
import copy

def avg_ckpts():
    base_path = "/home/ubuntu/Documents/jiarui/experiments/FAT_progressive/FedAvg_adv_progressive/weights"
    epochs = [29, 49, 59, 67, 69, 79, 89]
    all_path = [f"{base_path}/gt{i}/chkpts_0.pt" for i in epochs]

    weights = []
    for i, path in enumerate(all_path):
        weights.append(torch.load(path))


    avg_dict = copy.deepcopy(weights[0])
    for key in avg_dict:
        avg_dict[key] = torch.zeros_like(avg_dict[key])

    for i, weight in enumerate(weights):
        for key in weight:
            if avg_dict[key].data.dtype == torch.float32:
                avg_dict[key] += weight[key] / len(weights)
            else:
                avg_dict[key] += weight[key]

    torch.save(avg_dict, f"{base_path}/avg_weights.pt")

def markdown_table():
    base_path = "/home/ubuntu/Documents/jiarui/experiments/extra_train_inject/extra_train/client_num10/weights"

    epoch_set = [i for i in range(1,50,2)]
    epoch_set.sort()

    labels = [
        str(epoch) for epoch in epoch_set
    ]

    all_label_acc = [
        f"{base_path}/chkpts_{ epoch }/eval/all_acc_zeroVar_fullClients.npy" for epoch in epoch_set
    ]
    all_label_acc.append(
        "/home/ubuntu/Documents/jiarui/experiments/FedAvg_adv/gt_1leaner_adv/weights/gt200/eval/all_acc.npy"
    )
    labels.append("FAT")
    all_label_acc.append(
        "/home/ubuntu/Documents/jiarui/experiments/fedavg/gt_epoch200/eval/all_acc.npy"
    )
    labels.append("fedavg")

    all_label_avg_acc = []
    for i, acc_path in enumerate(all_label_acc):
        res = np.load(acc_path)
        all_label_avg_acc.append(
            np.sum(res) / (res.shape[0] * res.shape[1])
        )

    all_label_adv_acc = [
        f"{base_path}/chkpts_{ epoch }/eval/all_adv_acc_zeroVar_fullClients.npy" for epoch in epoch_set
    ]
    all_label_adv_acc.append(
        "/home/ubuntu/Documents/jiarui/experiments/FedAvg_adv/gt_1leaner_adv/weights/gt200/eval/all_adv_acc.npy"
    )
    all_label_adv_acc.append(
        "/home/ubuntu/Documents/jiarui/experiments/fedavg/gt_epoch200/eval/all_adv_acc.npy"
    )

    all_label_avg_adv_acc = []
    for i, acc_path in enumerate(all_label_adv_acc):
        res = np.load(acc_path)
        all_label_avg_adv_acc.append(
            np.sum(res) / (res.shape[0] * res.shape[1])
        )

    placeHolder1 = '-' * 17
    print(f"| {placeHolder1} | {placeHolder1} | {placeHolder1} |")
    for i, label in enumerate(labels):
        if i >= len(epoch_set):
            print(f"| {label:<17} | {all_label_avg_acc[i]*100:16.3f}% | {all_label_avg_adv_acc[i]*100:16.3f}% |")
            continue
        print(f"| {epoch_set[i]:<17} | {all_label_avg_acc[i]*100:16.3f}% | {all_label_avg_adv_acc[i]*100:16.3f}% |")

def mv_file():
    weigths_path = "/home/ubuntu/Documents/jiarui/experiments/extra_train_inject/extra_train/train_client_weights.npy"
    base_path = "/home/ubuntu/Documents/jiarui/experiments/extra_train_inject/extra_train/client_num20/weights"
    epoch_set = [i for i in range(1,50,2)]
    epoch_set.sort()

    for epoch in epoch_set:
        os.system(f"cp {weigths_path} {base_path}/chkpts_{epoch}/weights/train_client_weights.npy")

def inspect_tm():
    base_path = "/home/ubuntu/Documents/jiarui/experiments/atk_pipeline/unharden_rep_pipeline_inspect_tm"
    rounds = [i for i in range(1, 16, 1)]
    tm_indices = [torch.load(f"{base_path}/tm_indices_round{i}.pt") for i in rounds]

    remove_idx = [0, 1, -1, -2]
    remove_map = torch.zeros(40, len(rounds))
    for i, index in enumerate(tm_indices):
        print(f"round {i+1}", end=": ")
        for idx in remove_idx:
            print(index[idx].shape, end=" ")
            removed = index[idx]
            flat = torch.flatten(removed)
            unique_values, counts = torch.unique(flat, return_counts=True)
            for value, count in zip(unique_values, counts):
                remove_map[value][i] += count
        print()
    
    print(remove_map)

markdown_table()
