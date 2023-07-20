import os
import torch
import numpy as np
import copy

from transfer_attacks.Args import *
from transfer_attacks.TA_utils import *

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
    base_path = "/home/ubuntu/Documents/jiarui/experiments/extra_train_inject/inject"

    client_weights = np.load(
        "/home/ubuntu/Documents/jiarui/experiments/multi_atker/client_weights.npy"
    )
    atk_nums = [1, 5, 10, 20]
    # calculate the theoretical best scale for each attack number
    best_scales = [1 / client_weights[0:atk_num].sum() for atk_num in atk_nums]
    print(best_scales)
    params = []
    for idx, scale in enumerate(best_scales):
        if scale - 4 <= 0:
            for i in range(1, int(scale) + 1):
                params.append((atk_nums[idx], i))
            params.append((atk_nums[idx], scale))
            params.append((atk_nums[idx], scale + 1))
            params.append((atk_nums[idx], scale + 2))
        else:
            for i in range(0, 10, 2):
                params.append((atk_nums[idx], scale + i - 4))
    for num, scale in params:
        print(f"atk_num: {num}, scale: {scale}")

    exp_names = [f"atk_{atk_num}_scale{scale}" for atk_num, scale in params]

    labels = [
        name for name in exp_names
    ]

    all_label_acc = [
        f"{base_path}/{ name }/eval/all_acc_zeroVar_fullClients.npy" for name in exp_names
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
        f"{base_path}/{ name }/eval/all_adv_acc_zeroVar_fullClients.npy" for name in exp_names
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
    print(f"| {placeHolder1} | {placeHolder1} | {placeHolder1} | {placeHolder1} |")
    for i, label in enumerate(labels):
        if i >= len(exp_names):
            print(f"| {label:<17} | {'-':<17} | {all_label_avg_acc[i]*100:16.3f}% | {all_label_avg_adv_acc[i]*100:16.3f}% |")
            continue
        print(f"| {params[i][0]:<17} | {params[i][1]:17.6f} | {all_label_avg_acc[i]*100:16.3f}% | {all_label_avg_adv_acc[i]*100:16.3f}% |")

def markdown_table2():
    base_path = "/home/ubuntu/Documents/jiarui/experiments/FAT_progressive/FedAvg_adv_progressive/weights"

    epochs = [i for i in range(1, 250, 2)]

    exp_names = [f"gt{epoch}" for epoch in epochs]

    labels = [
        epoch for epoch in epochs
    ]

    all_label_acc = [
        f"{base_path}/{ name }/eval/all_acc_external_epsilon_LL.npy" for name in exp_names
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
        f"{base_path}/{ name }/eval/all_adv_acc_external_epsilon_LL.npy" for name in exp_names
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
        if i >= len(exp_names):
            print(f"| {label:<17} | {all_label_avg_acc[i]*100:16.3f}% | {all_label_avg_adv_acc[i]*100:16.3f}% |")
            continue
        print(f"| {label:<17} | {all_label_avg_acc[i]*100:16.3f}% | {all_label_avg_adv_acc[i]*100:16.3f}% |")

def markdown_table3():
    base_path = "/home/ubuntu/Documents/jiarui/experiments/extra_train_inject/5_xhat_yhat_unhard_redo/FedAvg_adv_unharden_portion_1.0/weights"

    epochs = [i for i in range(0, 51, 2)]

    exp_names = [f"gt{epoch}" for epoch in epochs]

    labels = [
        epoch for epoch in epochs
    ]

    all_label_acc = [
        f"{base_path}/{ name }/eval/all_acc.npy" for name in exp_names
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
        f"{base_path}/{ name }/eval/all_adv_acc.npy" for name in exp_names
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
        if i >= len(exp_names):
            print(f"| {label:<17} | {all_label_avg_acc[i]*100:16.3f}% | {all_label_avg_adv_acc[i]*100:16.3f}% |")
            continue
        print(f"| {label:<17} | {all_label_avg_acc[i]*100:16.3f}% | {all_label_avg_adv_acc[i]*100:16.3f}% |")


def mv_file():
    weigths_path = "/home/ubuntu/Documents/jiarui/experiments/extra_train_inject/extra_train/train_client_weights.npy"
    base_path = "/home/ubuntu/Documents/jiarui/experiments/extra_train_inject/extra_train/client_num20/weights"
    epoch_set = [i for i in range(1,50,2)]
    epoch_set.sort()

    for epoch in epoch_set:
        os.system(f"cp {weigths_path} {base_path}/chkpts_{epoch}/weights/train_client_weights.npy")

def update_weights():
    base_path = "/home/ubuntu/Documents/jiarui/experiments/extra_train_inject/5clients_x_y_unhard/extra_benign/weights"
    epoch_set = [i for i in range(1,50,2)]
    epoch_set.sort()
    weights = np.ones((40,1))

    for epoch in epoch_set:
        np.save(f"{base_path}/chkpts_{epoch}/weights/train_client_weights.npy", weights)

def early_replace_updates():
    base_path = "/home/ubuntu/Documents/jiarui/experiments/early_rep/early_replace_X"
    epoch_set = [i for i in range(1,250,24)]
    epoch_set.sort()

    placeHolder1 = '-' * 10
    print(f"| {placeHolder1} | {placeHolder1} | {placeHolder1} |")
    for epoch in epoch_set:
        path = f"{base_path}/probe_at_epoch{epoch}/model_distance.npy"
        model_distance = np.load(path, allow_pickle=True)
        # print(f"--------------------------probe epoch {epoch}--------------------------")

        for i in range(model_distance.shape[0]):
            diff = (model_distance[i, 0] - model_distance[0, 0]).sum()
            # print(f"epoch {i}: {diff}")

            print(f"| {epoch:<10} | {i:<10} | {diff:9.3f} |")

def pipeline_results():
    defense_mechanisms = ["trimmed_mean", "median", "krum"] # "bulyan"
    global_model_fractions = [0.01, 0.05, 0.1]
    params = []
    for defense in defense_mechanisms:
        for global_model_fraction in global_model_fractions:
            params.append((defense, global_model_fraction))

    for param in params:
        defense, global_model_fraction = param
        print(f"\n--------------------------{defense} {global_model_fraction}--------------------------")

        base_path = f"/home/ubuntu/Documents/jiarui/experiments/atk_pipeline/fixedCode/unharden_global_weight_consistent/def_{defense}_frac{global_model_fraction}"
        stage_names = ["unharden", "before_replace", "replace"] # "atk_start", "unharden", "before_replace", "replace"
        stage_paths = [f"{base_path}/{name}/weights/eval" for name in stage_names]

        # train_names = ["FAT_train", "unharden_train"]
        # for i in range(0, 50, 5):
        #     stage_paths.append(f"{base_path}/{train_names[0]}/weights/gt{i}/eval")
        # for i in range(5, 50, 5):
        #     stage_paths.append(f"{base_path}/{train_names[1]}/weights/gt{i}/eval")
        
        # stage_names.extend(range(0, 50, 5))
        # stage_names.extend(range(5, 50, 5))

        placeHolder1 = '-' * 10
        print(f"| {placeHolder1} | {placeHolder1} | {placeHolder1}|")
        for i, stage_path in enumerate(stage_paths):
            acc_path = f"{stage_path}/all_acc.npy"
            adv_acc_path = f"{stage_path}/all_adv_acc.npy"
            acc = np.load(acc_path)
            adv_acc = np.load(adv_acc_path)
            print(f"| {stage_names[i]:<10} | {np.sum(acc) / (acc.shape[0] * acc.shape[1]) * 100:9.3f} | {np.sum(adv_acc) / (adv_acc.shape[0] * adv_acc.shape[1]) * 100:9.3f} |")

def removed_indices():
    defense = "krum" # "median", "trimmed_mean", "krum"
    base_path = "/home/ubuntu/Documents/jiarui/experiments/atk_pipeline/fixedCode/unharden_pip_def/defense_krum"
    base_path = f"{base_path}/dump"
    atk_clients = torch.arange(0, 5)
    all_clients = torch.arange(0, 40)
    total_params = 0

    normal_rounds = []
    for i in range(49):
        path = f"{base_path}/round{i}_krum.pkl"
        with open(path, 'rb') as file:
            loaded_list = pickle.load(file)
        
        removed_counts = torch.zeros(40, dtype=torch.long)
        
        for key, removed_indices in loaded_list:
            flat_removed_indices = torch.flatten(removed_indices)
            unique_numbers, counts = torch.unique(flat_removed_indices, return_counts=True)
            unique_numbers = unique_numbers.cpu()
            counts = counts.cpu()

            if defense == "median" or "krum":
                for unique_number, count in zip(unique_numbers, counts):
                    # add to other clients ececpt the current one 
                    # cause each count of the current unique number will cause the loss from all other clients
                    removed_counts[all_clients[all_clients != unique_number]] += count
            elif defense == "trimmed_mean":
                removed_counts[unique_numbers] += counts

            if i == 0:
                total_params += flat_removed_indices.shape[0]

        normal_rounds.append(removed_counts)

    # avg counts
    normal_rounds = torch.stack(normal_rounds).to(torch.float32)
    avg_normal = normal_rounds.mean(dim=0) / total_params
    # bar plot
    import matplotlib.pyplot as plt
    bars = plt.bar(torch.arange(1, 41), avg_normal)
    for i in atk_clients:
        bars[i].set_color('r')
    plt.show()
    # save plot
    plt.savefig(f"/home/ubuntu/Documents/jiarui/pFedDef/Evaluation/normal_rounds.png")


    path = f"{base_path}/round49_krum.pkl"
    with open(path, 'rb') as file:
        loaded_list = pickle.load(file)
    
    removed_counts = torch.zeros(40, dtype=torch.long)

    for key, removed_indices in loaded_list:
        flat_removed_indices = torch.flatten(removed_indices)
        unique_numbers, counts = torch.unique(flat_removed_indices, return_counts=True)
        unique_numbers = unique_numbers.cpu()
        counts = counts.cpu()

        if defense == "median" or "krum":
            for unique_number, count in zip(unique_numbers, counts):
                removed_counts[all_clients[all_clients != unique_number]] += count
        elif defense == "trimmed_mean":
            removed_counts[unique_numbers] += counts

    
    # bar plot
    plt.bar(torch.arange(1, 41), removed_counts / total_params)
    plt.show()
    # save plot
    plt.savefig(f"/home/ubuntu/Documents/jiarui/pFedDef/Evaluation/round49.png")

pipeline_results()
# removed_indices()


