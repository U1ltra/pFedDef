import os
import torch
import numpy as np
import copy
import matplotlib.pyplot as plt


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
    base_path = "/home/ubuntu/Documents/jiarui/experiments/atk_pipeline/fixedCode/def_krum_modelwise/def_krum_modelwise/FAT_train/weights"

    epochs = [i for i in range(0, 150, 10)]

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
    defense_mechanisms = [None]
    atk_clients = [1]
    atk_rounds = [0]
    alpha = [0.3]
    params = []
    for defense in defense_mechanisms:
        for atk_client in atk_clients:
            for atk_round in atk_rounds:
                for a in alpha:
                    params.append((defense, atk_client, atk_round, a))


    for param in params:
        defense, atk_client, atk_round, a = param
        print(f"\n--------------------------{defense} {atk_client} {atk_round} {a}--------------------------")

        # base_path = f"/home/ubuntu/Documents/jiarui/experiments/verify_correlation2/def_{defense}_atk_client_{atk_client}_atk_round_{atk_round}_alpha_{a}"
        base_path = f"/home/ubuntu/Documents/jiarui/experiments/NeurlPS_workshop/unharden_FAT_no_def_cifar10/def_None_atk_client_3_atk_round_1_reviewImprove"
        stage_names = ["unharden", "before_replace", "replace"] # "atk_start", "unharden", "before_replace", "replace"
        stage_paths = [f"{base_path}/{name}/weights/eval" for name in stage_names]

        train_names = ["FAT_train", "unharden_train"]
        # for i in range(0, 250, 20):
        #     if i != 0:
        #         stage_paths.append(f"{base_path}/{train_names[0]}/weights/gt{i-1}/eval")
        #     else:
        #         stage_paths.append(f"{base_path}/{train_names[0]}/weights/gt{i}/eval")
        for i in range(0, 50, 5):
            stage_paths.append(f"{base_path}/{train_names[1]}/weights/gt{i}/eval")
        
        # stage_names.extend(range(0, 250, 20))
        stage_names.extend(range(0, 50, 5))

        placeHolder1 = '-' * 10
        print(f"| {placeHolder1} | {placeHolder1} | {placeHolder1} |")
        for i, stage_path in enumerate(stage_paths):
            acc_path = f"{stage_path}/all_acc.npy"
            adv_acc_path = f"{stage_path}/all_adv_acc.npy"
            if not os.path.exists(acc_path):
                print(f"not exist: {acc_path}")
                continue
            acc = np.load(acc_path)
            adv_acc = np.load(adv_acc_path)
            print(f"| {stage_names[i]:<10} | {np.sum(acc) / (acc.shape[0] * acc.shape[1]) * 100:9.3f} | {np.sum(adv_acc) / (adv_acc.shape[0] * adv_acc.shape[1]) * 100:9.3f} |")
            
            # pring std
            external_adv_acc_path = f"{stage_path}/all_adv_acc_external.npy"
            external_adv_acc = 0
            print(f"| {stage_names[i]:<10} | {np.std(acc) * 100:9.4f} | {np.std(adv_acc) * 100:9.4f} | {np.std(external_adv_acc) * 100:9.4f} | (std)")

def cal_params_l2(model_dict):
    total_params = 0
    for key in model_dict:
        if model_dict[key].data.dtype == torch.float32:
            total_params += torch.norm(model_dict[key], p=2).item()
    return total_params ** 0.5

def experiments_loop():
    defenses = ["trimmed_mean", "median"]
    fracs = [1]
    atk_clients = [5]
    params = []
    for defense in defenses:
        for atk_client in atk_clients:
            params.append((defense, atk_client))
    round_list = [0, 0]
    
    for param in params:
        defense, atk_client = param
        abbrev = "tm" if defense == "trimmed_mean" else defense
        print(f"\n--------------------------{defense}--------------------------")

        # base_path = f"/home/ubuntu/Documents/jiarui/experiments/atk_pipeline/fixedCode/unharden_global_weight_consistent/def_{defense}_frac{frac}"
        # base_path = f"/home/ubuntu/Documents/jiarui/experiments/atk_pipeline/fixedCode/unharden_pip_def2/def_{defense}"
        # base_path = f"/home/ubuntu/Documents/jiarui/experiments/atk_pipeline/test/def_trimmed_mean_weight_0.0"
        base_path = f"/home/ubuntu/Documents/jiarui/experiments/NeurlPS_workshop/diff_replace_configs/fedavg2FAT/def_{defense}_atk_client_{atk_client}_atk_round_1"
        # base_path = f"/home/ubuntu/Documents/jiarui/experiments/NeurlPS_workshop/diff_replace_configs/fedavg2FAT_def/def_{defense}_atk_client_5_atk_round_1"

        process_removed_indices(base_path, defense, abbrev, base_path.split('/')[-1], 40, atk_client, round_list)
        # process_model_difference(base_path, round_list)
        # process_updates(base_path, round_list)

def process_removed_indices(base_path, defense, abbrev, id, client_num, atk_client_num, round_list):
    base_path = f"{base_path}/dump"
    atk_clients = torch.arange(0, atk_client_num)
    all_clients = torch.arange(0, client_num)
    total_params = 0
    saved_or_removed = "saved" if (defense == "median" or defense == "krum" or defense == "krum_modelwise") else "removed"

    normal_rounds = []
    for idx_i, i in enumerate(round_list[:-1]):
        if (idx_i+1) % 5 == 0:
            print(f"Working on round {i}")
        path = f"{base_path}/round{i}_{abbrev}.pkl"
        if not os.path.exists(path):
            continue
        with open(path, 'rb') as file:
            loaded_list = pickle.load(file)
        
        removed_counts = torch.zeros(40, dtype=torch.long)
        
        for key, removed_indices in loaded_list:
            flat_removed_indices = torch.flatten(removed_indices)
            unique_numbers, counts = torch.unique(flat_removed_indices, return_counts=True)
            unique_numbers = unique_numbers.cpu()
            counts = counts.cpu()

            if defense == "median" or defense == "krum" or defense == "krum_modelwise":
                saved_counts = torch.zeros(40, dtype=torch.long)
                for unique_number, count in zip(unique_numbers, counts):
                    # add to other clients ececpt the current one 
                    # cause each count of the current unique number will cause the loss from all other clients
                    # removed_counts[all_clients[all_clients != unique_number]] += count

                    # add to the current client
                    removed_counts[unique_number] += count # lets plot the saved parameters portion instead of the removed portion despite the varaible name

            elif defense == "trimmed_mean":
                removed_counts[unique_numbers] += counts

            if i == 0:
                total_params += flat_removed_indices.shape[0]

        normal_rounds.append(removed_counts)

    # avg counts
    if len(normal_rounds) != 0:
        normal_rounds = torch.stack(normal_rounds).to(torch.float32)
        avg_normal = normal_rounds.mean(dim=0) / total_params
        print(torch.mean(avg_normal[0:atk_client_num]))
        # bar plot
        # new figure
        plt.figure()
        bars = plt.bar(torch.arange(1, 41), avg_normal, color='tab:blue')
        plt.xlabel("Client ID")
        plt.ylabel(f"Portion of {saved_or_removed} parameters")
        plt.title(f"Portion of {saved_or_removed} parameters - {abbrev} defense (normal rounds)")
        for i in atk_clients:
            bars[i].set_color('r')
        plt.show()
        # save plot
        plt.savefig(f"/home/ubuntu/Documents/jiarui/pFedDef/Evaluation/plots/normal_rounds_{abbrev}_{id}.png")


    print(f"Working on round {round_list[-1]}")
    path = f"{base_path}/round{round_list[-1]}_{abbrev}.pkl"
    with open(path, 'rb') as file:
        loaded_list = pickle.load(file)
    
    removed_counts = torch.zeros(40, dtype=torch.long)

    for key, removed_indices in loaded_list:
        flat_removed_indices = torch.flatten(removed_indices)
        unique_numbers, counts = torch.unique(flat_removed_indices, return_counts=True)
        unique_numbers = unique_numbers.cpu()
        counts = counts.cpu()

        if defense == "median" or defense == "krum" or defense == "krum_modelwise":
            saved_counts = torch.zeros(40, dtype=torch.long)
            for unique_number, count in zip(unique_numbers, counts):
                # removed_counts[all_clients[all_clients != unique_number]] += count
                removed_counts[unique_number] += count
        elif defense == "trimmed_mean":
            removed_counts[unique_numbers] += counts    
    
    # bar plot
    plt.figure()
    bars = plt.bar(torch.arange(1, 41), removed_counts / total_params, color='tab:orange')
    for i in atk_clients:
        bars[i].set_color('r')
    plt.xlabel("Client ID")
    plt.ylabel(f"Portion of {saved_or_removed} parameters")
    plt.title(f"Portion of {saved_or_removed} parameters - {abbrev} defense (atk round)")
    plt.show()
    # save plot
    plt.savefig(f"/home/ubuntu/Documents/jiarui/pFedDef/Evaluation/plots/round{round_list[-1]}_{abbrev}_{id}.png")
    plt.close("all")

def process_model_difference(base_path, round_list):
    diff_norms = []
    for round in round_list:
        pkl_path = f"{base_path}/dump/round{round}_model_diff.pkl"
        print(f"Working on {pkl_path}")
        with open(pkl_path, 'rb') as file:
            model_diff = pickle.load(file)
        diff_norm = cal_params_l2(model_diff)
        diff_norms.append(diff_norm)
    # markdown table
    placeHolder1 = '-' * 10
    print(f"| {placeHolder1} | {placeHolder1} |")
    for i, round in enumerate(round_list):
        print(f"| {round:<10} | {diff_norms[i]:<10.4f} |")

def process_updates(base_path, round_list):
    updates_norms = []
    for round in round_list:
        pkl_path = f"{base_path}/dump/round{round}_update.pkl"
        print(f"Working on {pkl_path}")
        with open(pkl_path, 'rb') as file:
            load_updates = pickle.load(file)
        global_updates = load_updates[0]
        atk_updates = load_updates[1]

        all_norms = []
        for g_update in global_updates:
            global_updates_norm = cal_params_l2(g_update)
            all_norms.append(global_updates_norm)
        updates_norms.append((all_norms, None))
        print(len(global_updates))
        print(updates_norms)
        
    # markdown table
    placeHolder1 = '-' * 10
    placeHolder2 = f"{placeHolder1} |" * len(global_updates) # total num of clients
    placeHolder3 = [f" client {i} |" for i in range(len(global_updates))]

    print(f"| round | {''.join(placeHolder3)}")
    print(f"| {placeHolder1} | {placeHolder2}")
    for i, round in enumerate(round_list):
        print(f"| {round:<10} | ", end='')
        for j in range(len(global_updates)):
            print(f"{updates_norms[i][0][j]:<10.3f} | ", end='')
        print()


def linux_command():
    defenses = ["trimmed_mean", "median", "krum_modelwise"]
    atk_clients = [5, 10]
    atk_rounds = [5, 10, 20]
    params = []
    for defense in defenses:
        for atk_num in atk_clients:
            for atk_round in atk_rounds:
                params.append((defense, atk_num, atk_round))
    
    for param in params:
        defense, atk_num, atk_round = param
        abbrev = "tm" if defense == "trimmed_mean" else defense

        base_path = f"/home/ubuntu/Documents/jiarui/experiments/multi_round_atk/atk_def_{defense}_atk_client_{atk_num}_atk_round_{atk_round}/dump"
        rounds_set = [i for i in range(0, atk_round+1)]
        round_save = [i for i in range(0, atk_round, 5)] + [atk_round]
        to_remove = []
        for i in rounds_set:
            if i not in round_save:
                to_remove.append(i)

        for i in to_remove:
            print("--------------------------")
            command = f"rm -r {base_path}/round{i}_{abbrev}.pkl"
            print(command)
            os.system(command)

            command = f"rm -r {base_path}/round{i}_model_diff.pkl"
            print(command)
            os.system(command)

            command = f"rm -r {base_path}/round{i}_update.pkl"
            print(command)
            os.system(command)

def model_difference():
    model1_path = "/home/ubuntu/Documents/jiarui/experiments/atk_pipeline/unharden_rep_pipeline_unhard_portions/unharden_portion0.4/before_replace/weights"
    model2_path = "/home/ubuntu/Documents/jiarui/experiments/atk_pipeline/unharden_rep_pipeline_unhard_portions/unharden_portion0.4/unharden/weights"
    
    model1_path = "/home/ubuntu/Documents/jiarui/experiments/atk_pipeline/fixedCode/check_pgd/def_trimmed_mean_weight_0.0/"
    model2_path = "/home/ubuntu/Documents/jiarui/experiments/atk_pipeline/fixedCode/check_pgd/def_trimmed_mean_weight_0.0/"

    model1 = torch.load(f"{model1_path}/chkpts_0.pt")
    model2 = torch.load(f"{model2_path}/chkpts_0.pt")

    for key in model1:
        if model1[key].data.dtype == torch.float32:
            print(key, (model1[key] - model2[key]).norm())
        else:
            print(key, (model1[key] - model2[key]).sum())

def cal_correlation():
    def model_weighted_avg(dict1, dict2, alpha=0.5):
        combined_dict = {}
    
        for key1, value1 in dict1.items():
            if key1 in dict2:
                value2 = dict2[key1]
                combined_dict[key1] = alpha * value1 + (1 - alpha) * value2
            else:
                # Handle the case where the key is not present in both dictionaries
                combined_dict[key1] = value1
        return combined_dict
    
    # benign_model_path = "/home/ubuntu/Documents/jiarui/experiments/atk_pipeline/fixedCode/unharden_pip_unharden_portions/unharden_0.4/before_replace/weights"
    benign_model_path = "/home/ubuntu/Documents/jiarui/experiments/NeurlPS_workshop/unharden_FAT_no_def_cifar10/def_None_atk_client_5_atk_round_1/before_replace/weights"
    atk_model_path = "/home/ubuntu/Documents/jiarui/experiments/atk_pipeline/fixedCode/unharden_pip_unharden_portions/unharden_0.4/unharden/weights"
    # atk_model_path = "/home/ubuntu/Documents/jiarui/experiments/fedavg/gt_epoch200/weights/round_199"
    alphas = [0.9]

    for alpha in alphas:

        benign_model = torch.load(f"{benign_model_path}/chkpts_0.pt")
        target_model = torch.load(f"{atk_model_path}/chkpts_0.pt")

        atk_model = model_weighted_avg(benign_model, target_model, alpha=alpha)

        diffs_dict = {}
        for key in benign_model:
            if benign_model[key].data.dtype == torch.float32:
                diffs_dict[key] = benign_model[key] - target_model[key]
            else:
                diffs_dict[key] = benign_model[key]
        
        update_dump_path = f"/home/ubuntu/Documents/jiarui/experiments/improvement_231016/unharden_direction_2/def_None_atk_client_5_atk_round_1_alpha_{alpha}/dump"
        with open(f"{update_dump_path}/round0_update.pkl", 'rb') as file:
            load_updates = pickle.load(file)
        global_updates = load_updates[0]
        malicious_updates = load_updates[1]

        def flatten_parameters(model_dict):
            flattened_params = []
            for key, value in model_dict.items():
                if benign_model[key].data.dtype == torch.float32:
                    if isinstance(value, torch.Tensor):
                        value = value.cpu()
                        flattened_params.extend(value.flatten().numpy())
            return np.array(flattened_params)

        print("benign model")
        correlations = []
        for idx, update in enumerate(global_updates):
            # calculate the correlation
            update_flat = flatten_parameters(update)
            diffs_flat = flatten_parameters(diffs_dict)
            # normalize the update
            update_flat = update_flat / np.linalg.norm(update_flat)
            diffs_flat = diffs_flat / np.linalg.norm(diffs_flat)

            correlation = np.corrcoef(update_flat, diffs_flat)
            correlations.append(correlation[0,1])
            # print("", idx, " ", correlation[0,1])
        print("avg correlation: ", np.mean(correlations))
        print("std correlation: ", np.std(correlations))

        print("malicious updates")
        correlations = []
        for idx, update in enumerate(malicious_updates):
            # calculate the correlation
            update_flat = flatten_parameters(update)
            diffs_flat = flatten_parameters(diffs_dict)
            # normalize the update
            update_flat = update_flat / np.linalg.norm(update_flat)
            diffs_flat = diffs_flat / np.linalg.norm(diffs_flat)

            correlation = np.corrcoef(update_flat, diffs_flat)
            correlations.append(correlation[0,1])
            # print("", idx, " ", correlation[0,1])
        print("avg correlation: ", np.mean(correlations))
        print("std correlation: ", np.std(correlations))



# removed_indices()
pipeline_results()
# markdown_table3()
# experiments_loop()
# cal_correlation()

