# Import General Libraries
import os
import argparse
import torch
import copy
import pickle
import random
import numpy as np
import pandas as pd

import sys

# Import FedEM based Libraries
from utils.utils import *
from utils.constants import *
from utils.args import *
from run_experiment import *
from models import *

# Import Transfer Attack
from transfer_attacks.Personalized_NN import *
from transfer_attacks.Params import *
from transfer_attacks.Transferer import *
from transfer_attacks.Args import *
from transfer_attacks.TA_utils import *
from transfer_attacks.Boundary_Transferer import *


def load_path():
    base_path = input("Enter the root path where path_log file exists >>>")
    f_ptr = open(f"{base_path}/path_log", mode="r")
    paths = [i[:-1] for i in f_ptr]
    for i in paths:
        print(i)

    setting = paths[0]
    if setting == "FedEM":
        nL = 3
    else:
        nL = 1

    return paths, setting, nL


def init_aggregator(num_models=40):
    # Manually set argument parameters
    args_ = Args()
    args_.experiment = "cifar10"
    args_.method = setting
    args_.decentralized = False
    args_.sampling_rate = 1.0
    args_.input_dimension = None
    args_.output_dimension = None
    args_.n_learners = nL
    args_.n_rounds = 10
    args_.bz = 128
    args_.local_steps = 1
    args_.lr_lambda = 0
    args_.lr = 0.03
    args_.lr_scheduler = "multi_step"
    args_.log_freq = 10
    args_.device = "cuda"
    args_.optimizer = "sgd"
    args_.mu = 0
    args_.communication_probability = 0.1
    args_.q = 1
    args_.locally_tune_clients = False
    args_.seed = 1234
    args_.verbose = 1
    args_.save_path = "weights/cifar/dummy/"
    args_.validation = False
    args_.aggregation_op = None

    # Generate the dummy values here
    aggregator, clients = dummy_aggregator(args_, num_user=num_models)
    return aggregator, clients, args_


def get_dataloader(clients_adv_gen):
    data_x = []
    data_y = []

    for i in range(len(clients_adv_gen)):
        daniloader = clients_adv_gen[i].test_iterator
        for x, y, idx in daniloader.dataset:
            data_x.append(x)
            data_y.append(y)

    data_x = torch.stack(data_x)
    try:
        data_y = torch.stack(data_y)
    except:
        data_y = torch.FloatTensor(data_y)

    dataloader = Custom_Dataloader(data_x, data_y)

    return data_x, data_y, dataloader


def load_agg_state(f_path, aggregator, args_, setting):
    if setting == "FedAvg":
        if os.path.exists(f"{f_path}/weights"):
            root_path = f"{f_path}/weights"
        else:
            root_path = f_path

        args_.save_path = root_path
        aggregator.load_state(args_.save_path)

        hypotheses = aggregator.global_learners_ensemble.learners

        weights_h = []

        for h in hypotheses:
            weights_h += [h.model.state_dict()]

        weights = np.load(f"{root_path}/train_client_weights.npy")

        model_weights = []

        for i in range(num_models):
            model_weights += [weights[i]]

        models_test = []

        for w0 in model_weights:
            new_model = copy.deepcopy(hypotheses[0].model)
            new_model.eval()
            new_weight_dict = copy.deepcopy(weights_h[0])
            for key in weights_h[0]:
                new_weight_dict[key] = w0[0] * weights_h[0][key]
            new_model.load_state_dict(new_weight_dict)
            models_test += [new_model]
    else:
        raise NotImplementedError

    return models_test


def init_metrics(num_models):
    metrics = [
        "orig_acc_transfers",
        "orig_similarities",
        "adv_acc_transfers",
        "adv_similarities_target",
        "adv_similarities_untarget",
        "adv_target",
        "adv_miss",
        "metric_alignment",
        "ib_distance_legit",
        "ib_distance_adv",
    ]
    adv_dict = {}
    for metric in metrics:
        adv_dict[metric] = None

    metric_dicts = [copy.deepcopy(adv_dict) for i in range(num_models)]
    return metric_dicts


### Main Code
if __name__ == "__main__":
    paths, setting, nL = load_path()

    num_models = 40
    aggregator_test, clients_test, args_test = init_aggregator(num_models)
    aggregator_adv_gen, clients_adv_gen, args_adv_gen = init_aggregator(num_models)

    models_adv_gen_path = "/home/ubuntu/Documents/jiarui/experiments/FAT_progressive/transferADV/FedAvg_adv_external/weights/chkpts_249"
    models_adv_gen = load_agg_state(
        models_adv_gen_path, aggregator_adv_gen, args_adv_gen, setting
    )

    for f_path in paths[1:]:
        print(f"Working on {f_path}", flush=True)
        sys.stdout.flush()

        data_x, _, _ = get_dataloader(clients_adv_gen)

        # Import Model Weights
        np.set_printoptions(formatter={"float": lambda x: "{0:0.2f}".format(x)})

        models_test_0 = load_agg_state(f_path, aggregator_test, args_test, setting)

        # Here we will make a dictionary that will hold results
        logs_adv = init_metrics(num_models)

        # Run Measurements for both targetted and untargeted analysis
        new_num_models = len(models_test_0)
        victim_idxs = range(new_num_models)
        custom_batch_size = 500
        eps = 100

        test_pair = [False]  # add True for paired testing
        for flag in test_pair:
            print(f"eval path {f_path}")

            for adv_idx in victim_idxs:
                print("\t Adv idx:", adv_idx, flush=True)

                dataloader = load_client_data(
                    clients=clients_adv_gen, c_id=adv_idx, mode="test", switch=flag
                )  # or test/train

                batch_size = min(custom_batch_size, dataloader.y_data.shape[0])


                t1 = Transferer(
                    models_list=models_test_0,
                    dataloader=dataloader,
                    adv_generate_model_list=models_adv_gen,
                )
                t1.generate_victims(victim_idxs)

                # Perform Attacks Targeted
                t1.atk_params = PGD_Params()
                t1.atk_params.set_params(
                    batch_size=batch_size,
                    iteration=10,
                    target=3,
                    x_val_min=torch.min(data_x),
                    x_val_max=torch.max(data_x),
                    step_size=0.01,
                    step_norm="inf",
                    eps=eps,
                    eps_norm=2,
                )

                t1.generate_advNN(adv_idx)
                t1.generate_xadv(atk_type="pgd")
                t1.send_to_victims(victim_idxs)

                # Log Performance
                logs_adv[adv_idx]["orig_acc_transfers"] = copy.deepcopy(
                    t1.orig_acc_transfers
                )
                logs_adv[adv_idx]["orig_similarities"] = copy.deepcopy(
                    t1.orig_similarities
                )
                logs_adv[adv_idx]["adv_acc_transfers"] = copy.deepcopy(
                    t1.adv_acc_transfers
                )
                logs_adv[adv_idx]["adv_similarities_target"] = copy.deepcopy(
                    t1.adv_similarities
                )
                logs_adv[adv_idx]["adv_target"] = copy.deepcopy(t1.adv_target_hit)

                # Miss attack Untargeted
                t1.atk_params.set_params(
                    batch_size=batch_size,
                    iteration=10,
                    target=-1,
                    x_val_min=torch.min(data_x),
                    x_val_max=torch.max(data_x),
                    step_size=0.01,
                    step_norm="inf",
                    eps=eps,
                    eps_norm=2,
                )
                t1.generate_xadv(atk_type="pgd")
                t1.send_to_victims(victim_idxs)
                logs_adv[adv_idx]["adv_miss"] = copy.deepcopy(t1.adv_acc_transfers)
                logs_adv[adv_idx]["adv_similarities_untarget"] = copy.deepcopy(
                    t1.adv_similarities
                )

            # Aggregate Results Across clients
            metrics = [
                "orig_acc_transfers",
                "orig_similarities",
                "adv_acc_transfers",
                "adv_similarities_target",
                "adv_similarities_untarget",
                "adv_target",
                "adv_miss",
            ]  # ,'metric_alignment']

            orig_acc = np.zeros([len(victim_idxs), len(victim_idxs)])
            orig_sim = np.zeros([len(victim_idxs), len(victim_idxs)])
            adv_acc = np.zeros([len(victim_idxs), len(victim_idxs)])
            adv_sim_target = np.zeros([len(victim_idxs), len(victim_idxs)])
            adv_sim_untarget = np.zeros([len(victim_idxs), len(victim_idxs)])
            adv_target = np.zeros([len(victim_idxs), len(victim_idxs)])
            adv_miss = np.zeros([len(victim_idxs), len(victim_idxs)])

            for adv_idx in range(len(victim_idxs)):
                for victim in range(len(victim_idxs)):
                    orig_acc[adv_idx, victim] = logs_adv[victim_idxs[adv_idx]][
                        metrics[0]
                    ][victim_idxs[victim]].data.tolist()
                    orig_sim[adv_idx, victim] = logs_adv[victim_idxs[adv_idx]][
                        metrics[1]
                    ][victim_idxs[victim]].data.tolist()
                    adv_acc[adv_idx, victim] = logs_adv[victim_idxs[adv_idx]][
                        metrics[2]
                    ][victim_idxs[victim]].data.tolist()
                    adv_sim_target[adv_idx, victim] = logs_adv[victim_idxs[adv_idx]][
                        metrics[3]
                    ][victim_idxs[victim]].data.tolist()
                    adv_sim_untarget[adv_idx, victim] = logs_adv[victim_idxs[adv_idx]][
                        metrics[4]
                    ][victim_idxs[victim]].data.tolist()
                    adv_target[adv_idx, victim] = logs_adv[victim_idxs[adv_idx]][
                        metrics[5]
                    ][victim_idxs[victim]].data.tolist()
                    adv_miss[adv_idx, victim] = logs_adv[victim_idxs[adv_idx]][
                        metrics[6]
                    ][victim_idxs[victim]].data.tolist()

            store_eval_path = f"{f_path}/eval"
            if not os.path.exists(store_eval_path):
                os.makedirs(store_eval_path)
            if flag:
                np.save(f"{store_eval_path}/pair_acc.npy", orig_acc)
                np.save(f"{store_eval_path}/pair_adv_acc.npy", adv_acc)
            else:
                np.save(f"{store_eval_path}/all_acc_external_epsilon_LL.npy", orig_acc)
                np.save(f"{store_eval_path}/all_adv_acc_external_epsilon_LL.npy", adv_acc)
