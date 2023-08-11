"""Run Experiment pFedDef

This script runs a pFedDef training on the FedEM model.
"""
from utils.utils import *
from utils.constants import *
from utils.args import *
from run_experiment import *

from torch.utils.tensorboard import SummaryWriter

# Import General Libraries
import os
import pytz
import argparse
import torch
import copy
import pickle
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from models import *

# Import Transfer Attack
from transfer_attacks.Personalized_NN import *
from transfer_attacks.Params import *
from transfer_attacks.Transferer import *
from transfer_attacks.Args import *
from transfer_attacks.TA_utils import *

import numba


if __name__ == "__main__":
    newYorkTz = pytz.timezone("America/New_York")
    timeInNewYork = datetime.now(newYorkTz)
    currentTimeInNewYork = timeInNewYork.strftime("%H:%M:%S")
    print("The current time in New York is:", currentTimeInNewYork)

    exp_root_path = input("exp_root_path>>>>\n")
    # exp_root_path = "/home/ubuntu/Documents/jiarui/experiments/atk_pipeline/test"

    path_log = open(f"{exp_root_path}/path_log", mode="w")
    path_log.write(f"FedAvg\n")

    defense_mechanisms = ["krum_modelwise"]  # "bulyan"
    unharden_type = None # "proj", "avg"
    params = []
    for defense in defense_mechanisms:
        params.append(defense)

    exp_names = [f"def_{para}" for para in params]
    G_val = [0.4] * len(exp_names)

    # print all experiment names
    for exp_name in exp_names:
        print(exp_name)

    torch.manual_seed(42)

    for itt in range(len(exp_names)):
        print("running exp_name:", exp_names[itt])

        ## INPUT GROUP 2 - experiment macro parameters ##
        args_ = Args()
        args_.experiment = "cifar10"  # dataset name
        args_.method = "FedAvg_adv"  # Method of training
        args_.decentralized = False
        args_.sampling_rate = 1.0
        args_.input_dimension = None
        args_.output_dimension = None
        args_.n_learners = 1  # Number of hypotheses assumed in system
        args_.n_rounds = 150  # Number of rounds training takes place
        args_.bz = 128
        args_.local_steps = 1
        args_.lr_lambda = 0
        args_.lr = 0.03  # Learning rate
        args_.lr_scheduler = "multi_step"
        args_.log_freq = 20
        args_.device = "cuda"
        args_.optimizer = "sgd"
        args_.mu = 0
        args_.communication_probability = 0.1
        args_.q = 1
        args_.locally_tune_clients = False
        args_.seed = 1234
        args_.verbose = 1
        args_.logs_root = f"{exp_root_path}/{exp_names[itt]}/logs"
        args_.save_path = f"{exp_root_path}/{exp_names[itt]}"
        # args_.load_path = f"/home/ubuntu/Documents/jiarui/experiments/atk_pipeline/unharden_rep_pipeline/atk_start/weights"  # load the model from the 150 FAT epoch
        # args_.load_path = f"/home/ubuntu/Documents/jiarui/experiments/atk_pipeline/unharden_rep_pipeline_unhard_portions/unharden_portion0.4/before_replace/weights"  # load the model from the 150 FAT epoch
        args_.validation = False
        args_.aggregation_op = params[itt]
        args_.save_interval = 10
        args_.eval_train = True
        args_.synthetic_train_portion = None
        args_.reserve_size = None
        args_.data_portions = None
        args_.unharden_source = None
        args_.dump_path = f"{exp_root_path}/{exp_names[itt]}/dump"
        args_.num_clients = 40  # Number of clients to train with

        Q = 10  # ADV dataset update freq
        G = G_val[itt]  # Adversarial proportion aimed globally
        G_global = 0.4  # Global proportion of adversaries
        S = 0.05  # Threshold param for robustness propagation
        step_size = 0.01  # Attack step size
        K = 10  # Number of steps when generating adv examples
        eps = 0.1  # Projection magnitude
        ## END INPUT GROUP 2 ##

        # Generate the dummy values here
        aggregator, clients = dummy_aggregator(args_, args_.num_clients, random_sample=False)
        Ru, atk_params, num_h, Du = get_atk_params(args_, clients, args_.num_clients, K, eps)
        if "load_path" in args_:
            print(f"Loading model from {args_.load_path}")
            load_root = os.path.join(args_.load_path)
            aggregator.load_state(load_root)
            aggregator.update_clients()  # update the client's parameters immediatebly, since they should have an up-to-date consistent global model before training starts

        args_adv = copy.deepcopy(args_)
        args_adv.method = "unharden"
        args_adv.num_clients = 5
        args_adv.reserve_size = 3.0 # data sample size reserved for each client. 3.0 means 3 times the size of the original dataset at a given client
        args_adv.synthetic_train_portion = 1.0 # the portion of the synthetic data in proportion to the original dataset
        args_adv.unharden_source = "orig" # the source of the unharden data (orig, synthetic, or orig+synthetic)
        args_adv.data_portions = (0.0, 0.0, 0.0) # portions of orig, synthetic, and unharden data in final training dataset, sum smaller than 3.0 (orig, synthetic, or unharden)
        args_adv.aggregation_op = None

        adv_aggregator, adv_clients = dummy_aggregator(args_adv, args_adv.num_clients, random_sample=False)
        # args_adv.load_path = f"/home/ubuntu/Documents/jiarui/experiments/atk_pipeline/unharden_rep_pipeline/unharden/weights"
        # args_adv.load_path = f"/home/ubuntu/Documents/jiarui/experiments/atk_pipeline/unharden_rep_pipeline_unhard_portions/unharden_portion0.4/unharden/weights"
        if "load_path" in args_adv:
            print(f"Loading model from {args_adv.load_path}")
            load_root = os.path.join(args_adv.load_path)
            adv_aggregator.load_state(load_root)
            adv_aggregator.update_clients()

        args_adv.unharden_start_round = 150
        args_adv.atk_rounds = 1

        args_adv.adv_params = dict()
        args_adv.adv_params["Q"] = Q
        args_adv.adv_params["G"] = G
        args_adv.adv_params["S"] = S
        args_adv.adv_params["step_size"] = step_size
        (
            args_adv.adv_params["Ru"],
            args_adv.adv_params["atk_params"],
            args_adv.adv_params["num_h"],
            args_adv.adv_params["Du"],
        ) = get_atk_params(args_adv, adv_clients, args_adv.num_clients, K, eps)
        adv_aggregator.set_atk_params(args_adv.adv_params)

        args_adv.unharden_type = None
        args_adv.unharden_params = dict()
        args_adv.unharden_params["global_model"] = None
        args_adv.unharden_params["global_model_fraction"] = None
        args_adv.unharden_params["epsilon"] = None
        args_adv.unharden_params["norm_type"] = 2

        # path_log.write(f"{exp_root_path}/{exp_names[itt]}/atk_start/weights\n")
        path_log.write(f"{exp_root_path}/{exp_names[itt]}/unharden/weights\n")
        path_log.write(f"{exp_root_path}/{exp_names[itt]}/before_replace/weights\n")
        path_log.write(f"{exp_root_path}/{exp_names[itt]}/replace/weights\n")
        if args_.eval_train and args_adv.save_interval is not None:
            for i in range(0, args_.n_rounds, args_.save_interval):
                path_log.write(
                    f"{exp_root_path}/{exp_names[itt]}/FAT_train/weights/gt{i}\n"
                )
            # for i in range(
            #     args_adv.unharden_start_round + 5, args_.n_rounds, args_.save_interval
            # ):
            #     path_log.write(
            #         f"{exp_root_path}/{exp_names[itt]}/unharden_train/weights/gt{i}\n"
            #     )

        model_diff_log = []
        update_log = []
        
        # Train the model
        print("Training..")
        pbar = tqdm(total=args_.n_rounds)
        current_round = 0
        while current_round < args_.n_rounds:
            # The conditions here happens before the round starts
            if current_round == args_adv.unharden_start_round:
                # save the chkpt for unharden
                save_root = os.path.join(args_.save_path, "atk_start/weights")
                os.makedirs(save_root, exist_ok=True)
                aggregator.save_state(save_root)

                # load the chkpt for unharden
                print(f"Epoch {current_round} | Loading model from {save_root}")
                adv_aggregator.load_state(save_root)
                adv_aggregator.update_clients()

            if current_round == args_.n_rounds - args_adv.atk_rounds:
                save_root = os.path.join(args_.save_path, "before_replace/weights")
                print(
                    f"Epoch {current_round} | Saving model before replacement to {save_root}"
                )
                os.makedirs(save_root, exist_ok=True)
                aggregator.save_state(save_root)

                # save the unharden chkpt for replacement
                save_root = os.path.join(args_.save_path, "unharden/weights")
                print(f"Epoch {current_round} | Saving unhardened model to {save_root}")
                os.makedirs(save_root, exist_ok=True)
                adv_aggregator.save_state(save_root)

                for client_idx in range(args_adv.num_clients):
                    aggregator.clients[client_idx].turn_malicious(
                        adv_aggregator.best_replace_scale(),
                        "replacement",
                        args_.n_rounds - args_adv.atk_rounds,
                        os.path.join(save_root, f"chkpts_0.pt"),
                        global_model_fraction=0.0,
                    )

            # If statement catching every Q rounds -- update dataset
            if current_round != 0 and current_round % Q == 0:  #
                # Obtaining hypothesis information
                Whu = np.zeros([args_.num_clients, num_h])  # Hypothesis weight for each user
                for i in range(len(clients)):
                    # print("client", i)
                    temp_client = aggregator.clients[i]
                    hyp_weights = temp_client.learners_ensemble.learners_weights
                    Whu[i] = hyp_weights

                row_sums = Whu.sum(axis=1)
                Whu = Whu / row_sums[:, np.newaxis]
                Wh = np.sum(Whu, axis=0) / args_.num_clients

                # Solve for adversarial ratio at every client
                Fu = solve_proportions(G, args_.num_clients, num_h, Du, Whu, S, Ru, step_size)
                print("global agg Fu", Fu)

                # Assign proportion and attack params
                # Assign proportion and compute new dataset
                for i in range(len(clients)):
                    aggregator.clients[i].set_adv_params(Fu[i], atk_params)
                    aggregator.clients[i].update_advnn()
                    aggregator.clients[i].assign_advdataset()

            aggregator.mix()
            if (
                args_.save_interval is not None
                and current_round % args_.save_interval == 0
            ):
                save_root = os.path.join(
                    args_.save_path, f"FAT_train/weights/gt{current_round}"
                )
                os.makedirs(save_root, exist_ok=True)
                aggregator.save_state(save_root)

            # assume that the adversaries cannot finish the current round faster than the global FL clients
            if current_round >= args_adv.unharden_start_round:
                args_adv.unharden_params[
                    "global_model"
                ] = aggregator.global_learners_ensemble

                # if current_round == args_.n_rounds - args_adv.atk_rounds - 1:
                #     args_adv.unharden_type = unharden_type  # use weight projection for the round before last round, where the replacement happens
                # else:
                #     args_adv.unharden_type = None

                adv_aggregator.mix(args_adv.unharden_type, args_adv.unharden_params)

                model_diff_log.append(diff_dict(aggregator.global_learners_ensemble[0], adv_aggregator.global_learners_ensemble[0]))
                update_log.append((aggregator.client_updates_record, adv_aggregator.client_updates_record))

                if (
                    args_.save_interval is not None
                    and current_round % args_.save_interval == 0
                ):
                    save_root = os.path.join(
                        args_.save_path, f"unharden_train/weights/gt{current_round}"
                    )
                    os.makedirs(save_root, exist_ok=True)
                    adv_aggregator.save_state(save_root)

            if aggregator.c_round != current_round:
                pbar.update(1)
                current_round = aggregator.c_round

        if "save_path" in args_:
            save_root = os.path.join(args_.save_path, "replace/weights")

            os.makedirs(save_root, exist_ok=True)
            aggregator.save_state(save_root)

        save_arg_log(f_path=args_.logs_root, args=args_, exp_name="args")
        save_arg_log(f_path=args_adv.logs_root, args=args_adv, exp_name="args_adv")

        np.save(
            f"{exp_root_path}/{exp_names[itt]}/client_dist_to_prev_gt_in_each_round.npy",
            np.array(aggregator.client_dist_to_prev_gt_in_each_round),
        )
        with open(
            f"{exp_root_path}/{exp_names[itt]}/unharden_weight_dist_to_global_model.pkl",
            "wb",
        ) as f:
            pickle.dump(adv_aggregator.weight_dist_to_global_model, f)
        print(
            "dumped indices to {}".format(
                f"{exp_root_path}/{exp_names[itt]}/unharden_weight_dist_to_global_model.pkl"
            )
        )

        with open(
            f"{exp_root_path}/{exp_names[itt]}/model_diff_log.pkl",
            "wb",
        ) as f:
            pickle.dump(model_diff_log, f)
        with open(
            f"{exp_root_path}/{exp_names[itt]}/update_log.pkl",
            "wb",
        ) as f:
            pickle.dump(update_log, f)

        del args_, aggregator, clients
        torch.cuda.empty_cache()

    path_log.close()

    newYorkTz = pytz.timezone("America/New_York")
    timeInNewYork = datetime.now(newYorkTz)
    currentTimeInNewYork = timeInNewYork.strftime("%H:%M:%S")
    print("The current time in New York is:", currentTimeInNewYork)
