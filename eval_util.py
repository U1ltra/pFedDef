
from utils.utils import *
from utils.constants import *
from utils.args import *
from run_experiment import * 

from torch.utils.tensorboard import SummaryWriter

# Import General Libraries
import os
import argparse
import torch
import copy
import pickle
import random
import numpy as np
import pandas as pd
from models import *

# Import Transfer Attack
from transfer_attacks.Personalized_NN import *
from transfer_attacks.Params import *
from transfer_attacks.Transferer import *
from transfer_attacks.Args import *
from transfer_attacks.TA_utils import *

import numba 
from datetime import datetime
import pytz
import time

if __name__ == "__main__":
    newYorkTz = pytz.timezone("America/New_York") 
    timeInNewYork = datetime.now(newYorkTz)
    currentTimeInNewYork = timeInNewYork.strftime("%H:%M:%S")

    print("The current time in New York is:", currentTimeInNewYork)
    
    ## INPUT GROUP 1 - experiment macro parameters ##
    lr_set = [0.5, 1, 3]
    exp_names = [f'rep_lr_{int(i)}' for i in lr_set]
    G_val = [0.4]*len(exp_names)
    n_learners = 1
    ## END INPUT GROUP 1 ##
    
    itt = "Parameter Norm Evaluation"
    print("running trial:", itt)
    
    ## INPUT GROUP 2 - experiment macro parameters ##
    args_ = Args()
    args_.experiment = "cifar10"      # dataset name
    args_.method = 'FedAvg'       # Method of training
    args_.decentralized = False
    args_.sampling_rate = 1.0
    args_.input_dimension = None
    args_.output_dimension = None
    args_.n_learners= n_learners      # Number of hypotheses assumed in system
    args_.n_rounds = 150              # Number of rounds training takes place
    args_.bz = 128
    args_.local_steps = 1
    args_.lr_lambda = 0
    args_.lr = 0.03                   # Learning rate
    args_.lr_scheduler = 'multi_step'
    args_.log_freq = 20
    args_.device = 'cuda'
    args_.optimizer = 'sgd'
    args_.mu = 0
    args_.communication_probability = 0.1
    args_.q = 1
    args_.locally_tune_clients = False
    args_.seed = 1234
    args_.verbose = 1
    args_.logs_root = f'/home/ubuntu/Documents/jiarui/experiments/eval_dummy/logs'
    args_.save_path = f'/home/ubuntu/Documents/jiarui/experiments/eval_dummy/weights'      # weight save path
    # args_.load_path = f'/home/ubuntu/Documents/jiarui/experiments/{args_.method}/{args_.experiment}/replace/replace_fail_1/weights'
    args_.validation = False

    num_clients = 40                  # Number of clients to train with
    num_classes = 10                  # Number of classes in the data set we are training with
    atk_count = 1

    aggregator, clients = dummy_aggregator(args_, num_clients)
    
    # ckp_before_rep = "/home/ubuntu/Documents/jiarui/experiments/FedAvg/cifar10/replace/replace_fail_1/weights"
    ckp_before_rep = "/home/ubuntu/Documents/jiarui/experiments/pFedDef/weights/cifar10/FedAvg_all_label_switch/pfeddef/"
    print(f"Loading model from {ckp_before_rep}")
    load_root = os.path.join(ckp_before_rep)
    aggregator.load_state(load_root)
    model_GT = copy.deepcopy(aggregator.global_learners_ensemble)

    ckp_to_eval = [
        "/home/ubuntu/Documents/jiarui/experiments/FedAvg/cifar10/replacement_one_atk/weights",
        "/home/ubuntu/Documents/jiarui/experiments/FedAvg/cifar10/replacement_one_round/weights",
        "/home/ubuntu/Documents/jiarui/experiments/FedAvg/cifar10/from_scratch_one_r/weights",
        "/home/ubuntu/Documents/jiarui/experiments/FedAvg/cifar10/rep_lr_0/weights",
        "/home/ubuntu/Documents/jiarui/experiments/FedAvg/cifar10/rep_lr_1/weights",
        "/home/ubuntu/Documents/jiarui/experiments/FedAvg/cifar10/rep_lr_3/weights",
        "/home/ubuntu/Documents/jiarui/experiments/FedAvg/cifar10/rep_lr_005/weights",
        "/home/ubuntu/Documents/jiarui/experiments/FedAvg/cifar10/rep_lr_008/weights",
        "/home/ubuntu/Documents/jiarui/experiments/FedAvg/cifar10/rep_lr_0010/weights",
    ]
    models_to_eval = []
    for i, path in enumerate(ckp_to_eval):
        load_root = os.path.join(path)
        aggregator.load_state(load_root)
        models_to_eval.append(copy.deepcopy(aggregator.global_learners_ensemble))

 
    distance = []
    for i, learners_ensemble in enumerate(models_to_eval):
        norms = []
        for learner_id, learner in enumerate(learners_ensemble):

            GT_state = model_GT[learner_id].model.state_dict(keep_vars=True)
            learner_state = learner.model.state_dict(keep_vars=True)

            for key in GT_state:
                if GT_state[key].data.dtype == torch.int64:
                    continue

                norm_res = torch.norm(
                        GT_state[key].data.clone() - learner_state[key].data.clone()
                    )
                norms.append(norm_res.item() / 1000)

                if torch.isnan(norm_res):
                    print(GT_state[key].data.clone())
                    print(learner_state[key].data.clone())
                    print(norms[-1])
                    time.sleep(2)

        distance.append(
            sum(norms)
        )

    for i, dist in enumerate(distance):
        print(f"Norm distance for {ckp_to_eval[i]}")
        print(f">>> {dist} K")
