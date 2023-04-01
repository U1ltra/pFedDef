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


if __name__ == "__main__":
    
    ## INPUT GROUP 1 - experiment macro parameters ##
    exp_names = ['all_clients_all_switch']
    G_val = [0.4]
    n_learners = 1
    ## END INPUT GROUP 1 ##
    
    for itt in range(len(exp_names)):
        
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
        args_.logs_root = f'/home/ubuntu/Documents/jiarui/experiments/{args_.method}/{args_.experiment}/{exp_names[itt]}/logs'
        args_.save_path = f'/home/ubuntu/Documents/jiarui/experiments/{args_.method}/{args_.experiment}/{exp_names[itt]}/weights'      # weight save path
        args_.validation = False
        args_.atk_count = 40

        Q = 10                            # ADV dataset update freq
        G = G_val[itt]                    # Adversarial proportion aimed globally
        num_clients = 40                  # Number of clients to train with
        S = 0.05                          # Threshold param for robustness propagation
        step_size = 0.01                  # Attack step size
        K = 10                            # Number of steps when generating adv examples
        eps = 0.1                         # Projection magnitude 

        num_classes = 10                  # Number of classes in the data set we are training with
        atk_count = args_.atk_count
        ## END INPUT GROUP 2 ##
        

        # Randomized Parameters
        Ru = np.ones(num_clients)
        
        # Generate the dummy values here
        aggregator, clients = dummy_aggregator(args_, num_clients)

        # Perform label swapping attack for a set number of clients
        for i in range(atk_count):
            aggregator.clients[i].swap_dataset_labels(num_classes, switch_pair = False)


        # Train the model
        print("Training..")
        pbar = tqdm(total=args_.n_rounds)
        current_round = 0
        while current_round <= args_.n_rounds:

            aggregator.mix()

            if aggregator.c_round != current_round:
                pbar.update(1)
                current_round = aggregator.c_round

        if "save_path" in args_:
            save_root = os.path.join(args_.save_path)

            os.makedirs(save_root, exist_ok=True)
            aggregator.save_state(save_root)
        
        save_arg_log(f_path = args_.logs_root, args = args_)

        del args_, aggregator, clients
        torch.cuda.empty_cache()
            