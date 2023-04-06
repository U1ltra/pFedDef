"""Run Experiment label atk

This script runs training given that some clients have flipped labels to compromise model performance.
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
    
    # Input group 1
    exp_names = ['fedavg']
    exp_method = ['FedAvg']
    exp_num_learners = [1]
    # exp_lr = 0.01
    adv_mode = [False]
    
    for itt in range(len(exp_names)):    
        
        # Manually set argument parameters
        args_ = Args()
        args_.experiment = "cifar10"
        args_.method = exp_method[itt]
        args_.decentralized = False
        args_.sampling_rate = 1.0
        args_.input_dimension = None
        args_.output_dimension = None
        args_.n_learners= exp_num_learners[itt]
        args_.n_rounds = 200 # Reduced number of steps
        args_.bz = 128
        args_.local_steps = 1
        args_.lr_lambda = 0
        args_.lr =0.03
        args_.lr_scheduler = 'multi_step'
        args_.log_freq = 10
        args_.device = 'cuda'
        args_.optimizer = 'sgd'
        args_.mu = 0
        args_.communication_probability = 0.1
        args_.q = 1
        args_.locally_tune_clients = False
        args_.seed = 1234
        args_.verbose = 1
        args_.validation = False
        args_.save_freq = 10

        # Other Argument Parameters
        Q = 10 # update per round
        G = 0.15
        num_clients = 40 # 40 for cifar 10, 50 for cifar 100
        S = 0.05 # Threshold
        step_size = 0.01
        K = 10
        eps = 0.1
        prob = 0.8
        Ru = np.ones(num_clients)

        num_classes = 10 # Number of classes in the data set we are training with
        atk_count = 10   # Number of clients performing label swap attack
                
        
        print("running trial:", itt, "out of", len(exp_names)-1)
        
        args_.logs_root = f'/home/ubuntu/Documents/jiarui/experiments/{exp_names[itt]}/gt_epoch200/logs'
        args_.save_path = f'/home/ubuntu/Documents/jiarui/experiments/{exp_names[itt]}/gt_epoch200/weights'      # weight save path

        # Generate the dummy values here
        aggregator, clients = dummy_aggregator(args_, num_clients)
        

        # Train the model
        print("Training..")
        pbar = tqdm(total=args_.n_rounds)
        current_round = 0
        while current_round <= args_.n_rounds:
            if "save_path" in args_:    # store the 199 epoch checkpoint
                save_root = os.path.join(args_.save_path, f"round_{current_round}")

                os.makedirs(save_root, exist_ok=True)
                aggregator.save_state(save_root)

            aggregator.mix()
            

            if aggregator.c_round != current_round:
                pbar.update(1)
                current_round = aggregator.c_round

        if "save_path" in args_:
            save_root = os.path.join(args_.save_path)

            os.makedirs(save_root, exist_ok=True)
            aggregator.save_state(save_root)

        save_arg_log(f_path = args_.logs_root, args = args_)
            
        del aggregator, clients
        torch.cuda.empty_cache()
            