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
import pytz
from datetime import datetime

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

from Evaluation.eval_internal_influence import *

newYorkTz = pytz.timezone("America/New_York")
timeInNewYork = datetime.now(newYorkTz)
currentTimeInNewYork = timeInNewYork.strftime("%H:%M:%S")
print("The current time in New York is:", currentTimeInNewYork)

paths = [
    "FedAvg",
    "cifar10",
    "40",
    "/home/ubuntu/Documents/jiarui/experiments/NeurlPS_workshop/unharden_FAT_no_def_cifar10/def_None_atk_client_1_atk_round_1_reviewImprove/atk_start/weights",
]

for i in paths:
    print(i)

# Generating Empty Aggregator to be loaded 

setting = paths[0]
dataset = paths[1]
num_clients = int(paths[2])

if setting == 'FedEM':
    nL = 3
else:
    nL = 1

# Manually set argument parameters
args_ = Args()
args_.experiment = dataset
args_.method = setting
args_.decentralized = False
args_.sampling_rate = 1.0
args_.input_dimension = None
args_.output_dimension = None
args_.n_learners= nL
args_.n_rounds = 10
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
args_.save_path = 'weights/cifar/dummy/'
args_.validation = False
args_.aggregation_op = None
args_.synthetic_train_portion = None
args_.reserve_size = None 
args_.data_portions = None
args_.unharden_source = None
args_.dump_path = None

# Generate the dummy values here
aggregator, clients = dummy_aggregator(args_, num_user=num_clients)

for f_path in paths[3:]:
    print(f"Working on {f_path}")
    sys.stdout.flush()
    # Compiling Dataset from Clients
    # Combine Validation Data across all clients as test
    data_x = []
    data_y = []

    for i in range(len(clients)):
        daniloader = clients[i].test_iterator
        for (x,y,idx) in daniloader.dataset:
            data_x.append(x)
            data_y.append(y)

    data_x = torch.stack(data_x)
    try:
        data_y = torch.stack(data_y)        
    except:
        data_y = torch.FloatTensor(data_y) 
        
    dataloader = Custom_Dataloader(data_x, data_y)


    # Import Model Weights
    num_models = num_clients

    np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})

    if setting == 'FedAvg':
        
        if os.path.exists(f"{f_path}/weights"):
            root_path = f"{f_path}/weights"
        else:
            root_path = f"{f_path}"
        
        args_.save_path = root_path
        aggregator.load_state(args_.save_path)
        
        hypotheses = aggregator.global_learners_ensemble.learners

        weights_h = []

        for h in hypotheses:
            weights_h += [h.model.state_dict()]

        model = hypotheses[0].model

        # weights = np.load(f'{root_path}/train_client_weights.npy')
        # if weights.shape[0] != num_models:
        #     weights = np.ones((num_models, nL))
        
        # model_weights = []

        # for i in range(num_models):
        #     model_weights += [weights[i]]

        # models_test = []

        # for (w0) in model_weights:
        #     new_model = copy.deepcopy(hypotheses[0].model)
        #     new_model.eval()
        #     new_weight_dict = copy.deepcopy(weights_h[0])
        #     for key in weights_h[0]:
        #         new_weight_dict[key] = w0[0]*weights_h[0][key] 
        #     new_model.load_state_dict(new_weight_dict)
        #     models_test += [new_model]

    # Evaluate the model
    print("Evaluating the model")
    sys.stdout.flush()

    print("Model Architecture")
    print(model)
    sys.stdout.flush()

    for client_id, client in enumerate(clients):
        print(f"Client {client_id}")
        sys.stdout.flush()
        dataset_iterator = client.train_iterator

        evalInternalInfluence_ = evalInternalInfluence(model, dataset_iterator)
        evalInternalInfluence_.print_layer_names()
        evalInternalInfluence_.reset_influence()
        evalInternalInfluence_.eval_influence()
        evalInternalInfluence_.save_influence(f"{f_path}/influence_client_{client_id}")

        for layer in evalInternalInfluence_.model_layers():
            evalInternalInfluence_.print_influence(layer[0])
            print("")
            sys.stdout.flush()
            


