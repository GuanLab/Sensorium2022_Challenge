import os
import sys
sys.path.insert(0, os.path.abspath("./"))

import pickle
import torch
import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings('ignore')

from nnfabrik.builder import get_data, get_model
import MySubmission as submission
from sensorium import evaluate

datasets = ["pretrain_21067-10-18", "pretrain_23343-5-17", "pretrain_22846-10-16",
            "pretrain_23656-14-22", "pretrain_23964-4-22", "sensorium+_27204-5-13",
            "sensorium_26872-17-20"]
# as filenames, we'll select all 7 datasets
filenames = [f"../dataset/{dataset}" for dataset in datasets]

model_n = int(sys.argv[1])
dataset_fn = 'MyDataloader.static_loaders'
dataset_config = {'paths': filenames,
                 'normalize': True,
                 'include_behavior': False,
                 'include_eye_position': False,
                 'add_behavior_as_channels': False,
                 'batch_size': 128,
                 'scale': None,
                 }

dataloaders = get_data(dataset_fn, dataset_config)

model_fn = 'models.stacked_core_full_gauss_readout'
model_config = {'pad_input': False,
  'layers': 4,
  'input_kern': 9,
  'gamma_input': 6.3831,
  'gamma_readout': 0.0076,
  'hidden_kern': 7,
  'hidden_channels': 128,
  'depth_separable': True,
  'grid_mean_predictor': {'type': 'cortex',
   'input_dimensions': 2,
   'hidden_layers': 1,
   'hidden_features': 30,
   'final_tanh': True},
  'init_sigma': 0.1,
  'init_mu_range': 0.3,
  'gauss_type': 'full',
  'shifter': False,
  'stack': -1,
}

models = []
for i in range(1, model_n+1):
    model = get_model(model_fn=model_fn,
                    model_config=model_config,
                    dataloaders=dataloaders,
                    seed=314,)
    model.load_state_dict(torch.load(f"./model_checkpoints/generalization_model.{i}.pth"));
    model.eval();
    models.append(model)
    
performances = {}
for filename in filenames:
    data_key = filename.split("_")[1]
    submission.generate_ground_truth_file([filename], "./", "test")
    submission.generate_submission_file(models, 
                                        dataloaders,
                                        data_key=data_key, 
                                        path=f"./", 
                                        device="cuda", 
                                        tier="test")
    ground_truth_file = './ground_truth_file_test.csv'
    submission_file = './submission_file_live_test.csv'
    out = evaluate(submission_file, ground_truth_file, per_neuron=True)
    performances[data_key] = out
    
with open(f"performances_per_neuron.{model_n}.pkl", "wb") as f:
    pickle.dump(performances, f)
