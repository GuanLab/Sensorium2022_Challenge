
import sys
import torch
import numpy as np
import pandas as pd
import glob

import warnings
warnings.filterwarnings('ignore')

from nnfabrik.builder import get_data, get_model, get_trainer


datasets = ["pretrain_21067-10-18", "pretrain_23343-5-17", "pretrain_22846-10-16",
            "pretrain_23656-14-22", "pretrain_23964-4-22", "sensorium+_27204-5-13",
            "sensorium_26872-17-20"]
# as filenames, we'll select all 7 datasets
filenames = [f"../dataset/{dataset}" for dataset in datasets]

model_id = int(sys.argv[1])
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dataset_fn = 'MyDataloader.static_loaders'
dataset_config = {'paths': filenames,
                 'normalize': True,
                 'include_behavior': False,
                 'include_eye_position': False,
                 'add_behavior_as_channels': False,
                 'batch_size': 128,
                 'scale': None,
                 'model_id': model_id,
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

model = get_model(model_fn=model_fn,
                  model_config=model_config,
                  dataloaders=dataloaders,
                  seed=42,)
print(model)

trainer_fn = "sensorium.training.standard_trainer"

trainer_config = {'max_iter': 200,
                 'verbose': False,
                 'lr_decay_steps': 4,
                 'avg_loss': False,
                 'lr_init': 0.009,
                 }

trainer = get_trainer(trainer_fn=trainer_fn, 
                     trainer_config=trainer_config)

validation_score, trainer_output, state_dict = trainer(model, dataloaders, seed=42)

torch.save(model.state_dict(), f'./model_checkpoints/generalization_model.{model_id}.pth')
