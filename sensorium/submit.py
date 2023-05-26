
import os
import torch
import glob

import warnings
warnings.filterwarnings('ignore')

from nnfabrik.builder import get_data, get_model, get_trainer
import MySubmission as submission


datasets = ["pretrain_21067-10-18", "pretrain_23343-5-17", "pretrain_22846-10-16",
            "pretrain_23656-14-22", "pretrain_23964-4-22", "sensorium+_27204-5-13",
            "sensorium_26872-17-20"]
data_keys = [key.split("_")[1] for key in datasets]

# as filenames, we'll select all 7 datasets
filenames = [f"../dataset/{dataset}" for dataset in datasets]

# model_id = int(sys.argv[1])
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
for model_id in range(1, 11):
    model = get_model(model_fn=model_fn,
                    model_config=model_config,
                    dataloaders=dataloaders,
                    seed=42,)
    model.load_state_dict(torch.load(f"./model_checkpoints/generalization_model.{model_id}.pth"));
    model.eval();
    models.append(model)


for data_key, filename in zip(data_keys[:5], filenames[:5]):
    if not os.path.exists(f"./preds_gt/{data_key}"):
        os.makedirs(f"./preds_gt/{data_key}")
        
    submission.generate_ground_truth_file([filename], f"./preds_gt/{data_key}", "test")
    
    
for data_key, filename in zip(data_keys[:5], filenames[:5]):
    if not os.path.exists(f"./preds_gt/{data_key}"):
        os.makedirs(f"./preds_gt/{data_key}")
        
    submission.generate_submission_file(models, 
                                        dataloaders,
                                        data_key=data_key, 
                                        path=f"./preds_gt/{data_key}", 
                                        device="cuda", tier="test")

