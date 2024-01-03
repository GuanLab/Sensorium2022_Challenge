# reference: https://github.com/bryanlimy/V1T

import sys
import torch
import random
import pickle
import numpy as np
import scipy.optimize as opt
from tqdm import tqdm
from nnfabrik.builder import get_data, get_model
from torch.utils.data import TensorDataset, DataLoader
from einops import rearrange, einsum
from neuralpredictors.training import eval_state, device_state

import warnings
warnings.simplefilter("error", opt.OptimizeWarning)
IMAGE_SIZE = (1, 144, 256)


def generate_ds(num_samples: int):
    """Generate num_samples of white noise images from uniform distribution
    Return:
        ds: DataLoader, the DataLoader object with the white noise images
        noise: np.ndarray, the array with the raw white noise images
    """
    noise = torch.rand((num_samples, *IMAGE_SIZE), dtype=torch.float32)
    # standardize images
    mean, std = torch.mean(noise), torch.std(noise)
    
    images = (noise - mean) / std  # 1st channel
    images_center = images - images.mean(dim=[2, 3], keepdim=True)  # 2nd channel
    x = torch.ones_like(images) * 0.5  # 3rd channel
    y = torch.ones_like(images) * 0.5  # 4th channel
    w = torch.ones_like(images)  # 5th channel
    h = torch.ones_like(images)  # 6th channel
    images = torch.cat([images, images_center, x, y, w, h], dim=1)
    
    # create DataLoader
    ds = DataLoader(TensorDataset(images), shuffle=False, batch_size=512, pin_memory=True)
    return ds, noise

def model_predictions(model, dataloader, data_key, device="cpu"):
    """
    computes model predictions for a given dataloader and a model
    Returns:
        output: responses as predicted by the network
    """
    output = torch.empty(0)
    for batch in tqdm(dataloader):
        images = batch[0]

        with torch.no_grad():
            with device_state(model, device):
                output = torch.cat(
                    (
                        output,
                        (model(images.to(device), data_key=data_key).detach().cpu()),
                    ),
                    dim=0,
                )
    return output

def inference(models, ds, data_key):
    predictions = model_predictions(models[0], ds, data_key, device="cuda")
    for model in models[1:]:
        cur_predictions = model_predictions(model, ds, data_key, device="cuda")
        predictions += cur_predictions
    predictions /= len(models)
    return predictions

def Gaussian2d(
    xy: np.ndarray,
    amplitude: float,
    xo: float,
    yo: float,
    sigma_x: float,
    sigma_y: float,
    theta: float,
    offset: float,
):

    x, y = xy
    xo = float(xo)
    yo = float(yo)
    a = (np.cos(theta) ** 2) / (2 * sigma_x**2) + (np.sin(theta) ** 2) / (
        2 * sigma_y**2
    )
    b = -(np.sin(2 * theta)) / (4 * sigma_x**2) + (np.sin(2 * theta)) / (
        4 * sigma_y**2
    )
    c = (np.sin(theta) ** 2) / (2 * sigma_x**2) + (np.cos(theta) ** 2) / (
        2 * sigma_y**2
    )
    g = offset + amplitude * np.exp(
        -(a * ((x - xo) ** 2) + 2 * b * (x - xo) * (y - yo) + c * ((y - yo) ** 2))
    )
    return g.ravel()

def fit_gaussian(num_units, aRFs: np.ndarray):
    """Fit 2D Gaussian to each aRFs using SciPy curve_fit

    Gaussian fit reference: https://stackoverflow.com/a/21566831

    Returns:
        popts: np.ndarray, a (num. units, 7) array with fitted parameters in
            [amplitude, center x, center y, sigma x, sigma y, theta, offset]
    """
    # standardize RFs and take absolute values to remove background noise
    mean = np.mean(aRFs, axis=(1, 2, 3))
    std = np.std(aRFs, axis=(1, 2, 3))
    broadcast = lambda a: rearrange(a, "n -> n 1 1 1")
    aRFs = (aRFs - broadcast(mean)) / broadcast(std)
    aRFs = np.abs(aRFs)

    height, width = aRFs.shape[2:]
    x, y = np.linspace(0, width - 1, width), np.linspace(0, height - 1, height)
    x, y = np.meshgrid(x, y)

    # numpy array of optimal parameters where rows are unit index
    popts = np.full(shape=(num_units, 7), fill_value=np.inf, dtype=np.float32)
    for i, unit in enumerate(tqdm(range(num_units), desc="Fit 2D Gaussian")):
        data = aRFs[unit][0]
        data = data.ravel()
        data_noisy = data + 0.2 * np.random.normal(size=data.shape)
        try:
            popt, pcov = opt.curve_fit(
                f=Gaussian2d,
                xdata=(x, y),
                ydata=data_noisy,
                p0=(3, width // 2, height // 2, 10, 10, 0, 10),
            )
            popts[unit] = popt
        except (RuntimeError, opt.OptimizeWarning):
            pass

    # filter out the last 5% of the results to eliminate poor fit
    num_drops = int(0.05 * len(popts))
    large_sigma_x = np.argsort(popts[:, 3])[-num_drops:]
    large_sigma_y = np.argsort(popts[:, 4])[-num_drops:]
    drop_units = np.unique(np.concatenate((large_sigma_x, large_sigma_y), axis=0))
    popts[drop_units] = np.nan

    print(
        f"sigma X: {np.nanmean(popts[:, 3]):.03f} \pm {np.nanstd(popts[:, 3]):.03f}\n"
        f"sigma Y: {np.nanmean(popts[:, 4]):.03f} \pm {np.nanstd(popts[:, 4]):.03f}"
    )

    return popts

def estimate_RFs(activations: torch.Tensor, noise: torch.Tensor):
    aRFs = einsum(activations, noise[:, :1, :, :], "b n, b c h w -> n c h w")
    return aRFs.numpy()


if __name__ == "__main__":
    data_key = str(sys.argv[1])
    filename = f"aRFs_{data_key}_ensemble.pkl"
    
    datasets = ["pretrain_21067-10-18", "pretrain_23343-5-17", "pretrain_22846-10-16",
                "pretrain_23656-14-22", "pretrain_23964-4-22", "sensorium+_27204-5-13",
                "sensorium_26872-17-20"]
    # as filenames, we'll select all 7 datasets
    filenames = [f"../dataset/{dataset}" for dataset in datasets]

    model_n = 10
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
        
    results = {}
    noise_all, activations_all = [], []
    for rep in range(10):
        torch.manual_seed(rep)
        torch.cuda.manual_seed(rep)
        np.random.seed(rep)
        random.seed(rep)
        ds, noise = generate_ds(num_samples=200000)
        activations = inference(models, ds, data_key=data_key)
        noise_all.append(noise)
        activations_all.append(activations)
        del ds, noise, activations
        
    noise_all = torch.cat(noise_all, dim=0)
    activations_all = torch.cat(activations_all, dim=0)
    aRFs = estimate_RFs(activations=activations_all, noise=noise_all)
    results["aRFs"] = aRFs
    
    results["popts"] = fit_gaussian(num_units=activations_all.shape[1], aRFs=aRFs)
    
    with open(filename, "wb") as file:
        pickle.dump(results, file)

    print(f"Saved aRFs and Gaussian fits to {filename}.")
    
