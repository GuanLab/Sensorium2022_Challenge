import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import scipy
from scipy.interpolate import griddata, make_interp_spline


def get_corrleation(array1, array2):
    return np.corrcoef(array1, array2)[0, 1]


def get_distance(x, y):
    return np.sqrt(np.sum((x - y) ** 2))


def get_grids(merged_pos_preds, target_coords, data_keys, image_ids):
    # distances = [get_distance(coord, target_coords[0]) for coord in target_coords]
    # target_coords = target_coords[np.argsort(distances)]
    
    grids_keepNaN = {}
    for data_key in data_keys:
        grids_keepNaN[data_key] = {}
        sub_pos_preds = merged_pos_preds[merged_pos_preds["dataset"] == data_key]
        origin_coords = sub_pos_preds[["X", "Y", "Z"]]
        
        for image_id in image_ids:
            origin_values = sub_pos_preds[int(image_id)]
            target_values = griddata(origin_coords.to_numpy(), origin_values.to_numpy(), target_coords, method="linear")
            
            interpolations = np.concatenate([target_coords, target_values.reshape(-1, 1)], axis=1)
            
            grids_keepNaN[data_key][image_id] = interpolations

    return grids_keepNaN


def get_corrs_alongZ(preds, data_keys, targetX, targetY):
    all_result = []
    for data_key in data_keys:
        test = preds[data_key]
        zs = np.unique(test[:, 2])
        result = pd.DataFrame(test[test[:, 2] == zs[0], :3], columns=["X", "Y", "Z"])[["X", "Y"]]

        corrs_to_array1 = np.zeros((25*25, len(zs)))
        for i, z in enumerate(zs):
            array1 = test[(test[:, 2] == z) & (test[:, 1] == targetY) & (test[:, 0] == targetX), 3:]
            corrs_to_array1[:, i] = [get_corrleation(array1, array2) for array2 in test[(test[:, 2] == z), 3:]]
        
        corrs_to_array1 = np.nanmean(corrs_to_array1, axis=1)
        result["corrs"] = corrs_to_array1
        result = result.pivot(index="Y", columns="X", values="corrs")
        the_index = np.round(result.index, 4)
        the_column = np.round(result.columns, 4)
        all_result.append(result.values)
    
    all_result_mean = pd.DataFrame(np.nanmean(np.array(all_result), axis=0), index=the_index, columns=the_column)
    return all_result_mean


def get_corrs_alongZ_cross_brain(preds, data_keys, targetX, targetY):
    all_result = []
    for ref_key in data_keys:
        ref_test = preds[ref_key]
        
        for data_key in data_keys:
            if data_key == ref_key:
                continue
            
            test = preds[data_key]
            zs = np.unique(test[:, 2])
            result = pd.DataFrame(test[test[:, 2] == zs[0], :3], columns=["X", "Y", "Z"])[["X", "Y"]]

            corrs_to_array1 = np.zeros((25*25, len(zs)))
            for i, z in enumerate(zs):
                array1 = ref_test[(ref_test[:, 2] == z) & (ref_test[:, 1] == targetY) & (ref_test[:, 0] == targetX), 3:]
                corrs_to_array1[:, i] = [get_corrleation(array1, array2) for array2 in test[(test[:, 2] == z), 3:]]
            
            corrs_to_array1 = np.nanmean(corrs_to_array1, axis=1)
            result["corrs"] = corrs_to_array1
            result = result.pivot(index="Y", columns="X", values="corrs")
            the_index = np.round(result.index, 4)
            the_column = np.round(result.columns, 4)
            all_result.append(result.values)
    
    all_result_mean = pd.DataFrame(np.nanmean(np.array(all_result), axis=0), index=the_index, columns=the_column)
    return all_result_mean


def get_corrs_alongY(preds, data_keys, targetX, targetZ):
    all_result = []
    for data_key in data_keys:
        test = preds[data_key]
        ys = np.unique(test[:, 1])
        result = pd.DataFrame(test[test[:, 1] == ys[0], :3], columns=["X", "Y", "Z"])[["X", "Z"]]

        corrs_to_array1 = np.zeros((25*10, len(ys)))
        for i, y in enumerate(ys):
            array1 = test[(test[:, 2] == targetZ) & (test[:, 1] == y) & (test[:, 0] == targetX), 3:]
            corrs_to_array1[:, i] = [get_corrleation(array1, array2) for array2 in test[(test[:, 1] == y), 3:]]
        
        corrs_to_array1 = np.nanmean(corrs_to_array1, axis=1)
        result["corrs"] = corrs_to_array1
        result = result.pivot(index="Z", columns="X", values="corrs")
        the_index = np.round(result.index, 4)
        the_column = np.round(result.columns, 4)
        all_result.append(result.values)
    
    all_result_mean = pd.DataFrame(np.nanmean(np.array(all_result), axis=0), index=the_index, columns=the_column)
    return all_result_mean


def get_corrs_alongY_cross_brain(preds, data_keys, targetX, targetZ):
    all_result = []
    for ref_key in data_keys:
        ref_test = preds[ref_key]
    
        for data_key in data_keys:
            if data_key == ref_key:
                continue
            
            test = preds[data_key]
            ys = np.unique(test[:, 1])
            result = pd.DataFrame(test[test[:, 1] == ys[0], :3], columns=["X", "Y", "Z"])[["X", "Z"]]

            corrs_to_array1 = np.zeros((25*10, len(ys)))
            for i, y in enumerate(ys):
                array1 = ref_test[(ref_test[:, 2] == targetZ) & (ref_test[:, 1] == y) & (ref_test[:, 0] == targetX), 3:]
                corrs_to_array1[:, i] = [get_corrleation(array1, array2) for array2 in test[(test[:, 1] == y), 3:]]
            
            corrs_to_array1 = np.nanmean(corrs_to_array1, axis=1)
            result["corrs"] = corrs_to_array1
            result = result.pivot(index="Z", columns="X", values="corrs")
            the_index = np.round(result.index, 4)
            the_column = np.round(result.columns, 4)
            all_result.append(result.values)
    
    all_result_mean = pd.DataFrame(np.nanmean(np.array(all_result), axis=0), index=the_index, columns=the_column)
    return all_result_mean


def get_corrs_alongX(preds, data_keys, targetY, targetZ):
    all_result = []
    for data_key in data_keys:
        test = preds[data_key]
        xs = np.unique(test[:, 0])
        result = pd.DataFrame(test[test[:, 0] == xs[0], :3], columns=["X", "Y", "Z"])[["Y", "Z"]]

        corrs_to_array1 = np.zeros((25*10, len(xs)))
        for i, x in enumerate(xs):
            array1 = test[(test[:, 2] == targetZ) & (test[:, 1] == targetY) & (test[:, 0] == x), 3:]
            corrs_to_array1[:, i] = [get_corrleation(array1, array2) for array2 in test[(test[:, 0] == x), 3:]]
        
        corrs_to_array1 = np.nanmean(corrs_to_array1, axis=1)
        result["corrs"] = corrs_to_array1
        result = result.pivot(index="Z", columns="Y", values="corrs")
        the_index = np.round(result.index, 4)
        the_column = np.round(result.columns, 4)
        all_result.append(result.values)
    
    all_result_mean = pd.DataFrame(np.nanmean(np.array(all_result), axis=0), index=the_index, columns=the_column)
    return all_result_mean


def get_corrs_alongX_cross_brain(preds, data_keys, targetY, targetZ):
    all_result = []
    for ref_key in data_keys:
        ref_test = preds[ref_key]
    
        for data_key in data_keys:
            if data_key == ref_key:
                continue
            
            test = preds[data_key]
            xs = np.unique(test[:, 0])
            result = pd.DataFrame(test[test[:, 0] == xs[0], :3], columns=["X", "Y", "Z"])[["Y", "Z"]]

            corrs_to_array1 = np.zeros((25*10, len(xs)))
            for i, x in enumerate(xs):
                array1 = ref_test[(ref_test[:, 2] == targetZ) & (ref_test[:, 1] == targetY) & (ref_test[:, 0] == x), 3:]
                corrs_to_array1[:, i] = [get_corrleation(array1, array2) for array2 in test[(test[:, 0] == x), 3:]]
            
            corrs_to_array1 = np.nanmean(corrs_to_array1, axis=1)
            result["corrs"] = corrs_to_array1
            result = result.pivot(index="Z", columns="Y", values="corrs")
            the_index = np.round(result.index, 4)
            the_column = np.round(result.columns, 4)
            all_result.append(result.values)
    
    all_result_mean = pd.DataFrame(np.nanmean(np.array(all_result), axis=0), index=the_index, columns=the_column)
    return all_result_mean


def plot_average_neuron_corr_distance(corr_matrix, figsize, remove,
                                      xlabel=None, ylabel=None, title=None, ylim=None):
    f, ax = plt.subplots(figsize=figsize)
    neuron_diff_corr = []
    for i, (keys_pair, the_matrix) in enumerate(corr_matrix.items()):
        cur_neuron_diff_corr = np.empty((6250,))
        cur_neuron_diff_corr[:] = np.nan
        for idx in range(len(the_matrix)):
            cur_neuron_diff_corr[idx] = (np.diag(the_matrix, k=idx).mean() + np.diag(the_matrix, k=-idx).mean()) / 2
        neuron_diff_corr.append(cur_neuron_diff_corr)

    neuron_diff_corr = np.array(neuron_diff_corr)
    neuron_diff_corr_mean = np.nanmean(neuron_diff_corr, axis=0)
    neuron_diff_corr_mean = neuron_diff_corr_mean[~np.isnan(neuron_diff_corr_mean)]

    neuron_diff_corr_quantiles = {}
    for q in [0.05, 0.95]:
        neuron_diff_corr_quantile = np.nanquantile(neuron_diff_corr, q, axis=0)
        neuron_diff_corr_quantiles["q"+str(int(q*100))] = neuron_diff_corr_quantile[~np.isnan(neuron_diff_corr_quantile)]
    data = {"distance": np.arange(len(neuron_diff_corr_mean)), 
            "corr_mean": neuron_diff_corr_mean}
    data.update(neuron_diff_corr_quantiles)
    neuron_diff_corr_mean = pd.DataFrame(data)
    neuron_diff_corr_mean = neuron_diff_corr_mean.iloc[:-remove]
    corr_score = scipy.stats.pearsonr(neuron_diff_corr_mean["distance"], neuron_diff_corr_mean["corr_mean"])

    ax.scatter(x="distance", y="corr_mean", data=neuron_diff_corr_mean, c="blue", s=1)
    x_smooth = np.linspace(neuron_diff_corr_mean["distance"].min(),
                       neuron_diff_corr_mean["distance"].max(), 1000)
    spl_q1 = make_interp_spline(neuron_diff_corr_mean["distance"], 
                                neuron_diff_corr_mean["q5"], k=3)
    spl_q2 = make_interp_spline(neuron_diff_corr_mean["distance"], 
                                neuron_diff_corr_mean["q95"], k=3)
    y_smooth_q1 = spl_q1(x_smooth)
    y_smooth_q2 = spl_q2(x_smooth)
    ax.fill_between(x_smooth, y_smooth_q1, y_smooth_q2, color = "g", alpha = .3)
    
    ax.set_title(f"{title} (r = {round(corr_score[0], 4)})", fontsize=18)
    ax.set_xlabel(xlabel, fontsize=18)
    ax.set_ylabel(ylabel, fontsize=18)
    ax.tick_params(labelsize=16)
    
    if ylim is not None:
        ax.set_ylim(*ylim)
    
    return f, ax


def plot_neuron_corr_distance(corr_matrix, figsize, remove, nrow, ncol,
                              xlabel=None, ylabel=None, ylim=None):
    f, ax = plt.subplots(nrow, ncol, figsize=figsize)

    bigax = f.add_subplot(111, frameon=False)
    bigax.spines['top'].set_color('none')
    bigax.spines['bottom'].set_color('none')
    bigax.spines['left'].set_color('none')
    bigax.spines['right'].set_color('none')
    bigax.tick_params(labelcolor="w", top=False, bottom=False, left=False, right=False, pad=15)
    bigax.set_xticks([0.0])
    bigax.set_yticks([0.0, 0.1])
    bigax.set_xlabel(xlabel, fontsize=20)
    bigax.set_ylabel(ylabel, fontsize=20)

    for i, (keys_pair, the_matrix) in enumerate(corr_matrix.items()):
        neuron_diff_corr_mean = []
        for idx in range(len(the_matrix)-remove):
            neuron_diff_corr_mean.append([idx, (np.diag(the_matrix, k=idx).mean() + np.diag(the_matrix, k=-idx).mean()) / 2])
        neuron_diff_corr_mean = pd.DataFrame(neuron_diff_corr_mean, columns=["distance", "corr_mean"])
        corr_score = scipy.stats.pearsonr(neuron_diff_corr_mean["distance"], neuron_diff_corr_mean["corr_mean"])
        ax[i//ncol][i%ncol].scatter(x="distance", y="corr_mean", data=neuron_diff_corr_mean, c="blue", s=1)
        if ylim is not None:
            ax[i//ncol][i%ncol].set_ylim(*ylim) 
        ax[i//ncol][i%ncol].set_title(f"{keys_pair}: {round(corr_score[0], 4)}", fontsize=12)
        ax[i//ncol][i%ncol].tick_params(labelsize=16)
        
    return f, ax


