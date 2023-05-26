
import os
import sys
import numpy as np

datasets = ["pretrain_21067-10-18", "pretrain_23343-5-17", "pretrain_22846-10-16",
            "pretrain_23656-14-22", "pretrain_23964-4-22", "sensorium_26872-17-20",
            "sensorium+_27204-5-13"]

for i, dataset in enumerate(datasets):
    np.random.seed(i)
    
    tiers = np.load(f"./dataset/{dataset}/meta/trials/tiers.npy")
    training_num = np.sum(tiers == "train")
    validation_num = np.sum(tiers == "validation")
    training_validation_idxes = np.where((tiers == "train") | (tiers == "validation"))[0]
    
    for model in range(10):
        new_tiers = np.empty_like(tiers)
        new_tiers[tiers == "test"] = "test"
        new_tiers[tiers == "final_test"] = "final_test"
        
        np.random.shuffle(training_validation_idxes)
        training_idx = training_validation_idxes[:training_num]
        validation_idx = training_validation_idxes[training_num:]
        new_tiers[training_idx] = "train"
        new_tiers[validation_idx] = "validation"
        
        # assert for debug
        assert np.sum(new_tiers == "") == 0
        assert (np.where(tiers == "test")[0] == np.where(new_tiers == "test")).all()
        assert (np.where(tiers == "final_test")[0] == np.where(new_tiers == "final_test")).all()
        assert np.sum(new_tiers == "train") == training_num
        assert np.sum(new_tiers == "validation") == validation_num
        
        np.save(f"./dataset/{dataset}/meta/trials/tiers_model_{model+1}.npy", new_tiers)