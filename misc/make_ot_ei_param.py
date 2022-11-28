#!/usr/bin/env python3

import os
import fire
import copy
import hashlib
import json
import pickle

import numpy as np

def main(target='./results/params/train/ot_ei/'):
    os.makedirs(target, exist_ok=True)

    model_path = "./results/models/ot_ei/"
    os.makedirs(model_path, exist_ok=True)
    log_path = "./logs/train/ot_ei/"
    os.makedirs(log_path, exist_ok=True)

    defaults = {
        'name': 'ot_ei',
        'method': './method/ot_method_ei.py',
        'learning_seed': 1,
        'depth': 3,
        'shared_depth': 3,
        'hidden': 64,
        'batch_size': 128,
        'lr': 1e-4,
        'weight_decay': 0.0,
        'skip': 10000,
        'alpha': 0.1,
        'rs': 0.01,
        'num_epoches': 1000,
    }
    eps_low = 0.01
    eps_high = 0.99

    datasets = [
        "../data/comm_crime/comm_crime.pd",
        "../data/nlsy79/nlsy79.pd"
    ]

    for dataset in datasets:
        for dataset_seed in range(10):
            for eps in np.linspace(eps_low, eps_high, num=10).tolist():
                param = copy.copy(defaults)
                param['dataset'] = dataset
                param['dataset_seed'] = dataset_seed
                param['eps'] = eps

                ident = hashlib.sha512(json.dumps(param, sort_keys=True).encode('utf-8')).hexdigest()
                param['model_path'] = os.path.join(model_path, "{}.model".format(ident))
                param['log_path'] = os.path.join(log_path, "{}.log".format(ident))

                outpath = os.path.join(target, "{}.param".format(ident))
                with open(outpath, "wb") as f:
                    pickle.dump(param, f)

if __name__ == '__main__':
    fire.Fire(main)