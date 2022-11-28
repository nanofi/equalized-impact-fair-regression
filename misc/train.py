#!/usr/bin/env python3

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import pickle
import importlib
import fire

from sklearn.model_selection import train_test_split

import jax
import jax.numpy as np

import pandas as pd

import util

def main(param_path):
    with open(param_path, "rb") as f:
        param = pickle.load(f)

    print(param)
    
    dataset_seed = param['dataset_seed']
    data_param = pd.read_pickle(param['dataset'])
    num_train = data_param['num_train']
    num_groups = data_param['num_groups']
    D = data_param['dataset']
    train, test = train_test_split(D, train_size=num_train, random_state=dataset_seed)

    Y = np.asarray(train['outcome'].to_numpy())
    S = np.asarray(train['sensitive'].to_numpy())
    X = np.asarray(train.drop(['outcome', 'sensitive'], axis=1).to_numpy())

    data = util.equal_group_batch(X, S, Y, batch_size=param['batch_size'], num_groups=num_groups)
    shape_info = {
        'num_groups': data_param['num_groups'],
        'num_features': X.shape[1]
    }

    method_path = param['method']
    method = importlib.machinery.SourceFileLoader('method', method_path).load_module()

    method.main(data, shape_info, param)
    

if __name__ == '__main__':
    fire.Fire(main)