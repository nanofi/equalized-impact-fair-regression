#!/usr/bin/env python3

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import pickle
import importlib
import fire
from functools import partial

from sklearn.model_selection import train_test_split

import jax
import jax.numpy as np
import flax
import numpy as onp

import pandas as pd

import model as md
import util

@partial(jax.jit, static_argnums=(0,))
def predict(module, params, s, x):
    z = module().apply({'params': params}, s, x.astype(np.float32))
    return z

def dump_pred(module, num_groups, dataset, params):
    x, s, y = dataset
    z = predict(module, params, s, x)
    return s, y, z

def main(param_path):
    with open(param_path, "rb") as f:
        param = pickle.load(f)

    print(param)
    
    train_param_path = param['train_param']
    with open(train_param_path, "rb") as f:
        train_param = pickle.load(f)
    print(train_param)
    dataset_seed = train_param['dataset_seed']
    data_param = pd.read_pickle(train_param['dataset'])
    num_train = data_param['num_train']
    num_groups = data_param['num_groups']
    D = data_param['dataset']
    _, test = train_test_split(D, train_size=num_train, random_state=dataset_seed)

    Y = np.asarray(test[['outcome']].to_numpy()).ravel()
    S = np.asarray(test[['sensitive']].to_numpy()).ravel()
    X = np.asarray(test.drop(['outcome', 'sensitive'], axis=1).to_numpy())
    dataset = (X, S, Y)

    print("Initialize and load model...", end='')
    module = partial(md.SensitiveNet, depth=train_param['depth'], shared_depth=train_param['shared_depth'], hidden=train_param['hidden'], num_groups=num_groups)
    key = jax.random.PRNGKey(train_param['learning_seed'])
    key, k0 = jax.random.split(key)
    feature_size = X.shape[1]
    variables = util.init_model(k0, module, train_param['batch_size'], feature_size)
    model_state, params = variables.pop('params')
    state = util.load_model(train_param['model_path'], params)
    print("done")

    print("Dumping predictions...")
    dumps = []
    for p in state.params:
        s, y, z = dump_pred(module, num_groups, dataset, p)
        df = pd.DataFrame()
        df["s"] = onp.asarray(s)
        df["y"] = onp.asarray(y)
        df["z"] = onp.asarray(z)
        dumps.append(df)

    with open(param['result_path'], "wb") as f:
        out = {
            'train_param': train_param_path,
            'test_param': param_path,
            'dumps': dumps,
            'prob': onp.asarray(state.prob)
        }
        pickle.dump(out, f)

if __name__ == '__main__':
    fire.Fire(main)