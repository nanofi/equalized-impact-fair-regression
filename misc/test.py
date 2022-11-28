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

@jax.jit
def mse(z, y):
    return np.mean((z - y)**2)

def unfair_impact(num_groups, s, z, y):
    num = min([np.count_nonzero(s == m) for m in range(num_groups)])
    u = y - z
    rhos = []
    for m1 in range(num_groups):
        for m2 in range(num_groups):
            if m1 == m2: continue
            u1 = u[np.where(s == m1)]
            idx1 = jax.random.permutation(jax.random.PRNGKey(m1), u1.shape[0])
            u1 = u1[idx1[:num]]
            u2 = u[np.where(s == m2)]
            idx2 = jax.random.permutation(jax.random.PRNGKey(m1), u2.shape[0])
            u2 = u2[idx2[:num]]

            u1 = np.sort(u1)
            u2 = np.sort(u2)

            wp = np.mean(np.maximum(0.0, u1 - u2)**2)
            wm = np.mean(np.maximum(0.0, u2 - u1)**2)
            rho = wp - wm
            rhos.append(rho)

    return np.array(rhos)

def unfair_odds(num_groups, s, z, y):
    av = []
    for m1 in range(num_groups):
        for m2 in range(num_groups):
            if m1 == m2: continue
            D = np.triu((z[np.where(s == m1), np.newaxis] - z[np.newaxis, np.where(s == m2)] > 0).astype(np.int32))
            E = (y[np.where(s == m1), np.newaxis] - y[np.newaxis, np.where(s == m2)] > 0).astype(np.int32)
            m = np.count_nonzero(E)
            V = (E * D).sum() / m
            av.append(V)
    av = np.asarray(av)
    return (av[:, np.newaxis] - av[np.newaxis, :]).ravel()

def joint3(num_groups, Y, Z, S, damping=1e-10):
    n = Y.shape[0]
    d = 2
    Y = (Y - Y.mean()) / Y.std()
    Z = (Z - Z.mean()) / Z.std()
    std = (n * (d + 2) / 4.) ** (-1. / (d + 4))

    nbins = int(min(50, 5. / std))
    Yc = np.linspace(-2.5, 2.5, nbins)
    Zc = np.linspace(-2.5, 2.5, nbins)

    dist = (Y[np.newaxis, np.newaxis, :] - Yc[:, np.newaxis, np.newaxis])**2 + (Z[np.newaxis, np.newaxis, :] - Zc[np.newaxis, :, np.newaxis])**2
    K = np.exp(-(dist / (std ** 2) / 2)) / np.sqrt(2 * np.pi) / std

    pdf = []
    for v in range(num_groups):
        nv = np.count_nonzero(S == v)
        pv = np.dot(K, (S == v).astype(np.float32)) / nv
        pdf.append(pv)
    pdf = np.stack(pdf)

    h2d = pdf + damping
    h2d /= h2d.sum()
    return h2d

def unfair_chi2(num_groups, s, z, y, damping = 1e-10):
    h2d = joint3(num_groups, y, z, s, damping=damping)
    marginal_yz = h2d.sum(axis=0)
    marginal_sy = h2d.sum(axis=2)
    Q = h2d / ((np.sqrt(marginal_yz[np.newaxis, :, :]) * np.sqrt(marginal_sy[:, :, np.newaxis])))
    return ((Q ** 2).sum(axis=(0,2)) - 1.)

def scores(module, num_groups, dataset, params):
    x, s, y = dataset
    z = predict(module, params, s, x)
    scores = {
        'mse': mse(z, y),
        'var_y': np.var(y),
        'var_z': np.var(z),
        'unfair_impact': unfair_impact(num_groups, s, z, y),
        'unfair_odds': unfair_odds(num_groups, s, z, y),
        'unfair_chi2': unfair_chi2(num_groups, s, z, y)
    }
    scores['r2'] = 1.0 - scores['mse']/scores['var_y']
    scores['unfair_n_impact'] = scores['unfair_impact'] / scores['var_y']
    return scores

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

    print("Evalating...")
    evals = {}
    for p in state.params:
        e = scores(module, num_groups, dataset, p)
        for k, v in e.items():
            if k not in evals:
                evals[k] = []
            evals[k].append(v)
    def exp(kv):
        k, x = kv
        if len(x) > 1:
            x = onp.asarray(np.dot(np.array(x).T, state.prob))
        else:
            x = onp.asarray(x[0])
        if not np.isscalar(x):
            x = np.amax(x)
        x = x.item()
        return (k, x)
    result = dict(map(exp, evals.items()))
    print(result)

    with open(param['result_path'], "wb") as f:
        out = {
            'train_param': train_param_path,
            'test_param': param_path,
            'result': result
        }
        pickle.dump(out, f)

if __name__ == '__main__':
    fire.Fire(main)