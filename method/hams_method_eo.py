""" 
This is an implimentation of the method in 
Harikrishna Narasimhan, Andrew Cotter, Maya Gupta, Serena Wang. Pairwise Fairness for Ranking and Regression. In Proc. of AAAI2020, 2020.
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import functools
from functools import partial

import jax
import jax.numpy as np
import flax

import model
import util
import opt

jax.config.enable_omnistaging()

@jax.jit
def surrogate(D):
     return jax.nn.relu(D + 1.0)

@functools.partial(jax.jit, static_argnums=(0,))
def hams_loss(num_groups, ratios, s, z, y, eps=0.1):
    loss = np.dot(ratios, np.mean((z - y)**2, axis=1))
    mean = np.dot(ratios, np.mean(y, axis=1))
    var = np.dot(ratios, np.mean((y - mean)**2, axis=1))
    loss /= var
    #loss = 0
    av = []
    ag = []
    for i in range(num_groups):
        for j in range(num_groups):
            if i == j:
                continue
            Dij = z[i, :, np.newaxis] - z[j, np.newaxis, :]
            Eij = (y[i, :, np.newaxis] - y[j, np.newaxis, :] > 0).astype(np.int32)
            m = np.count_nonzero(Eij)
            vij = (Eij * (Dij > 0).astype(np.float32)).sum() / m
            gij = (Eij * surrogate(Dij)).sum() / m
            av.append(vij)
            ag.append(gij)
    av = np.asarray(av)
    ag = np.asarray(ag)
    v = (av[:, np.newaxis] - av[np.newaxis, :]).ravel() - eps
    g = np.hstack([loss, (ag[:, np.newaxis] - ag[np.newaxis, :]).ravel() - eps])
    return g, v

def main(data, shape_info, param):
    module = partial(model.SensitiveNet, depth=param['depth'], shared_depth=param['shared_depth'], hidden=param['hidden'])
    key = jax.random.PRNGKey(param['learning_seed'])

    print("Initialize learning state...", end='')
    feature_size = shape_info['num_features']
    num_groups = shape_info['num_groups']
    opt_def = flax.optim.Adam(learning_rate=param['lr'], weight_decay=param['weight_decay'])
    cons = (num_groups * (num_groups - 1))**2
    module = partial(module, num_groups=num_groups)
    key, state = util.create_constrained_train_state(key, module, opt_def, param['batch_size'], feature_size, cons)
    print("done")

    loss_fun = partial(hams_loss, num_groups, eps=param['eps'])
    p_step = jax.jit(partial(util.step_constrained, module, loss_fun))

    print("Start training:")
    key, state = util.learning_loop_constrained(key, p_step, data, param['num_holds'], param['num_epoches'], state)

    util.save_model(param['model_path'], state)