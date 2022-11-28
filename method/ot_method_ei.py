import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../lib'))
import functools
from functools import partial

import jax
import jax.numpy as np
import flax

import model
import util
import opt

jax.config.enable_omnistaging()

def impact(num_groups, z, y, rs=1.0):
    up = y - z
    ups = util.sort(up)
    rhos = []
    for i in range(num_groups):
        for j in range(num_groups):
            if i == j: continue
            dij = ups[i, :] - ups[j, :]
            rho = np.mean(np.maximum(dij, 0)**2 - np.minimum(dij, 0)**2)
            rhos.append(rho)
    return np.amax(np.asarray(rhos))

def ot_impact_pen_mse(num_groups, ratios, s, z, y, eps=0.1, rs=1.0):
    mse = np.dot(ratios, np.mean((z - y)**2, axis=1))
    pen = impact(num_groups, z, y, rs=rs)
    loss = eps * mse + (1-eps) * pen
    mean = np.dot(ratios, np.mean(y, axis=1))
    var = np.dot(ratios, np.mean((y - mean)**2, axis=1))
    loss /= var
    return loss

def main(data, shape_info, param):
    module = partial(model.SensitiveNet, depth=param['depth'], shared_depth=param['shared_depth'], hidden=param['hidden'])
    key = jax.random.PRNGKey(param['learning_seed'])

    print("Initialize learning state...", end='')
    opt_def = flax.optim.Adam(learning_rate=param['lr'], weight_decay=param['weight_decay'])
    num_epoches = param['num_epoches']
    feature_size = shape_info['num_features']
    num_groups = shape_info['num_groups']
    module = partial(module, num_groups=num_groups)
    key, state = util.create_train_state(key, module, opt_def, param['batch_size'], feature_size)
    print("done")

    loss_fun = partial(ot_impact_pen_mse, num_groups, eps=param['eps'], rs=param['rs'])
    p_step = jax.jit(partial(util.step_nonconstrained, module, loss_fun))
    evaluator = jax.jit(partial(util.loss_evaluator, module, loss_fun))

    print("Start training:")
    key, state = util.learning_loop_with_best(key, p_step, data, num_epoches, state, evaluator)
    params = util.deterministic_params(state)

    util.save_model(param['model_path'], params)