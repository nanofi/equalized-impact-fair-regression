""" 
This is an implimentation of the method in 
Jeremie Mary, Clement Calauzenes, Noureddine El Karoui. Fairness-Aware Learning for Continuous Attributes and Treatments. In Proc. of 36th ICML, PMLR 97, 2019.
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


@functools.partial(jax.jit, static_argnums=(0,))
def _joint_3(num_groups, ratios, Y, Z, S, damping=1e-10):
    n = Y.shape[0] * Y.shape[1]
    d = 2
    Y = (Y - np.dot(ratios, Y.mean(axis=1))) / np.sqrt(np.dot(ratios, Y.var(axis=1)))
    Z = (Z - np.dot(ratios, Z.mean(axis=1))) / np.sqrt(np.dot(ratios, Z.var(axis=1)))
    std = (n * (d + 2) / 4.) ** (-1. / (d + 4))

    nbins = int(min(50, 5. / std))
    Yc = np.linspace(-2.5, 2.5, nbins)
    Zc = np.linspace(-2.5, 2.5, nbins)

    dist = (Y[:, np.newaxis, np.newaxis, :] - Yc[np.newaxis, :, np.newaxis, np.newaxis])**2 + (Z[:, np.newaxis, np.newaxis, :] - Zc[np.newaxis, np.newaxis, :, np.newaxis])**2
    K = np.exp(-(dist / (std ** 2) / 2)) / np.sqrt(2 * np.pi) / std

    pdf = np.mean(K, axis=3)

    h2d = pdf + damping
    h2d /= h2d.sum()
    return h2d

@functools.partial(jax.jit, static_argnums=(0,))
def chi_2_cond(num_groups, ratios, Y, Z, S, damping = 1e-10):
    h2d = _joint_3(num_groups, ratios, Y, Z, S, damping=damping)
    marginal_yz = h2d.sum(axis=0)
    marginal_sy = h2d.sum(axis=2)
    Q = h2d / ((np.sqrt(marginal_yz[np.newaxis, :, :]) * np.sqrt(marginal_sy[:, :, np.newaxis])) + damping)
    return ((Q ** 2).sum(axis=(0,2)) - 1.)

@functools.partial(jax.jit, static_argnums=(0,))
def hdr_pen_mse(num_groups, ratios, s, z, y, eps=0.1):
    mse = np.dot(ratios, np.mean((z - y)**2, axis=1))
    mean = np.dot(ratios, np.mean(y, axis=1))
    var = np.dot(ratios, np.mean((y - mean)**2, axis=1))
    mse /= var
    pen = np.mean(chi_2_cond(num_groups, ratios, y, z, s))
    loss = (eps / (1-eps)) * mse + pen
    return loss

def mse(s, y, z):
    return np.mean((z - y)**2)

def main(data, shape_info, param):
    module = partial(model.SensitiveNet, depth=param['depth'], shared_depth=param['shared_depth'], hidden=param['hidden'])
    key = jax.random.PRNGKey(param['learning_seed'])

    print("Initialize learning state...", end='')
    inner_opt = flax.optim.Adam(learning_rate=param['lr'], weight_decay=param['weight_decay'])
    opt_def = opt.ClipGrad(inner_opt, clipsize=1.0)
    feature_size = shape_info['num_features']
    num_groups = shape_info['num_groups']
    module = partial(module, num_groups=num_groups)
    key, state = util.create_train_state(key, module, opt_def, param['batch_size'], feature_size)
    print("done")

    loss_fun = partial(hdr_pen_mse, num_groups, eps=param['eps'])
    p_step = jax.jit(partial(util.step_nonconstrained, module, loss_fun))
    evaluator = jax.jit(partial(util.loss_evaluator, module, loss_fun))

    print("Start training:")
    key, state = util.learning_loop_with_best(key, p_step, data, param['num_epoches'], state, evaluator)
    params = util.deterministic_params(state)

    util.save_model(param['model_path'], params)