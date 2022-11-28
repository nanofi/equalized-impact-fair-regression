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
def mse(ratios, s, z, y):
    mse = np.dot(ratios, np.mean((z - y)**2, axis=1))
    mean = np.dot(ratios, np.mean(y, axis=1))
    var = np.dot(ratios, np.mean((y - mean)**2, axis=1))
    mse /= var
    return mse

def main(data, shape_info, param):
    module = partial(model.SensitiveNet, depth=param['depth'], shared_depth=param['shared_depth'], hidden=param['hidden'])
    key = jax.random.PRNGKey(param['learning_seed'])

    print("Initialize learning state...", end='')
    opt_def = flax.optim.Adam(learning_rate=param['lr'], weight_decay=param['weight_decay'])
    feature_size = shape_info['num_features']
    num_groups = shape_info['num_groups']
    module = partial(module, num_groups=num_groups)
    key, state = util.create_train_state(key, module, opt_def, param['batch_size'], feature_size)
    print("done")

    loss_fun = mse
    p_step = jax.jit(partial(util.step_nonconstrained, module, loss_fun))
    evaluator = jax.jit(partial(util.loss_evaluator, module, loss_fun))

    print("Start training:")
    key, state = util.learning_loop_with_best(key, p_step, data, param['num_epoches'], state, evaluator)
    params = util.deterministic_params(state)

    util.save_model(param['model_path'], params)