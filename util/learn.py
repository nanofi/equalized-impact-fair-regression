import sys
import time
import functools
from typing import Any, List
import struct

import jax
import jax.numpy as np
import flax
from flax import linen as nn
import numpy as onp

import opt

Params = Any

@flax.struct.dataclass
class TrainState:
    step: int
    optimizer: flax.optim.Optimizer
    model_state: Params

    def params(self):
        return self.optimizer.target

    def variables(self):
        return flax.core.freeze({'params': self.optimizer.target, **self.model_state})

@flax.struct.dataclass
class StochasticParams():
    prob: np.ndarray
    params: List[Params]

def deterministic_params(state: TrainState):
    return StochasticParams(None, [state.params()])

def save_model(path, params: StochasticParams):
    with open(path, "wb") as f:
        if params.prob is None:
            f.write(struct.pack("l", 0))
        else:
            f.write(struct.pack("l", params.prob.shape[0]))
            prob_bytes = params.prob.astype(onp.float32).tobytes()
            f.write(struct.pack("l", len(prob_bytes)))
            f.write(prob_bytes)
        for p in params.params:
            p_bytes = flax.serialization.to_bytes(p)
            f.write(struct.pack("l", len(p_bytes)))
            f.write(p_bytes)

def load_model(path, template):
    with open(path, "rb") as f:
        size = struct.unpack("l", f.read(8))[0]
        if size > 0:
            prob_size = struct.unpack("l", f.read(8))[0]
            prob_buff = f.read(prob_size)
            prob = onp.frombuffer(prob_buff, dtype=onp.float32)
        else:
            prob = None
        params = []
        for _ in range(max(1, size)):
            p_size = struct.unpack("l", f.read(8))[0]
            p_buff = f.read(p_size)
            p = flax.serialization.from_bytes(template, p_buff)
            params.append(p)
        return StochasticParams(prob, params)


def init_model(key, module, batch_size, feature_size):
    s = np.zeros((batch_size,), dtype=np.int32)
    x = np.zeros((batch_size, feature_size), dtype=np.float32)
    return module().init(key, s, x)

def create_train_state(key, module, opt_def, batch_size, feature_size):
    key, k0 = jax.random.split(key)
    variables = init_model(k0, module, batch_size, feature_size)
    model_state, params = variables.pop('params')
    del variables
    optimizer = opt_def.create(params)
    del params
    state = TrainState(step=0, optimizer=optimizer, model_state=model_state)
    del optimizer, model_state
    return key, state


def create_constrained_train_state(key, module, opt_def, batch_size, feature_size, cons):
    key, k0 = jax.random.split(key)
    variables = init_model(k0, module, batch_size, feature_size)
    model_state, params = variables.pop('params')
    del variables
    target = opt.cons.make_con_target(params, cons)
    del params
    optimizer = opt_def.create(target)
    del target
    state = TrainState(step=0, optimizer=optimizer, model_state=model_state)
    del optimizer, model_state
    return key, state

def loss_evaluator(module, loss_fun, state, ratios, data):
    x, s, y = data
    num_features = x.shape[-1]
    num_batch = x.shape[1]
    mutable = list(state.model_state.keys())
    mutable.append("_") # force to return updated variables
    def loss(params):
        z, model_state = module().apply({'params': params, **state.model_state}, s.reshape((-1,)), x.astype(np.float32).reshape((-1, num_features)), mutable=mutable)
        z = z.reshape((-1, num_batch))
        loss = loss_fun(ratios, s, z, y)
        return loss
    return loss(state.optimizer.target)

#@functools.partial(jax.jit, static_argnums=(0,1))
def step_nonconstrained(module, loss_fun, state, ratios, batch):
    x, s, y = batch
    num_features = x.shape[-1]
    num_batch = x.shape[1]
    mutable = list(state.model_state.keys())
    mutable.append("_") # force to return updated variables
    def loss(params):
        z, model_state = module().apply({'params': params, **state.model_state}, s.reshape((-1,)), x.astype(np.float32).reshape((-1, num_features)), mutable=mutable)
        z = z.reshape((-1, num_batch))
        loss = loss_fun(ratios, s, z, y)
        return loss, (z, model_state)
    (val, (z, model_state)), grad = jax.value_and_grad(loss, has_aux=True)(state.optimizer.target)
    optimizer = state.optimizer.apply_gradient(grad)

    state = state.replace(step=state.step + 1, optimizer=optimizer, model_state=model_state)
    return state, val, y, z

#@functools.partial(jax.jit, static_argnums=(0,1))
def step_constrained(module, loss_fun, state, ratios, batch):
    x, s, y = batch
    num_features = x.shape[-1]
    num_batch = x.shape[1]
    mutable = list(state.model_state.keys())
    mutable.append("_") # force to return updated variables
    def loss(params):
        z, model_state = module().apply({'params': params, **state.model_state}, s.reshape((-1,)), x.astype(np.float32).reshape((-1, num_features)), mutable=mutable)
        z = z.reshape((-1, num_batch))
        g, v = loss_fun(ratios, s, z, y)
        return g, (z, v, model_state)
    prox, f_vjp, (z, value, model_state) = jax.vjp(loss, state.optimizer.target.params, has_aux=True)
    grad = opt.cons.make_grad(f_vjp, value, state.optimizer.target.markov)
    optimizer = state.optimizer.apply_gradient(grad)
    opt_target = opt.cons.project_markov(optimizer.target)
    optimizer = optimizer.replace(target = opt_target)

    state = state.replace(step=state.step + 1, optimizer=optimizer, model_state=model_state)
    return state, value, prox, y, z

@jax.jit
def mse(ratios, y, z):
    return np.dot(ratios, np.mean((z - y)**2, axis=1))

@jax.jit
def r2(ratios, y, z):
    mean = np.dot(ratios, np.mean(y, axis=1))
    var = np.dot(ratios, np.mean((y - mean)**2, axis=1))
    return 1.0 - mse(ratios, y, z)/var


def learning_loop(key, step, train, num_epoches, state):
    for epoch in range(num_epoches):
        start_time = time.time()
        key, ratios, batched = train.batched(key)
        for batch in batched():
            state, loss, y, z = step(state, ratios, batch)
        duration = time.time() - start_time
        print("Epoch", epoch, "Duration", duration, "[s]", "Loss", loss.item(), "MSE", mse(ratios, y, z).item(), "R2", r2(ratios, y, z).item())
        sys.stdout.flush()
    jax.random.normal(jax.random.PRNGKey(0), ()).block_until_ready()
    return key, state

def learning_loop_with_best(key, step, train, num_epoches, state, evaluator):
    best = state
    best_score = np.inf
    for epoch in range(num_epoches):
        start_time = time.time()
        key, ratios, batched = train.batched(key)
        for batch in batched():
            state, loss, y, z = step(state, ratios, batch)
        key, ratios, aligned = train.align_group(key)
        score = evaluator(state, ratios, aligned)
        if score < best_score:
            best = state
            best_score = score
        duration = time.time() - start_time
        print("Epoch", epoch, "Duration", duration, "[s]", "Loss", loss.item(), "MSE", mse(ratios, y, z).item(), "R2", r2(ratios, y, z).item(), "BestScore", best_score.item())
        sys.stdout.flush()
    jax.random.normal(jax.random.PRNGKey(0), ()).block_until_ready()
    return key, best

def update_average(previous, updated, n):
    averaged = jax.tree_util.tree_multimap(lambda a, b: (n * a + b) / (n + 1), previous, updated)
    return averaged

def learning_loop_average(key, step, train, num_epoches, state, skip=1000):
    params = state.optimizer.target
    for epoch in range(num_epoches):
        start_time = time.time()
        key, ratios, batched = train.batched(key)
        for batch in batched():
            if state.step >= skip:
                state, loss, y, z = step(state, ratios, batch)
                params = update_average(params, state.optimizer.target, state.step - skip - 1)
            else:
                state, loss, y, z = step(state, ratios, batch)
        duration = time.time() - start_time
        print("Epoch", epoch, "Duration", duration, "[s]", "Loss", loss.item(), "MSE", mse(ratios, y, z).item(), "R2", r2(ratios, y, z).item())
        sys.stdout.flush()
    jax.random.normal(jax.random.PRNGKey(0), ()).block_until_ready()
    return key, params

def learning_loop_constrained(key, step, train, num_holds, num_epoches, state):
    shrinkager = opt.cons.Shrinkager()
    for hold in range(num_holds):
        for epoch in range(num_epoches):
            start_time = time.time()
            key, ratios, batched = train.batched(key)
            for batch in batched():
                state, value, prox, y, z = step(state, ratios, batch)
            duration = time.time() - start_time
            l = opt.cons.lagurange(state.optimizer.target.markov)
            #onp.set_printoptions(linewidth=500)
            #print(onp.asarray(np.stack([l, np.hstack([0, value]), prox])))
            #onp.set_printoptions(linewidth=75)
            print("Hold", hold, "Epoch", epoch, "Duration", duration, "[s]", "Loss", prox[0].item(), "MSE", mse(ratios, y, z).item(), "R2", r2(ratios, y, z).item(), "Loss lag", l[0].item(), "Cons", np.amax(value).item(), "Prox", np.amax(prox[1:]).item(), "Cons lag", np.sum(l[1:]).item())
            sys.stdout.flush()
        shrinkager.append(state.optimizer.target.params, onp.asarray(np.hstack([prox[0], value])))
    jax.random.normal(jax.random.PRNGKey(0), ()).block_until_ready()

    shrinked = shrinkager.shrinkage(state.optimizer.target.params)
    shrinkager.close()
    return key, shrinked
