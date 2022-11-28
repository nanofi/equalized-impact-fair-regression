""" 
This is an implimentation of the method in 
Andrew Cotter, Heinrich Jiang, and Karthik Sridharan. Two-Player Games for Efficient Non-Convex Constrained Optimization. In: arXiv:1804.06500, 2018. https://arxiv.org/abs/1804.06500.
"""

import os
from typing import Any, List
import functools
import tempfile
import shutil
import jax
import jax.numpy as np
import flax
from flax import struct
import numpy as onp
import scipy as sp

import util


@struct.dataclass
class ConstrainedParams():
    markov: np.ndarray
    params: Any

def top_eigenvec(X, maxiter=50):
    v = np.ones(X.shape[0]) / np.sqrt(X.shape[0])
    for _ in range(maxiter):
        v = np.dot(X, v)
        v /= np.linalg.norm(v)
    return v

def lagurange(markov):
    l = top_eigenvec(np.exp(markov))
    l /= np.sum(l)
    return l

def make_con_target(params: Any, cons: int) -> ConstrainedParams:
    d = cons + 1
    markov = -np.ones((d,d)) * np.log(d)
    return ConstrainedParams(markov = markov, params = params)

def make_grad(g_vjp, c, markov):
    l = lagurange(markov)
    params = g_vjp(l)[0]
    gl = np.hstack([0, c])
    markov = -gl[:, np.newaxis] * l[np.newaxis, :]
    return ConstrainedParams(markov=markov, params=params)

def project_markov(con_params: ConstrainedParams, min_multi=1e-3):
    markov = con_params.markov
    markov = markov - jax.scipy.special.logsumexp(markov, axis=0)[np.newaxis, :]
    markov = np.maximum(markov, np.log(min_multi / markov.shape[0]))
    return con_params.replace(markov = markov)

class Shrinkager(object):

    def __init__(self):
        self.basedir = tempfile.mkdtemp(prefix="cons-opt-cotter-")
        self.iter = 0
        self.values = []
    
    def close(self):
        self.cleanup()

    def cleanup(self):
        shutil.rmtree(self.basedir)

    def append(self, params, value):
        params_path = os.path.join(self.basedir, "{}.model".format(self.iter))
        with open(params_path, 'wb') as f:
            f.write(flax.serialization.to_bytes(params))
        self.values.append(value)
        self.iter += 1

    def shrinkage(self, template, vio=0.0, th=1e-5):
        V = onp.stack(self.values)

        c = V[:, 0]
        Aub = V[:, 1:].T
        Aeq = onp.ones((1, c.shape[0]))
        bu = onp.ones((Aub.shape[0],)) * vio
        be = onp.ones((Aeq.shape[0],))
        res = sp.optimize.linprog(c, Aub, bu, Aeq, be, (0.0, 1.0), options={'lstsq': True, 'sym_pos':False, 'cholesky': False}) 

        idx = onp.where(res.x > th)[0]
        prob = res.x[idx]
        prob = onp.maximum(0.0, prob)
        prob /= onp.sum(prob)

        params = []
        for i in idx:
            state_path = os.path.join(self.basedir, "{}.model".format(i))
            with open(state_path, 'rb') as f:
                pi = flax.serialization.from_bytes(template, f.read())
                params.append(pi)

        return util.StochasticParams(prob, params)
