
import functools
from typing import Any, List

import jax
import jax.numpy as np


#@jax.custom_vjp
#def soft_sort(x, axis=-1):
#    return np.sort(x, axis=axis)

#def _soft_sort_fwd(x, axis=-1):


@jax.custom_vjp
def sort(x):
    return np.sort(x, axis=-1)

def _sort_fwd(x):
    perm = np.argsort(x)
    return np.take_along_axis(x, perm, -1), perm
def _sort_bwd(perm, g):
    return (np.take_along_axis(g, perm, -1), )
sort.defvjp(_sort_fwd, _sort_bwd)

