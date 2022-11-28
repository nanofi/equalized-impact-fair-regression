import functools
from typing import Any, List

import jax
import jax.numpy as np
import flax

def grouped_batch(*args, batch_size=1000):
    args = list(args)
    size = args[0].shape[1]
    def _batched():
        start = 0
        while start <= size:
            end = min(start + batch_size, size)
            batched = [arg[:, start:end] for arg in args]
            yield tuple(batched)
            start += batch_size
    return _batched

class EqualGroupBatch(object):
    def __init__(self, *args, axis=1, num_groups=2, batch_size=128):
        self.args = args
        self.axis = axis
        self.num_groups = num_groups
        self.batch_size = batch_size
    
    def align_group(self, key):
        S = self.args[self.axis]
        num_groups = self.num_groups
        idx = np.arange(S.shape[0], dtype=np.int32)
        groups = []
        for s in range(num_groups):
            grouped = np.compress(S == s, idx, axis=0)
            groups.append(grouped)
        m = max([g.shape[0] for g in groups])
        for s in range(num_groups):
            grouped = groups[s]
            grouped = np.repeat(grouped, m // grouped.shape[0], axis=0)
            key, k0 = jax.random.split(key)
            grouped = jax.random.permutation(k0, grouped)
            rest = m - grouped.shape[0]
            if rest > 0:
                key, k0 = jax.random.split(key)
                idx = jax.random.choice(k0, grouped.shape[0], (rest,))
                grouped = np.concatenate([grouped, grouped[idx]])
            groups[s] = grouped
        aligned = []
        for arg in self.args:
            aligned.append(np.stack([arg[gx] for gx in groups], axis=0))
        _, counts = np.unique(S, return_counts=True)
        ratios = counts / S.shape[0]        
        return key, ratios, aligned

    def batched(self, key):
        key, ratios, args = self.align_group(key)
        return key, ratios, grouped_batch(*args, batch_size=self.batch_size)

def equal_group_batch(*args, axis=1, num_groups=2, batch_size=128):
    return EqualGroupBatch(*args, axis=axis, num_groups=num_groups, batch_size=batch_size)
