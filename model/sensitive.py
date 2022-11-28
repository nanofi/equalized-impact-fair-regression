
from functools import partial
from typing import Any, Callable, Sequence, Tuple

import jax
import jax.numpy as np
import flax
from flax import linen as nn

PRNGKey = Any
Shape = Tuple[int]
Dtype = Any  # this could be a real type?
Array = Any

class SensitiveDense(nn.Module):
    features: int
    num_groups: int = 2
    use_bias: bool = True
    dtype: Dtype = np.float32
    kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = jax.nn.initializers.glorot_normal()
    bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = jax.nn.initializers.zeros
    precision: Any = None
    
    @nn.compact
    def __call__(self, s, x):
        x = np.asarray(x, self.dtype)
        kernel_shape = (self.num_groups, x.shape[-1], self.features)
        kernel = self.param('kernel', self.kernel_init, kernel_shape)
        kernel = np.asarray(kernel, self.dtype)
        y = jax.lax.dot_general(kernel[s, :, :], x,
                (((1,), (1,)), ((0,), (0,))),
                precision=self.precision)

        if self.use_bias:
            bias_shape = (self.num_groups, self.features)
            bias = self.param('bias', self.bias_init, bias_shape)
            bias = np.asarray(bias, self.dtype)
            y = y + bias[s, :]
        return y

class SensitiveNet(nn.Module):
    num_groups: int = 2
    activation: Callable[[Array], Array] = jax.nn.gelu
    depth: int = 3
    shared_depth: int = 3
    hidden: int = 1024
    kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = jax.nn.initializers.glorot_normal()
    bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = jax.nn.initializers.zeros
    precision: Any = None

    @nn.compact
    def __call__(self, s, x):
        dense = partial(nn.Dense, precision=self.precision, kernel_init=self.kernel_init, bias_init=self.bias_init)
        #norm = nn.BatchNorm
        #x = dense(self.hidden)(x)

        for _ in range(self.shared_depth):
            y = dense(self.hidden)(x)
            x = self.activation(y)
            #x = norm()(x)

        sensitive_dense = partial(SensitiveDense, num_groups=self.num_groups, precision=self.precision, kernel_init=self.kernel_init, bias_init=self.bias_init)

        for _ in range(self.depth):
            y = sensitive_dense(self.hidden)(s, x)
            x = self.activation(y)
            #x = norm()(x)

        y = sensitive_dense(1)(s, x)
        #y = dense(1)(x)
        o = np.squeeze(y)
        return o