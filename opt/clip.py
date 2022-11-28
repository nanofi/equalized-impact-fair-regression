
from typing import Any
import jax
import jax.numpy as np
import flax

@flax.struct.dataclass
class _ClipGradHyperParams:
    inner: Any
    clipsize: Any

class ClipGrad(flax.optim.OptimizerDef):
    def __init__(self, inner_opt, clipsize=1.0):
        hps = _ClipGradHyperParams(inner_opt.hyper_params, clipsize)
        super().__init__(hps)
        self.inner_opt = inner_opt
    
    def update_hyper_params(self, **hyper_param_overrides):
        clipsize = hyper_param_overrides.pop('clipsize', self.hyper_params.clipsize)
        inner = self.inner_opt.update_hyper_params(**hyper_param_overrides)
        return self.hyper_params.replace(inner=inner, clipsize=clipsize)
    def init_state(self, params):
        return self.inner_opt.init_state(params)
    def apply_gradient(self, hyper_params, params, state, grads):
        clip = hyper_params.clipsize
        leaves, _ = jax.tree_flatten(grads)
        norm = np.sqrt(sum(np.vdot(x, x) for x in leaves))
        clip *= np.sqrt(sum(x.shape[0] for x in leaves))        
        normalize = lambda g: np.where(norm < clip, g, g * (clip / norm))
        grads = jax.tree_map(normalize, grads)
        new_params, new_state = self.inner_opt.apply_gradient(hyper_params.inner, params, state, grads)
        return new_params, new_state
    
