
from typing import Any
import jax
import jax.numpy as np
import flax

@flax.struct.dataclass
class _SelectBestParams:
    inner: Any
    evaluator: Any

class SelectBest(flax.optim.OptimizerDef):
    def __init__(self, inner_opt, evaluator):
        hps = _SelectBestParams(inner_opt.hyper_params, evaluator)
        super().__init__(hps)
        self.inner_opt = inner_opt
    
    def update_hyper_params(self, **hyper_param_overrides):
        evaluator = hyper_param_overrides.pop('evaluator', self.hyper_params.clipsize)
        inner = self.inner_opt.update_hyper_params(**hyper_param_overrides)
        return self.hyper_params.replace(inner=inner, evaluator=clipsize)
    def init_state(self, params):
        return self.inner_opt.init_state(params)
    def apply_gradient(self, hyper_params, params, state, grads):
        new_params, new_state = self.inner_opt.apply_gradient(hyper_params.inner, params, state, grads)
        return new_params, new_state
    
