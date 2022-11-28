
from typing import Any
import jax
import jax.numpy as np
import flax

def default_scheduler(k):
    return 1.0 / (1.0 + 5.0 * k / 1e5)

@flax.struct.dataclass
class _PathAveragingParams:
    inner: Any
    scheduler: Any
    alpha: Any

@flax.struct.dataclass
class _PathAveragingParamState:
    inner: Any
    avg_grad: Any

class PathAveraging(flax.optim.OptimizerDef):
    def __init__(self, inner_opt, scheduler=default_scheduler, alpha=0.1):
        hps = _PathAveragingParams(inner_opt.hyper_params, scheduler, alpha)
        super().__init__(hps)
        self.inner_opt = inner_opt
    
    def update_hyper_params(self, **hyper_param_overrides):
        kwargs = {}
        kwargs['scheduler'] = hyper_param_overrides.pop('scheduler', self.hyper_params.scheduler)
        kwargs['alpha'] = hyper_param_overrides.pop('alpha', self.hyper_params.alpha)
        inner = self.inner_opt.update_hyper_params(**hyper_param_overrides)
        return self.hyper_params.replace(inner=inner, **kwargs)

    def init_param_state(self, param):
        inner = self.inner_opt.init_param_state(param)
        return _PathAveragingParamState(inner, np.zeros_like(param))

    def apply_param_gradient(self, step, hyper_params, param, state, grad):
        scheduler = hyper_params.scheduler
        alpha = hyper_params.alpha
        tau = scheduler(step)
        ta = alpha * tau

        # calculate z
        z = state.avg_grad
        z = (1 - ta) * z + ta * grad

        # calculate y
        y, new_inner_state = self.inner_opt.apply_param_gradient(step, hyper_params.inner, param, state.inner, z)
        new_state = _PathAveragingParamState(new_inner_state, z)

        # calculate x
        new_param = (1 - tau) * param + tau * y

        return new_param, new_state
