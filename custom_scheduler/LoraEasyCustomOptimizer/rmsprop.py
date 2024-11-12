import torch
from torch.optim import Optimizer
from .utils import copy_stochastic_, quantize, dequantize, agc
import math
from torch.nn.functional import softplus
from typing import Optional

from bitsandbytes.functional import quantize_blockwise, dequantize_blockwise
from pytorch_optimizer.base.exception import NoSparseGradientError, ZeroParameterSizeError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.types import BETAS, CLOSURE, DEFAULTS, LOSS, PARAMETERS
from pytorch_optimizer.optimizer.gc import centralize_gradient
from pytorch_optimizer.optimizer.utils import normalize_gradient, unit_norm


class RMSProp(BaseOptimizer):
    r"""
    Arguments:
        params (iterable):
            Iterable of parameters to optimize or dicts defining
            parameter groups.
        lr (float):
            Learning rate parameter (default 0.0025)
        betas (float, optional):
            coefficient used for computing running averages of
            gradient's square (default: 0.999).
        eps (float):
            Term added to the denominator outside of the root operation to
            improve numerical stability. (default: 1e-8).
        weight_decay (float):
            Weight decay, i.e. a L2 penalty (default: 0).
        centralization (float):
            center model grad (default: 0).
    """

    def __init__(
        self,
        params: PARAMETERS,
        lr: float = 1e-3,
        betas: float = 0.9,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        weight_decouple: bool = True,
        fixed_decay: bool = False,
        centralization: float = 0.0,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_range(betas, 'betas', 0.0, 1.0, range_type='[]')
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_non_negative(eps, 'eps')
        self.validate_non_negative(centralization, 'centralization')

        defaults: DEFAULTS = {
            'lr':lr,
            'betas':betas,
            'weight_decay' : weight_decay,
            'weight_decouple' : weight_decouple,
            'fixed_decay' : fixed_decay,
            'eps':eps,
            'centralization':centralization,
        }

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'RMSProp'
    
    @torch.no_grad()
    def reset(self):
        for group in self.param_groups:
            group['step'] = 0

            for p in group['params']:
                state = self.state[p]

                # Exponential moving average of squared gradient values
                state["exp_avg_sq"] = torch.zeros_like(p)

    @torch.no_grad()
    def step(self, closure: CLOSURE = None) -> LOSS:
        loss: LOSS = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if 'step' in group:
                group['step'] += 1
            else:
                group['step'] = 1

            step_size = group["lr"]
            beta = group["beta"]
            weight_decay = group["weight_decay"]
            fixed_decay = group["fixed_decay"]
            weight_decouple = group["weight_decouple"]
            centralization = group["centralization"]

            bias_correction_sq: float = math.sqrt(self.debias(beta, group['step']))

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise NoSparseGradientError(str(self))

                state = self.state[p]

                p = p_fp32

                # State initialization
                if len(state) == 0:
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p)

                exp_avg_sq = state["exp_avg_sq"]

                # unpack
                if p.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.to(torch.float32)
                    p_fp32 = p.clone().to(torch.float32)
                    exp_avg_sq = exp_avg_sq.to(torch.float32)

                # center the gradient vector
                if centralization > 0.0 and grad.dim() > 1:
                    grad.sub_(
                        grad.mean(dim=tuple(range(1, grad.dim())), keepdim=True).mul_(centralization)
                    )

                # exp_avg_sq = ema + (1 - beta2) * grad ** 2
                exp_avg_sq.mul_(beta).addcmul_(grad, grad, value=1.0 - beta)

                # lr scaler + eps to prevent zero division
                # denom = exp_avg_sq.sqrt() + group['eps']
                de_nom = (exp_avg_sq.sqrt() / bias_correction_sq).add_(group["eps"])

                if weight_decouple:
                    # Perform stepweight decay
                    p_fp32.mul_(1.0 - (1.0 if fixed_decay else step_size) * weight_decay)
                elif weight_decay > 0.0:
                    grad.add_(p_fp32, alpha=weight_decay)

                # p = p - lr * grad / denom
                p_fp32.addcdiv_(grad, de_nom, value=-step_size)

                # pack
                if p.dtype in {torch.float16, torch.bfloat16}:
                    copy_stochastic_(state["exp_avg_sq"], exp_avg_sq)
                    copy_stochastic_(p, p_fp32)

        return loss
