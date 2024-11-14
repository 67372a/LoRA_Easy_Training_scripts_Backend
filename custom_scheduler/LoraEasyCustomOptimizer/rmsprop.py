import torch
from .utils import copy_stochastic_, quantize, dequantize, agc
import math
from typing import Optional

from pytorch_optimizer.base.exception import NoSparseGradientError, ZeroParameterSizeError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.types import BETAS, CLOSURE, DEFAULTS, LOSS, PARAMETERS


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
        eps2 (float):
            Term to multiple the RMS of the grad to calculate adaptive eps. (default: 0.01).
        eps_floor (float):
            Term to set a floor for the eps, to prevent NaNs. (default: 1e-8).
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
        eps2: float = 0.01,
        eps_floor: float = 1e-8,
        weight_decay: float = 0.0,
        weight_decouple: bool = True,
        fixed_decay: bool = False,
        centralization: float = 0.0,
        stable_decay: bool = False,
        clip: float = 0.0,
        clip_eps: float = 1e-8,
        adaptive_clipping: bool = False,
        adaptive_clip_eps: float = 1e-3,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_range(betas, 'betas', 0.0, 1.0, range_type='[]')
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_non_negative(eps, 'eps')
        self.validate_non_negative(centralization, 'centralization')
        self.validate_non_negative(clip, 'clip')
        self.validate_non_negative(adaptive_clip_eps, 'adaptive_clip_eps')
        self.validate_non_negative(clip_eps, 'clip_eps')
        self.validate_non_negative(eps2, 'eps2')
        self.validate_non_negative(eps_floor, 'eps_floor')

        defaults: DEFAULTS = {
            'lr':lr,
            'betas':betas,
            'weight_decay' : weight_decay,
            'weight_decouple' : weight_decouple,
            'fixed_decay' : fixed_decay,
            'eps':eps,
            'centralization':centralization,
            'stable_decay':stable_decay,
            'clip':clip,
            'clip_eps':clip_eps,
            'adaptive_clipping':adaptive_clipping,
            'adaptive_clip_eps':adaptive_clip_eps,
            'eps2':eps2,
            'eps_floor':eps_floor,
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

        param_size: int = 0
        exp_avg_sq_sum: float = 0.0

        for group in self.param_groups:
            beta = group['betas']

            if 'step' in group:
                group['step'] += 1
            else:
                group['step'] = 1
                
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise NoSparseGradientError(str(self))

                param_size += p.numel()

                state = self.state[p]

                p_fp32 = p

                if len(state) == 0:
                    state['exp_avg_sq'] = torch.zeros_like(p)

                exp_avg_sq = state['exp_avg_sq']

                original_grad_dtype = grad.dtype

                # unpack
                if p.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.to(torch.float32)
                    p_fp32 = p.to(torch.float32)
                    exp_avg_sq = exp_avg_sq.to(torch.float32)

                bias_correction_sq: float = self.debias(beta, group['step'])

                # center the gradient vector
                if group["centralization"] > 0.0 and grad.dim() > 1:
                    grad.sub_(
                        grad.mean(dim=tuple(range(1, grad.dim())), keepdim=True).mul_(group["centralization"])
                    )

                if group['clip'] > 0.0:
                    if group['adaptive_clipping']:
                        # Apply Adaptive Gradient Clipping (AGC)
                        grad.copy_(agc(p_fp32, grad, group['adaptive_clip_eps'], group['clip'], group['clip_eps']))
                    else:
                        # Clip the gradient 
                        grad.div_((self.get_rms(grad).clamp_(group['clip_eps']) / group['clip']).clamp_(min=1.0))

                exp_avg_sq.mul_(beta).addcmul_(grad, grad, value=1.0 - beta)

                # pack
                if p.dtype in {torch.float16, torch.bfloat16}:
                    copy_stochastic_(state["exp_avg_sq"], exp_avg_sq)
                    
                # Need to pack grad for next phase
                if original_grad_dtype in {torch.float16, torch.bfloat16}:
                    copy_stochastic_(p.grad, grad)

                exp_avg_sq_sum += (exp_avg_sq / bias_correction_sq).sum()

        if param_size == 0:
            raise ZeroParameterSizeError()

        exp_avg_sq_mean: float = math.sqrt(exp_avg_sq_sum / param_size) #+ self.defaults['eps']

        for group in self.param_groups:
            step_size = group["lr"]
            beta = group["betas"]

            bias_correction_sqrt: float = math.sqrt(self.debias(beta, group['step']))

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad

                state = self.state[p]

                p_fp32 = p

                exp_avg_sq = state["exp_avg_sq"]

                # unpack
                if p.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.to(torch.float32)
                    p_fp32 = p.clone().to(torch.float32)
                    exp_avg_sq = exp_avg_sq.to(torch.float32)

                if group["eps_floor"] < group["eps"]:
                    curr_eps = max(min(self.get_rms(grad) * group["eps2"], group["eps"]), group["eps_floor"] if group["eps_floor"] > 0 else torch.finfo(torch.float32).tiny) # Set a floor for eps to avoid NaN
                else:
                    curr_eps = group["eps"]

                # lr scaler + eps to prevent zero division
                # denom = exp_avg_sq.sqrt() + group['eps']
                de_nom = (exp_avg_sq.sqrt() / bias_correction_sqrt).add_(curr_eps)

                if group["weight_decouple"]:
                    # Perform stepweight decay
                    p_fp32.mul_(1.0 - (1.0 if group["fixed_decay"] else step_size) * group["weight_decay"] / (exp_avg_sq_mean if group["stable_decay"] else 1.0))
                elif group["weight_decay"] > 0.0:
                    grad.add_(p_fp32, alpha=group["weight_decay"])

                # p = p - lr * grad / denom
                p_fp32.addcdiv_(grad, de_nom, value=-step_size)

                # pack
                if p.dtype in {torch.float16, torch.bfloat16}:
                    copy_stochastic_(p, p_fp32)

        return loss
