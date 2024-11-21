import torch
from .utils import copy_stochastic_, quantize, dequantize, agc
import math
from typing import Optional, Literal

from pytorch_optimizer.base.exception import NoSparseGradientError, ZeroParameterSizeError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.types import BETAS, CLOSURE, DEFAULTS, LOSS, PARAMETERS

DATA_FORMAT = Literal['gradient', 'update', 'both']

class RMSProp(BaseOptimizer):
    r"""
    Arguments:
        params (iterable):
            Iterable of parameters to optimize or dicts defining
            parameter groups.
        lr (float):
            Learning rate parameter (default 0.001)
        betas (float, optional):
            coefficient used for computing running averages of
            gradient's square (default: 0.95).
        eps (float):
            Term added to the denominator outside of the root operation to
            improve numerical stability. (default: 1e-8).
        eps2 (float):
            Term to multiple the RMS of the grad to calculate adaptive eps. (default: 0.01).
        eps_floor (float):
            Term to set a floor for adaptive eps, to prevent NaNs, set to >= 0 to turn on adaptive eps (default: None).
        weight_decay (float):
            Weight decay, i.e. a L2 penalty (default: 0).
        centralization (float):
            center model grad (default: 0).
        rectify (bool)
            Rectify variance as per RAdam - https://arxiv.org/abs/1908.03265 (Default: false)
        n_sma_threshold: (int)
            Simple moving average threshold for variance rectification (recommended is 5) (Default: 5).
        degenerated_to_sgd: (bool)
            degenerated to SGD. (Default: false)
        clip_loc: (string)
            Control where clipping is applied. Can be selectively applied: gradient, update, both (Default: gradient)
    """

    def __init__(
        self,
        params: PARAMETERS,
        lr: float = 1e-3,
        betas: float = 0.95, # normal default is 0.999, but was accidently 0.9 for awhile, so adjusting to 0.95 for now
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        weight_decouple: bool = True,
        fixed_decay: bool = False,
        centralization: float = 0.0,
        stable_decay: bool = False,
        clip: float = 0.0,
        clip_eps: float = 1e-8,
        adaptive_clipping: bool = False,
        adaptive_clip_eps: float = 1e-3,
        rectify_variance: bool = False,
        n_sma_threshold: int = 5,
        degenerated_to_sgd: bool = False,
        clip_loc: DATA_FORMAT = 'gradient',
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
            'rectify_variance':rectify_variance,
            'n_sma_threshold':n_sma_threshold,
            'degenerated_to_sgd':degenerated_to_sgd,
            'clip_loc':clip_loc,
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

                if group['clip'] > 0.0 and group['clip_loc'] in {'gradient','both'}:
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
            beta = group["betas"]

            bias_correction_sqrt: float = math.sqrt(self.debias(beta, group['step']))

            step_size, n_sma = self.get_rectify_step_size(
                is_rectify=group["rectify_variance"],
                step=group['step'],
                lr=group['lr'],
                beta2=beta,
                n_sma_threshold=group["n_sma_threshold"],
                degenerated_to_sgd=group["degenerated_to_sgd"],
            )

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

                if not group["rectify_variance"] or step_size > 0 or n_sma >= group["n_sma_threshold"]:
                    if group["weight_decouple"]:
                        # Perform stepweight decay
                        p_fp32.mul_(1.0 - (1.0 if group["fixed_decay"] else step_size) * group["weight_decay"] / (exp_avg_sq_mean if group["stable_decay"] else 1.0))
                    elif group["weight_decay"] > 0.0:
                        grad.add_(p_fp32, alpha=group["weight_decay"])

                if not group["rectify_variance"] or n_sma >= group["n_sma_threshold"]:
                    # lr scaler + eps to prevent zero division
                    # de_nom = exp_avg_sq.sqrt() + group['eps']
                    if group["rectify_variance"]:
                        de_nom = exp_avg_sq.sqrt().add_(group["eps"])
                    else:
                        de_nom = (exp_avg_sq.sqrt() / bias_correction_sqrt).add_(group["eps"])

                    # p = p - lr * grad / denom
                    update = grad.div(de_nom)
                elif step_size > 0:
                    update = grad

                if group['clip'] > 0.0 and group['clip_loc'] in {'update','both'} and (step_size > 0 or n_sma >= group["n_sma_threshold"]):
                    if group['adaptive_clipping']:
                        # Apply Adaptive Gradient Clipping (AGC)
                        update.copy_(agc(p_fp32, update, group['adaptive_clip_eps'], group['clip'], group['clip_eps']))
                    else:
                        # Clip the gradient 
                        update.div_((self.get_rms(update).clamp_(group['clip_eps']) / group['clip']).clamp_(min=1.0))

                if step_size > 0 or n_sma >= group["n_sma_threshold"]:
                    p_fp32.add_(update, alpha=-step_size)

                # pack
                if p.dtype in {torch.float16, torch.bfloat16}:
                    copy_stochastic_(p, p_fp32)

        return loss
