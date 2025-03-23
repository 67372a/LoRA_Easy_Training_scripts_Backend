# Authored by: https://github.com/kozistr
import math

import torch

from pytorch_optimizer.base.exception import NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import BETAS, CLOSURE, DEFAULTS, LOSS, PARAMETERS
from pytorch_optimizer.optimizer.utils import get_global_gradient_norm
from .utils import copy_stochastic_, UPDATE_STRATEGY, NORM_TYPE, orthograd, agc
from typing import Callable, Dict, Optional, Tuple, Union, List, Literal


class AdaGC(BaseOptimizer):
    r"""Improving Training Stability for Large Language Model Pretraining.

    :param params: PARAMETERS. iterable of parameters to optimize or dicts defining parameter groups.
    :param lr: float. learning rate.
    :param betas: BETAS. coefficients used for computing running averages of gradient and the squared hessian trace.
    :param beta: float. smoothing coefficient for EMA.
    :param lambda_abs: float. absolute clipping threshold to prevent unstable updates from gradient explosions.
    :param lambda_rel: float. relative clipping threshold to prevent unstable updates from gradient explosions.
    :param warmup_steps: int. warmup steps.
    :param weight_decay: float. weight decay (L2 penalty).
    :param weight_decouple: bool. the optimizer uses decoupled weight decay as in AdamW.
    :param fixed_decay: bool. fix weight decay.
    :param eps: float. term added to the denominator to improve numerical stability.
    """

    def __init__(
        self,
        params: PARAMETERS,
        lr: float = 1e-3,
        betas: BETAS = (0.9, 0.999),
        beta: float = 0.98,
        lambda_abs: float = 1.0,
        lambda_rel: float = 1.05,
        warmup_steps: int = 100,
        weight_decay: float = 1e-1,
        weight_decouple: bool = True,
        fixed_decay: bool = False,
        eps: float = 1e-8,
        eps2: float = 1e-2,
        eps_floor: Optional[float] = None,
        use_orthograd: bool = False,
        adaptive_clip: Optional[float] = None,
        adaptive_clip_eps: float = 1e-3,
        adaptive_clip_type: NORM_TYPE = 'layer',
        update_strategy: UPDATE_STRATEGY = 'unmodified',
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_betas(betas)
        self.validate_range(beta, 'beta', 0.0, 1.0, '[)')
        self.validate_positive(lambda_abs, 'lambda_abs')
        self.validate_positive(lambda_rel, 'lambda_rel')
        self.validate_non_negative(warmup_steps, 'warmup_steps')
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_non_negative(eps, 'eps')

        # Override zero to tiny
        if eps_floor is not None and eps_floor < eps and eps_floor <= 0:
            eps_floor = torch.finfo(torch.float32).tiny

        if update_strategy is not None and update_strategy not in {'unmodified','cautious','grams', 'both'}:
            raise ValueError("Invalid update strategy: {}".format(update_strategy))

        defaults: DEFAULTS = {
            'lr': lr,
            'betas': betas,
            'beta': beta,
            'lambda_abs': lambda_abs,
            'lambda_rel': lambda_rel,
            'warmup_steps': warmup_steps,
            'weight_decay': weight_decay,
            'weight_decouple': weight_decouple,
            'fixed_decay': fixed_decay,
            'eps': eps,
            'eps2': eps2,
            'eps_floor': eps_floor,
            'use_orthograd': use_orthograd,
            'adaptive_clip': adaptive_clip,
            'adaptive_clip_eps': adaptive_clip_eps,
            'adaptive_clip_type': adaptive_clip_type,
            'update_strategy': update_strategy,
        }
        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'AdaGC'

    @torch.no_grad()
    def reset(self):
        pass

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

            beta1, beta2 = group['betas']

            eps, eps2, eps_floor = group['eps'], group['eps2'], group['eps_floor']
            use_orthograd = group['use_orthograd']
            adaptive_clip = group['adaptive_clip']
            adaptive_clip_eps = group['adaptive_clip_eps']
            adaptive_clip_type = group['adaptive_clip_type']
            update_strategy  = group['update_strategy']

            bias_correction1: float = self.debias(beta1, group['step'])
            bias_correction2_sq: float = math.sqrt(self.debias(beta2, group['step']))

            for p in group['params']:
                if p.grad is None:
                    continue

                p_fp32 = p
                grad = p.grad
                if grad.is_sparse:
                    raise NoSparseGradientError(str(self))

                state = self.state[p]

                if 'exp_avg' not in state:
                    state['exp_avg'] = torch.zeros_like(grad)
                    state['exp_avg_sq'] = torch.zeros_like(grad)
                    state['gamma'] = torch.empty((1,), device=grad.device, dtype=torch.float32)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

                if p.dtype in torch.bfloat16:
                    grad = grad.to(torch.float32)
                    p_fp32 = p.to(torch.float32)
                    exp_avg, exp_avg_sq = exp_avg.to(torch.float32), exp_avg_sq.to(torch.float32)

                if use_orthograd and p.ndim >= 1 and p.numel() >= 2:
                    grad = orthograd(p_fp32, grad)

                if adaptive_clip is not None and adaptive_clip > 0 and p.numel() >= 2 and p.ndim >= 1:
                    grad = agc(p=p_fp32, grad=grad, agc_clip_val=adaptive_clip, agc_eps=adaptive_clip_eps, norm_type=adaptive_clip_type)

                if eps_floor is not None and eps_floor < eps:
                    rms_grad = grad.pow(2).mean().sqrt_()
                    curr_eps = max(min(eps, eps2 * rms_grad), eps_floor) # Set a floor for eps to avoid NaN
                else:
                    curr_eps = eps

                self.apply_weight_decay(
                    p=p_fp32,
                    grad=grad,
                    lr=group['lr'],
                    weight_decay=group['weight_decay'],
                    weight_decouple=group['weight_decouple'],
                    fixed_decay=group['fixed_decay'],
                )

                gamma = state['gamma']

                if group['step'] < group['warmup_steps']:
                    grad_norm = get_global_gradient_norm(self.param_groups).add_(curr_eps)

                    h_t = min(group['lambda_abs'] / grad_norm, 1.0)
                    g_hat = grad.mul(h_t)

                    g_hat_norm = g_hat.norm()

                    gamma.copy_(g_hat_norm if group['step'] == 1 else min(gamma, g_hat_norm))
                else:
                    h_t = min(group['lambda_rel'] * gamma / grad.norm(), 1.0)
                    g_hat = grad.mul(h_t)

                    gamma.mul_(group['beta']).add_(g_hat.norm(), alpha=1.0 - group['beta'])

                exp_avg.mul_(beta1).add_(g_hat, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(g_hat, g_hat, value=1.0 - beta2)

                update = (exp_avg / bias_correction1) / exp_avg_sq.sqrt().div_(bias_correction2_sq).add_(curr_eps)

                p_fp32.add_(update, alpha=-group['lr'])

                if p.dtype == torch.bfloat16:
                    copy_stochastic_(state["exp_avg"], exp_avg)
                    copy_stochastic_(state["exp_avg_sq"], exp_avg_sq)
                    copy_stochastic_(p, p_fp32)

        return loss