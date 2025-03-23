# Authored originally by: https://github.com/kozistr
import math
from typing import Optional

import torch
from torch.nn import Parameter, ParameterList
from torch.optim import SGD, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LRScheduler

from pytorch_optimizer.base.exception import NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import BETAS, CLOSURE, DEFAULTS, LOSS, PARAMETERS
from .utils import copy_stochastic_, UPDATE_STRATEGY, NORM_TYPE, orthograd, agc
from typing import Callable, Dict, Optional, Tuple, Union, List, Literal


class CosineDecay:
    r"""Applies cosine decay to a parameter (death_rate), using PyTorch's built-in `CosineAnnealingLR`.

    :param death_rate: float. initial value to be decayed.
    :param t_max: int. maximum number of iterations for the decay.
    :param eta_min: Optional[float]. minimum value of the parameter after decay. defaults to 0.
    :param last_epoch: Optional[int]. the index of the last epoch. Defaults to -1.
    """

    def __init__(self, death_rate: float, t_max: int, eta_min: float = 0.0, last_epoch: int = -1):
        self.sgd: Optimizer = SGD(ParameterList([Parameter(torch.zeros(1))]), lr=death_rate)
        self.cosine_stepper: LRScheduler = CosineAnnealingLR(self.sgd, t_max + 1, eta_min, last_epoch)
        self.t_max = t_max
        self.eta_min = eta_min

    def step(self, current_step: int) -> None:
        r"""One step of the cosine decay scheduler.

        :param current_step: int. Current step index.
        """
        self.cosine_stepper.step(current_step)

    def get_death_rate(self, current_step: int) -> float:
        r"""Get the updated rate (death_rate) at the given step.

        :param current_step: int. Current step index.
        """
        if current_step >= self.t_max:
            return self.eta_min

        self.step(current_step)

        return self.sgd.param_groups[0]['lr']

class StableSPAM(BaseOptimizer):
    r"""How to Train in 4-Bit More Stably than 16-Bit Adam.

    :param params: PARAMETERS. iterable of parameters to optimize or dicts defining parameter groups.
    :param lr: float. learning rate.
    :param betas: BETAS. coefficients used for computing running averages of gradient and the squared hessian trace.
    :param gamma1: float.
    :param gamma2: float.
    :param theta: float.
    :param t_max: Optional[int]. total number of steps.
    :param eta_min: float. eta_min of CosineDecay.
    :param weight_decay: float. weight decay (L2 penalty).
    :param update_proj_gap: int. update projection gap.
    :param eps: float. term added to the denominator to improve numerical stability.
    """

    def __init__(
        self,
        params: PARAMETERS,
        lr: float = 1e-3,
        betas: BETAS = (0.9, 0.999),
        gamma1: float = 0.7,
        gamma2: float = 0.9,
        theta: float = 0.999,
        t_max: Optional[int] = None,
        eta_min: float = 0.5,
        weight_decay: float = 0.0,
        update_proj_gap: int = 1000,
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
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_positive(update_proj_gap, 'update_proj_gap')
        self.validate_non_negative(eps, 'eps')

        # Override zero to tiny
        if eps_floor is not None and eps_floor < eps and eps_floor <= 0:
            eps_floor = torch.finfo(torch.float32).tiny

        if update_strategy is not None and update_strategy not in {'unmodified','cautious','grams', 'both'}:
            raise ValueError("Invalid update strategy: {}".format(update_strategy))

        self.gamma1: float = betas[0] if gamma1 == -1.0 else gamma1
        self.gamma2: float = gamma2
        self.theta: float = theta
        self.t_max = t_max
        self.update_proj_gap = update_proj_gap
        self.warmup = CosineDecay(1.0, t_max, eta_min=eta_min) if t_max is not None else None

        self.total_step: int = 0

        defaults: DEFAULTS = {
            'lr': lr, 
            'betas': betas, 
            'weight_decay': weight_decay, 
            'eps': eps,
            'eps2': eps2,
            'eps_floor': eps_floor, 
            'use_orthograd': use_orthograd,
            'adaptive_clip': adaptive_clip,
            'adaptive_clip_eps': adaptive_clip_eps,
            'adaptive_clip_type': adaptive_clip_type,
            'update_strategy': update_strategy,
            **kwargs}
        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'StableSPAM'

    @torch.no_grad()
    def reset(self):
        for group in self.param_groups:
            group['step'] = 0
            for p in group['params']:
                state = self.state[p]

                state['exp_avg'] = torch.zeros_like(p)
                state['exp_avg_sq'] = torch.zeros_like(p)
                state['m_norm_t'] = torch.zeros(1, device=p.device, dtype=torch.float32)
                state['v_norm_t'] = torch.zeros(1, device=p.device, dtype=torch.float32)
                state['m_max_t'] = torch.zeros(1, device=p.device, dtype=torch.float32)

    @torch.no_grad()
    def step(self, closure: CLOSURE = None) -> LOSS:
        loss: LOSS = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self.total_step += 1
        scale: float = self.warmup.get_death_rate(self.total_step) if self.warmup is not None else 1.0

        for group in self.param_groups:
            if 'step' not in group:
                group['step'] = 1
            else:
                group['step'] += 1

            beta1, beta2 = group['betas']
            beta1 *= scale

            eps, eps2, eps_floor = group['eps'], group['eps2'], group['eps_floor']
            use_orthograd = group['use_orthograd']
            adaptive_clip = group['adaptive_clip']
            adaptive_clip_eps = group['adaptive_clip_eps']
            adaptive_clip_type = group['adaptive_clip_type']
            update_strategy  = group['update_strategy']

            bias_correction1: float = self.debias(beta1, group['step'])
            bias_correction2: float = self.debias(beta2, group['step'])
            bias_correction2_sq: float = math.sqrt(bias_correction2)

            step_size: float = group['lr'] / bias_correction1

            theta_t: float = 1.0 - self.theta ** group['step']

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
                    state['m_norm_t'] = torch.zeros(1, device=grad.device, dtype=torch.float32)
                    state['v_norm_t'] = torch.zeros(1, device=grad.device, dtype=torch.float32)
                    state['m_max_t'] = torch.zeros(1, device=grad.device, dtype=torch.float32)

                exp_avg, exp_avg_sq, m_max_t = state['exp_avg'], state['exp_avg_sq'], state['m_max_t']

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
                    p_fp32,
                    grad=grad,
                    lr=group['lr'],
                    weight_decay=group['weight_decay'],
                    weight_decouple=True,
                    fixed_decay=False,
                )

                max_grad = torch.max(grad.abs())

                m_max_t.lerp_(max_grad, weight=1.0 - self.theta)

                m_max_hat = m_max_t / theta_t

                mask = grad.abs() > m_max_hat
                if mask.sum() > 0:
                    grad[mask].div_(max_grad).mul_(m_max_hat)

                grad_norm = torch.norm(grad)

                m_norm_t, v_norm_t = state['m_norm_t'], state['v_norm_t']
                m_norm_t.lerp_(grad_norm, weight=1.0 - self.gamma1 * scale)
                v_norm_t.lerp_(grad_norm.pow(2), weight=1.0 - self.gamma2)

                m_norm_hat = m_norm_t / (1.0 - (self.gamma1 * scale) ** group['step'])
                v_norm_hat = v_norm_t / (1.0 - self.gamma2 ** group['step'])

                c_norm_t = m_norm_hat.div_(v_norm_hat.sqrt_().add_(curr_eps))

                grad.div_(grad_norm).mul_(c_norm_t)

                if self.update_proj_gap > 0 and self.total_step % self.update_proj_gap == 0:
                    state['exp_avg'] = torch.zeros_like(grad)
                    state['exp_avg_sq'] = torch.zeros_like(grad)
                    group['step'] = 1

                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                de_nom = exp_avg_sq.sqrt().div_(bias_correction2_sq).add_(curr_eps)

                if update_strategy in {'cautious','grams'}:
                    if update_strategy in {'cautious','both'}:
                        mask = (exp_avg * grad > 0).to(grad.dtype)
                        mask.div_(mask.mean().clamp_(min=1e-3))
                        update = exp_avg * mask
                    if update_strategy in {'grams','both'}:
                        update.copy_(torch.sign(grad) * exp_avg.abs())

                p_fp32.addcdiv_(update, de_nom, value=-step_size)

                if p.dtype == torch.bfloat16:
                    copy_stochastic_(state["exp_avg"], exp_avg)
                    copy_stochastic_(state["exp_avg_sq"], exp_avg_sq)
                    copy_stochastic_(p, p_fp32)

        return loss