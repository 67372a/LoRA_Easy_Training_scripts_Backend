import math
from typing import Tuple

import torch

from pytorch_optimizer.base.exception import NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import BETAS, CLOSURE, DEFAULTS, LOSS, PARAMETERS

from .utils import copy_stochastic_

class RACS(BaseOptimizer):
    r"""Row and Column Scaled SGD.

    :param params: PARAMETERS. iterable of parameters to optimize or dicts defining parameter groups.
    :param lr: float. learning rate.
    :param beta: float. momentum factor.
    :param alpha: float. scaler.
    :param gamma: float. limiter threshold.
    :param weight_decay: float. weight decay (L2 penalty).
    :param weight_decouple: bool. the optimizer uses decoupled weight decay as in AdamW.
    :param fixed_decay: bool. fix weight decay.
    :param eps: float. term added to the denominator to improve numerical stability.
    """

    def __init__(
        self,
        params: PARAMETERS,
        lr: float = 1e-3,
        beta: float = 0.9,
        alpha: float = 0.05,
        gamma: float = 1.01,
        weight_decay: float = 0.0,
        weight_decouple: bool = True,
        fixed_decay: bool = False,
        eps: float = 1e-8,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_range(beta, 'beta', 0.0, 1.0)
        self.validate_range(alpha, 'alpha', 0.0, 1.0)
        self.validate_positive(gamma, 'gamma')
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_non_negative(eps, 'eps')

        defaults: DEFAULTS = {
            'lr': lr,
            'beta': beta,
            'alpha': alpha,
            'gamma': gamma,
            'weight_decay': weight_decay,
            'weight_decouple': weight_decouple,
            'fixed_decay': fixed_decay,
            'eps': eps,
        }
        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'RACS'

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

            beta = group['beta']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise NoSparseGradientError(str(self))

                state = self.state[p]

                # RACS doesn't support scalars or dim > 2
                # TODO - Nested optimizer for these?
                if p.ndim == 0 or p.ndim > 2:
                    p_fp32 = p
                    if p.dtype in {torch.float16, torch.bfloat16}:
                        grad = grad.to(torch.float32)
                        p_fp32 = p.to(torch.float32)

                    self.apply_weight_decay(
                        p=p_fp32,
                        grad=grad,
                        lr=group['lr'],
                        weight_decay=group['weight_decay'],
                        weight_decouple=group['weight_decouple'],
                        fixed_decay=group['fixed_decay'],
                    )

                    p_fp32.add_(grad, alpha=-group['lr'])

                    copy_stochastic_(p, p_fp32)
                    continue

                if len(p.shape) == 1:
                    p = p.unsqueeze(0)  # noqa: PLW2901
                    grad = grad.unsqueeze(0)

                if len(state) == 0:
                    state['s'] = torch.zeros(p.size(0), dtype=p.dtype, device=p.device)
                    state['q'] = torch.ones(p.size(1), dtype=p.dtype, device=p.device)
                    state['theta'] = torch.zeros((), dtype=grad.dtype, device=grad.device)

                s, q = state['s'], state['q']

                p_fp32 = p
                if p.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.to(torch.float32)
                    p_fp32 = p.to(torch.float32)
                    s, q = s.to(torch.float32), q.to(torch.float32)

                self.apply_weight_decay(
                    p=p_fp32,
                    grad=grad,
                    lr=group['lr'],
                    weight_decay=group['weight_decay'],
                    weight_decouple=group['weight_decouple'],
                    fixed_decay=group['fixed_decay'],
                )

                grad_p2 = grad.pow(2)
                s.mul_(beta).add_(grad_p2.mean(dim=1), alpha=1.0 - beta)
                q.mul_(beta).add_(grad_p2.mean(dim=0), alpha=1.0 - beta)

                s_sq = s.add(group['eps']).sqrt_().unsqueeze(1)
                q_sq = q.add(group['eps']).sqrt_().unsqueeze(0)

                grad_hat = grad / (s_sq * q_sq)

                grad_hat_norm = torch.norm(grad_hat)
                threshold = (
                    group['gamma'] / max(grad_hat_norm / (state['theta'] + group['eps']), group['gamma'])
                    if group['step'] > 1
                    else 1.0
                )
                copy_stochastic_(state['theta'], grad_hat_norm.mul(threshold))

                p_fp32.add_(grad_hat, alpha=-group['lr'] * group['alpha'] * threshold)

                copy_stochastic_(state['s'], s)
                copy_stochastic_(state['q'], q)
                copy_stochastic_(p, p_fp32)

        return loss
