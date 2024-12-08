# Source: https://github.com/kozistr/pytorch_optimizer/blob/main/pytorch_optimizer/optimizer/adopt.py
import torch

from pytorch_optimizer.base.exception import NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.types import BETAS, CLOSURE, DEFAULTS, LOSS, PARAMETERS
import math
from .utils import copy_stochastic_

class ADOPT(BaseOptimizer):
    r"""Modified Adam Can Converge with Any β2 with the Optimal Rate.

    :param params: PARAMETERS. iterable of parameters to optimize or dicts defining parameter groups.
    :param lr: float. learning rate.
    :param betas: BETAS. coefficients used for computing running averages of gradient and the squared hessian trace.
    :param weight_decay: float. weight decay (L2 penalty).
    :param weight_decouple: bool. the optimizer uses decoupled weight decay as in AdamW.
    :param fixed_decay: bool. fix weight decay.
    :param eps: float. term added to the denominator to improve numerical stability.
    :param clip: float. special form of clip for ADOPT, recommended and default value is 0.25.
    :param cautious: bool: Use cautious mask on parameter update - https://arxiv.org/abs/2411.16085 (default: False)
    """

    def __init__(
        self,
        params: PARAMETERS,
        lr: float = 1e-3,
        betas: BETAS = (0.9, 0.9999),
        weight_decay: float = 0.0,
        weight_decouple: bool = False,
        fixed_decay: bool = False,
        eps: float = 1e-6,
        clip: float = 0.25,
        cautious: bool = False,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_betas(betas)
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_non_negative(eps, 'eps')

        defaults: DEFAULTS = {
            'lr': lr,
            'betas': betas,
            'weight_decay': weight_decay,
            'weight_decouple': weight_decouple,
            'fixed_decay': fixed_decay,
            'eps': eps,
            'clip': clip,
            'cautious': cautious,
        }

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'ADOPT'
    
    @staticmethod
    def get_rms(x: torch.Tensor) -> float:
        r"""Get RMS."""
        return x.norm(2) / math.sqrt(x.numel())

    @torch.no_grad()
    def reset(self):
        for group in self.param_groups:
            group['step'] = 0
            for p in group['params']:
                state = self.state[p]

                state['exp_avg'] = torch.zeros_like(p)
                state['exp_avg_sq'] = torch.zeros_like(p)

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

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise NoSparseGradientError(str(self))

                state = self.state[p]

                p_fp32 = p

                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)

                 # unpack
                if p.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.to(torch.float32)
                    p_fp32 = p.clone().to(torch.float32)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

                 # unpack
                if p.dtype in {torch.float16, torch.bfloat16}:
                    exp_avg, exp_avg_sq = exp_avg.to(torch.float32), exp_avg_sq.to(torch.float32)

                if group['weight_decay'] != 0 and not group['weight_decouple']:
                    grad = grad.add(p_fp32, alpha=group['weight_decay'])

                if group['step'] == 1:
                    exp_avg_sq.addcmul_(grad, grad.conj())
                    copy_stochastic_(state["exp_avg_sq"], exp_avg_sq)
                    continue

                if group['weight_decay'] != 0 and group['weight_decouple']:
                    p_fp32.add_(p_fp32, alpha=-group['lr'] * group['weight_decay'])

                denom = torch.clamp(exp_avg_sq.sqrt(), group['eps'])
                normed_grad = grad.div(denom)
                if group['clip'] is not None:
                    clip = group['step']-1 **group['clip']
                    normed_grad.clamp_(-clip, clip)

                exp_avg.lerp_(normed_grad, 1 - beta1)

                if group["cautious"]:
                    # compute norm gradient
                    mask = (exp_avg * normed_grad > 0).to(normed_grad.dtype)
                    mask.div_(mask.mean().clamp_(min=1e-3))
                else:
                    mask = 1.0

                p_fp32.add_(exp_avg * mask, alpha=-group['lr'])
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad.conj(), value=1 - beta2)

                # pack
                if p.dtype in {torch.float16, torch.bfloat16}:
                    copy_stochastic_(state["exp_avg"], exp_avg)
                    copy_stochastic_(state["exp_avg_sq"], exp_avg_sq)
                    copy_stochastic_(p, p_fp32)

        return loss