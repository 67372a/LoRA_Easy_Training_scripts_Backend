# Source: https://github.com/kozistr/pytorch_optimizer/blob/main/pytorch_optimizer/optimizer/adopt.py
import torch

from pytorch_optimizer.base.exception import NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.types import BETAS, CLOSURE, DEFAULTS, LOSS, PARAMETERS
import math
from .utils import copy_stochastic_, agc

class ADOPT(BaseOptimizer):
    r"""Modified Adam Can Converge with Any β2 with the Optimal Rate.

    :param params: PARAMETERS. iterable of parameters to optimize or dicts defining parameter groups.
    :param lr: float. learning rate.
    :param betas: BETAS. coefficients used for computing running averages of gradient and the squared hessian trace.
    :param weight_decay: float. weight decay (L2 penalty).
    :param weight_decouple: bool. the optimizer uses decoupled weight decay as in AdamW.
    :param fixed_decay: bool. fix weight decay.
    :param r: float. EMA factor. between 0.9 ~ 0.99 is preferred.
    :param adanorm: bool. whether to use the AdaNorm variant.
    :param adam_debias: bool. Only correct the denominator to avoid inflating step sizes early in training.
    :param eps: float. term added to the denominator to improve numerical stability.
    :param clip: float. Clipping threshold for clipping (Root mean squared).
    :param adaptive_clipping: bool. Use adaptive clipping instead of root mean squared.
    :param adaptive_clip_eps: float. eps for adaptive gradient clipping.
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
        clip: float = 0.0,
        adaptive_clipping: bool = False,
        adaptive_clip_eps: float = 1e-3,
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
            'adaptive_clipping': adaptive_clipping,
            'adaptive_clip_eps': adaptive_clip_eps,
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

                if group['clip'] > 0.0:
                    if group['adaptive_clipping']:
                        # Apply Adaptive Gradient Clipping (AGC)
                        grad.copy_(agc(p_fp32, grad, group['adaptive_clip_eps'], group['clip']))
                    else:
                        # Clip the gradient 
                        grad.div_((self.get_rms(grad).add_(group['eps']) / group['clip']).clamp_(min=1.0))

                self.apply_weight_decay(
                    p=p_fp32,
                    grad=grad,
                    lr=group['lr'],
                    weight_decay=group['weight_decay'],
                    weight_decouple=group['weight_decouple'],
                    fixed_decay=group['fixed_decay'],
                )

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

                 # unpack
                if p.dtype in {torch.float16, torch.bfloat16}:
                    exp_avg, exp_avg_sq = exp_avg.to(torch.float32), exp_avg_sq.to(torch.float32)

                if group['step'] == 1:
                    exp_avg_sq.addcmul_(grad, grad.conj())

                    # pack
                    if p.dtype in {torch.float16, torch.bfloat16}:
                        copy_stochastic_(state["exp_avg_sq"], exp_avg_sq)
                        copy_stochastic_(p, p_fp32)
                    continue

                exp_avg_sq.mul_(beta2).addcmul_(grad, grad.conj(), value=1.0 - beta2)

                de_nom = exp_avg_sq.sqrt().clamp_(min=group['eps'])
                if group['step'] == 2:
                    exp_avg.addcdiv_(grad, de_nom)
                else:
                    exp_avg.mul_(beta1).addcdiv_(grad, de_nom, value=1.0 - beta1)

                p_fp32.add_(exp_avg, alpha=-group['lr'])

                # pack
                if p.dtype in {torch.float16, torch.bfloat16}:
                    copy_stochastic_(state["exp_avg"], exp_avg)
                    copy_stochastic_(state["exp_avg_sq"], exp_avg_sq)
                    copy_stochastic_(p, p_fp32)

        return loss