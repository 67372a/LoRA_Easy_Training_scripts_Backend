# Authored by: https://github.com/kozistr
# Source: https://github.com/kozistr/pytorch_optimizer/blob/main/pytorch_optimizer/optimizer/ademamix.py

import math
from typing import Optional, Literal

import torch

from pytorch_optimizer.base.exception import NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.types import BETAS, CLOSURE, DEFAULTS, LOSS, PARAMETERS
from .utils import copy_stochastic_, UPDATE_STRATEGY

class AdEMAMix(BaseOptimizer):
    r"""Better, Faster, Older.

    :param params: PARAMETERS. iterable of parameters to optimize or dicts defining parameter groups.
    :param lr: float. learning rate.
    :param betas: BETAS. coefficients used for computing running averages of gradient and the squared hessian trace.
    :param weight_decay: float. weight decay (L2 penalty).
    :param weight_decouple: bool. the optimizer uses decoupled weight decay as in AdamW.
    :param fixed_decay: bool. fix weight decay.
    :param clip: float. threshold of root-mean-square of gradient update.
    :param alpha: float. usually between 4 and 10 would work well.
    :param t_alpha_beta3: Optional[float]. total number of iterations is preferred when needed.
    :param eps: float. term added to the denominator to improve numerical stability.
    :param centralization: float. center model grad 
    cautious (bool) (deprecated, use update strategy)
        Use cautious mask on parameter update - https://arxiv.org/abs/2411.16085 (default: False)
    update_strategy (str) (NOTE: for backwards compatibility, cautious parameter being set to true will override to cautious)
        Determine the update strategy to use, valid values are 'unmodified', 'cautious' (https://arxiv.org/abs/2411.16085), 
        and 'grams' (https://arxiv.org/abs/2412.17107) (default: unmodified)
    """

    def __init__(
        self,
        params: PARAMETERS,
        lr: float = 1e-3,
        betas: BETAS = (0.9, 0.999, 0.9999),
        weight_decay: float = 0.0,
        weight_decouple: bool = False,
        fixed_decay: bool = False,
        clip: float = 0.0,
        alpha: float = 5.0,
        t_alpha_beta3: Optional[float] = None,
        eps: float = 1e-8,
        centralization: float = 0.0,
        cautious: bool = False,
        update_strategy: UPDATE_STRATEGY = 'unmodified',
        adopt: bool = False,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_betas(betas)
        self.validate_non_negative(alpha, 'alpha')
        self.validate_non_negative(t_alpha_beta3, 't_alpha_beta3')
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_non_negative(eps, 'eps')
        self.validate_non_negative(clip, 'clip')
        self.validate_non_negative(centralization, 'centralization')

        if update_strategy is not None and update_strategy not in {'unmodified','cautious','grams'}:
            raise ValueError("Invalid update strategy: {}".format(update_strategy))
        
        # If cautious true, override update strategy to cautious
        if cautious:
            update_strategy = 'cautious'

        defaults: DEFAULTS = {
            'lr': lr,
            'betas': betas,
            'weight_decay': weight_decay,
            'weight_decouple': weight_decouple,
            'clip': clip,
            'fixed_decay': fixed_decay,
            'alpha': alpha,
            't_alpha_beta3': t_alpha_beta3,
            'eps': eps,
            'centralization': centralization,
            'cautious': cautious,
            'update_strategy': update_strategy,
            'adopt': adopt,
        }

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'AdEMAMix'

    @torch.no_grad()
    def reset(self):
        for group in self.param_groups:
            group['step'] = 0
            beta1, beta2, beta3 = group['betas']

            for p in group['params']:
                state = self.state[p]

                if beta1 > 0.0: # save memory in case beta1 is 0.0
                    state['exp_avg'] = torch.zeros_like(p)
                else: 
                    state['exp_avg'] = None
                state['exp_avg_sq'] = torch.zeros_like(p)
                state['exp_avg_slow'] = torch.zeros_like(p)

    @staticmethod
    def schedule_alpha(t_alpha_beta3: Optional[float], step: int, alpha: float) -> float:
        if t_alpha_beta3 is None:
            return alpha
        return min(step * alpha / t_alpha_beta3, alpha)

    @staticmethod
    def schedule_beta3(t_alpha_beta3: Optional[float], step: int, beta1: float, beta3: float, eps: float) -> float:
        if t_alpha_beta3 is None:
            return beta3

        # Add eps to prevent log 0
        log_beta1, log_beta3 = math.log(beta1 + eps), math.log(beta3)

        return min(
            math.exp(
                log_beta1 * log_beta3 / ((1.0 - step / t_alpha_beta3) * log_beta3 + (step / t_alpha_beta3) * log_beta1)
            ),
            beta3,
        )
    
    @staticmethod
    def get_rms(x: torch.Tensor) -> float:
        r"""Get RMS."""
        return x.norm(2) / math.sqrt(x.numel())

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

            beta1, beta2, beta3 = group['betas']

            step = group['step']

            bias_correction1: float = self.debias(beta1, step)
            bias_correction2_sq: float = math.sqrt(self.debias(beta2, step))

            eps = group['eps']
            clip = group['clip']
            centralization = group['centralization']
            adopt = group['adopt']

            alpha_t: float = self.schedule_alpha(group['t_alpha_beta3'], step, group['alpha'])
            beta3_t: float = self.schedule_beta3(group['t_alpha_beta3'], step, beta1, beta3, eps)

            for p in group['params']:
                if p.grad is None:
                    continue
                    
                if p.grad.is_sparse:
                    raise NoSparseGradientError(str(self))
                
                p_fp32 = p
                grad = p.grad

                if p.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.to(torch.float32)
                    p_fp32 = p.to(torch.float32)
                
                state = self.state[p]

                if len(state) == 0:
                    if beta1 > 0.0: # save memory in case beta1 is 0.0
                        state['exp_avg'] = torch.zeros_like(p)
                    else: 
                        state['exp_avg'] = None
                    state['exp_avg_sq'] = torch.zeros_like(p)
                    state['exp_avg_slow'] = torch.zeros_like(p)

                # center the gradient vector
                if centralization > 0.0 and grad.dim() > 1:
                    grad.sub_(
                        grad.mean(dim=tuple(range(1, grad.dim())), keepdim=True).mul_(centralization)
                    )

                # Clip the gradient 
                if clip > 0.0:
                    grad.div_((self.get_rms(grad).add_(eps) / clip).clamp_(min=1.0))

                exp_avg, exp_avg_sq, exp_avg_slow = state['exp_avg'], state['exp_avg_sq'], state['exp_avg_slow']

                if p.dtype in {torch.float16, torch.bfloat16}:
                    if beta1 > 0.0:
                        exp_avg = exp_avg.to(torch.float32)
                    exp_avg_sq, exp_avg_slow = exp_avg_sq.to(torch.float32), exp_avg_slow.to(torch.float32)

                if adopt and step == 0:
                    exp_avg_sq.add_(grad)
                else:
                    og_grad = grad
                    if not adopt:
                        exp_avg_sq.mul_(beta2).addcmul_(og_grad, og_grad, value=1.0 - beta2)
                        de_nom = (exp_avg_sq.sqrt() / bias_correction2_sq).add_(eps)
                    else:
                        de_nom = (exp_avg_sq.sqrt()).add_(eps)
                        exp_avg_sq.mul_(beta2).addcmul_(og_grad, og_grad, value=1.0 - beta2)
                        adopt_clip: float = (step-1)**0.25
                        scaled_adopt_clip = adopt_clip * de_nom
                        grad = grad.clamp(-scaled_adopt_clip, scaled_adopt_clip)

                    if beta1 > 0.0:
                        exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                    else:
                        exp_avg = grad

                    exp_avg_slow.mul_(beta3_t).add_(grad, alpha=1.0 - beta3_t)

                    update = (exp_avg.div(bias_correction1) + alpha_t * exp_avg_slow)

                    if group['update_strategy'] in {'cautious','grams'}:
                        if group['update_strategy'] == 'cautious':
                            mask = (update * grad > 0).to(grad.dtype)
                            mask.div_(mask.mean().clamp_(min=1e-3))
                            update = update * mask
                        elif group['update_strategy'] == 'grams':
                            update.copy_(torch.sign(grad) * update.abs())

                    update = update / de_nom

                    self.apply_weight_decay(
                        p=p_fp32,
                        grad=update,
                        lr=group['lr'],
                        weight_decay=group['weight_decay'],
                        weight_decouple=group['weight_decouple'],
                        fixed_decay=group['fixed_decay'],
                    )

                    p_fp32.add_(-group['lr'] * update)
                    
                if p.dtype in {torch.float16, torch.bfloat16}:
                    if beta1 > 0.0:
                        copy_stochastic_(state["exp_avg"], exp_avg)
                    copy_stochastic_(state["exp_avg_sq"], exp_avg_sq)
                    copy_stochastic_(state["exp_avg_slow"], exp_avg_slow)
                    copy_stochastic_(p, p_fp32)

        return loss
