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
    r"""
    Implements AdamW with Adaptive Gradient Clipping (AdaGC) optimizer.
    
    Based on "AdaGC: Improving Training Stability for Large Language Model Pretraining"
    by Wang et al. This implementation follows Algorithm 1 from the paper.
    
    AdaGC eliminates loss spikes during training by adaptively adjusting local
    thresholds per parameter through an exponential moving average of gradient norms.

    Args:
        params (iterable): iterable of parameters to optimize
        alphas (float, list or callable): learning rate or list of learning rates α_t
        weight_decay (float): weight decay coefficient λ_w
        eps (float): epsilon for numerical stability ε_1
        beta1 (float): coefficient for momentum β_1
        beta2 (float): coefficient for velocity β_2
        beta (float): coefficient for EMA gradient norm history β (paper recommends 0.98)
        lambda_abs (float): absolute global clipping value λ_abs (used before t_start, paper recommends 1.0)
        lambda_rel (float): relative clipping threshold λ_rel (paper recommends 1.05)
        t_start (int): starting time for adaptive clipping T_start (paper recommends 100)
    """

    def __init__(self, 
                params: PARAMETERS,
                lr: float = 1e-3, 
                betas: BETAS = (0.9, 0.999),
                weight_decay: float = 0, 
                eps: float = 1e-8,
                eps2: float = 1e-2,
                eps_floor: Optional[float] = None,
                beta: float = 0.98, 
                lambda_abs: float = 1.0, 
                lambda_rel: float = 1.05, 
                warmup_steps: float = 20,
                use_orthograd: bool = False,
                update_strategy: UPDATE_STRATEGY = 'unmodified',
                **kwargs):
        
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
            'weight_decay': weight_decay,
            'eps': eps,
            'eps2': eps2,
            'eps_floor': eps_floor,
            'betas': betas,
            'beta': beta,
            'lambda_abs': lambda_abs,
            'lambda_rel': lambda_rel,
            'warmup_steps': warmup_steps,
            'use_orthograd': use_orthograd,
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
        """Performs a single optimization step.
        
        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss: LOSS = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                # Get gradient
                p_fp32 = p
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('AdamWAdaGC does not support sparse gradients')
                
                # Get state
                state = self.state[p]
                
                # State initialization (line 2)
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)  # m_0 ← 0
                    state['exp_avg_sq'] = torch.zeros_like(p.data)  # v_0 ← 0
                    state['gamma'] = torch.empty(1, device=grad.device, dtype=torch.float32)  # Gradient norm history (γ)
                    # γ is initialized in first iteration when calculating clipping
                
                # Update step counter
                state['step'] += 1
                step = state['step']
                
                # Get hyperparameters
                lr = group['lr']
                weight_decay = group['weight_decay']
                eps = group['eps']
                eps2 = group['eps2']
                eps_floor = group['eps_floor']
                beta1, beta2 = group['betas']
                beta = group['beta']
                lambda_abs = group['lambda_abs']
                lambda_rel = group['lambda_rel']
                warmup_steps = group['warmup_steps']
                use_orthograd = group['use_orthograd']
                update_strategy = group['update_strategy']
                
                # Retrieve state variables
                exp_avg, exp_avg_sq, gamma = state['exp_avg'], state['exp_avg_sq'], state['gamma']

                if p.dtype == torch.bfloat16:
                    grad = grad.to(torch.float32)
                    p_fp32 = p.to(torch.float32)
                    exp_avg, exp_avg_sq = exp_avg.to(torch.float32), exp_avg_sq.to(torch.float32)
                
                # Compute gradient (line 4: g_t = ∇_θ f_t(θ_{t-1}, X_t))
                g_t = grad

                if use_orthograd and p.ndim >= 1 and p.numel() >= 2:
                    g_t = orthograd(p_fp32, g_t)

                if eps_floor is not None and eps_floor < eps:
                    rms_grad = g_t.pow(2).mean().sqrt_()
                    curr_eps = max(min(eps, eps2 * rms_grad), eps_floor) # Set a floor for eps to avoid NaN
                else:
                    curr_eps = eps
                
                if step < warmup_steps:
                    # Before t_start: global clipping (lines 5-7)
                    g_t_norm = torch.linalg.norm(g_t)
                    
                    h_t = min(lambda_abs / g_t_norm.add_(curr_eps), 1.0)

                    g_t.mul_(h_t)  # g_t = h_t · g_t
                    
                    # Update gamma with minimum of current and previous gamma (lines 8-10)
                    g_t_norm = torch.linalg.norm(g_t)
                    if step == 1:
                        # First iteration: γ_0,i = ||g_1,i||
                        gamma.copy_(g_t_norm)
                    else:
                        # Subsequent iterations: γ_t,i = min{γ_{t-1,i}, ||g_t,i||}
                        gamma = torch.minimum(gamma, g_t_norm)
                else:
                    # After t_start: per-parameter adaptive clipping (lines 11-16)
                    g_t_norm = torch.linalg.norm(g_t)
                    
                    # Compute clipping factors (line 13): h_t,i = min{λ_rel·γ_{t-1,i}/||g_t,i||, 1.0}
                    h_t = torch.minimum(lambda_rel * gamma / g_t_norm.add_(curr_eps), 
                                        torch.ones_like(g_t_norm))
                    
                    # Apply clipping (line 14): g̃_t,i = h_t,i · g_t,i
                    g_t.mul_(h_t.item())
                    
                    # Update gamma using EMA (line 15): γ_t,i = β·γ_{t-1,i} + (1-β)||g̃_t1,i||
                    gamma.mul_(beta).add_(torch.linalg.norm(g_t), alpha= 1 - beta)

                state['gamma'] = gamma
                
                # Update momentum (line 18): m_t = β_1*m_{t-1} + (1-β_1)g̃_t
                exp_avg.mul_(beta1).add_(g_t, alpha=1 - beta1)
                
                # Update velocity (line 19): v_t = β_2*v_{t-1} + (1-β_2)g̃_t²
                exp_avg_sq.mul_(beta2).addcmul_(g_t, g_t, value=1 - beta2)
                
                # Compute bias-corrected momentum and velocity (line 20)
                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step
                m_hat = exp_avg / bias_correction1  # m̂_t = m_t/(1-β_1^t)
                v_hat = exp_avg_sq / bias_correction2  # v̂_t = v_t/(1-β_2^t)

                update = m_hat

                if update_strategy in {'cautious','grams','both'}:
                    if update_strategy in {'cautious','both'}:
                        mask = (update * g_t > 0).to(grad.dtype)
                        mask.div_(mask.mean().clamp_(min=1e-3))
                        update = update * mask
                    if update_strategy in {'grams','both'}:
                        update.copy_(torch.sign(g_t) * update.abs())
                
                # Update parameters (line 21)
                p_fp32.mul_(1 - lr * weight_decay)  # Weight decay term: θ_t = θ_{t-1} - α_t*λ_w*θ_{t-1}
                p_fp32.addcdiv_(update, torch.sqrt(v_hat) + curr_eps, value=-lr)  # - α_t*m̂_t/(√v̂_t + ε_1)

                if p.dtype == torch.bfloat16:
                    copy_stochastic_(state["exp_avg"], exp_avg)
                    copy_stochastic_(state["exp_avg_sq"], exp_avg_sq)
                    copy_stochastic_(p, p_fp32)
                
        return loss