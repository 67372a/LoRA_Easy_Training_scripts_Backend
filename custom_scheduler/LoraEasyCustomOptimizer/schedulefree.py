# Source: https://github.com/facebookresearch/schedule_free/blob/main/schedulefree/wrap_schedulefree.py
# Modified to be an actual optimizer, allowing it to wrap any optimizer and work in Kohya's
from typing import Callable, Dict, Optional, Tuple, Union, List, Literal

import torch
import math

from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.types import BETAS, CLOSURE, DEFAULTS, LOSS, PARAMETERS, OPTIMIZER
from pytorch_optimizer.base.exception import NoSparseGradientError
from .utils import copy_stochastic_, NORM_TYPE, agc

class ScheduleFreeWrapper(BaseOptimizer):
    r"""
        Wrap any optimizer to make it Schedule-Free. 
        
        This version uses a memory-efficient swap operation but may be slower than the reference version. In most cases
        the performance difference is negligible.
        For the best possible performance and memory-usage, Schedule-Free needs 
        to be directly integrated with the base optimizer.

        When using this version, you can disable the base optimizer's 
        momentum, as it's no longer necessary when using our wrapper's 
        momentum (although you can use both types of momentum if you want).

        If you set weight decay on the base optimizer, it computes weight decay
        at $z$. We offer the option to compute weight decay at $y$, via the 
        `weight_decay_at_y` parameter, which seems to give better results in 
        our experiments. This approach to decay only works correctly if the base
        optimizer uses group["lr"] as the current learning rate. 

        params (PARAMETERS): 
            iterable of parameters to optimize or dicts defining parameter groups.
        base_optimizer (OPTIMIZER): 
            PyTorch optimizer object, in Kohya's pass in an additional optimizer arg called 
            base_optimizer_type and the fully qualified optimizer name. 
            e.x. 
                base_optimizer_type=LoraEasyCustomOptimizer.compass.Compass
                base_optimizer_type=LoraEasyCustomOptimizer.came.CAME
                base_optimizer_type=LoraEasyCustomOptimizer.adopt.ADOPT
        sf_momentum (float): 
            Apply momentum on the outer optimizer (default 0.9)
        sf_weight_decay_at_y (float): 
            Weight decay calculated at the y point. Set weight decay on the 
            inner optimizer to instead calculate at z (default: 0.0).
        sf_r (float): Use polynomial weighting in the average 
            with power r (default 0.0).
        sf_weight_lr_power (float): The weights in the average will
            be equal to lr raised to this power. Set to 0 for no weighting
            (default 2.0).
    """
    def __init__(self, 
                 params: PARAMETERS,
                 base_optimizer : OPTIMIZER, 
                 sf_weight_decay_at_y : float = 0.0,
                 sf_momentum : float = 0.9,
                 sf_weight_lr_power : float = 2.0,
                 sf_r : float = 0.0,
                 **kwargs):
        
        self.validate_non_negative(sf_weight_decay_at_y, 'sf_weight_decay_at_y')
        self.validate_non_negative(sf_momentum, 'sf_momentum')
        self.validate_non_negative(sf_weight_lr_power, 'sf_weight_lr_power')
        self.validate_non_negative(sf_r, 'sf_r')

        self.sf_weight_decay_at_y = sf_weight_decay_at_y
        self.sf_weight_lr_power = sf_weight_lr_power
        self.sf_r = sf_r
        self.sf_momentum = sf_momentum
        self.train_mode = False

        defaults: DEFAULTS = {'sf_weight_decay_at_y': sf_weight_decay_at_y, 'sf_momentum': sf_momentum, 'sf_weight_lr_power': sf_weight_lr_power, 'sf_r': sf_r}
        defaults.update(kwargs)
        super().__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    def __str__(self) -> str:
        return 'ScheduleFreeWrapper'

    @torch.no_grad()
    def reset(self):
        for group in self.param_groups:
            group['sf_step'] = 0

            for p in group['params']:
                state = self.state[p]
                state['z'] = torch.clone(p, memory_format=torch.preserve_format)

        self.base_optimizer.reset()

    @torch.no_grad()
    def eval(self):
        if self.train_mode:
            for group in self.param_groups:
                for p in group['params']:
                    state = self.state[p]
                    if 'z' in state:
                        # Set p to x
                        p.lerp_(end=state['z'], weight=1-1/self.sf_momentum)
        self.train_mode = False

    @torch.no_grad()
    def train(self):
        if not self.train_mode:
            for group in self.param_groups:
                for p in group['params']:
                    state = self.state[p]
                    if 'z' in state:
                        # Set p to y
                        p.lerp_(end=state['z'], weight=1-self.sf_momentum)
        self.train_mode = True

    @staticmethod
    def swap(x: torch.Tensor, y: torch.Tensor):
        # If this crashes use ScheduleFreeWrapperReference instead
        x.view(torch.uint8).bitwise_xor_(y.view(torch.uint8))
        y.view(torch.uint8).bitwise_xor_(x.view(torch.uint8))
        x.view(torch.uint8).bitwise_xor_(y.view(torch.uint8))

    @torch.no_grad()
    def step(self, closure: CLOSURE = None) -> LOSS:
        if not self.train_mode:
            raise Exception("Optimizer was not in train mode when step is called. "
                            "Please insert .train() and .eval() calls on the "
                            "optimizer. See documentation for details.")
        loss: LOSS = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            if 'sf_step' in group:
                group['sf_step'] += 1
            else:
                group['sf_step'] = 1

            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]

                if 'z' not in state:
                    state['z'] = torch.clone(p, memory_format=torch.preserve_format)

                z = state['z']

                p_fp32 = p

                # unpack
                if p.dtype in {torch.float16, torch.bfloat16}:
                    p_fp32 = p.clone().to(torch.float32)
                    z = z.to(torch.float32)

                # Apply weight_decay_at_y
                if self.sf_weight_decay_at_y != 0.0:
                    z.sub_(p_fp32, alpha=lr*self.sf_weight_decay_at_y)    
                    p_fp32.sub_(p_fp32, alpha=lr*self.sf_weight_decay_at_y*(1-self.sf_momentum))

                # Unextrapolate p converting from y -> x
                p_fp32.lerp_(end=z, weight=1-1/self.sf_momentum)

                # pack
                if p.dtype in {torch.float16, torch.bfloat16}:
                    copy_stochastic_(state["z"], z)
                    copy_stochastic_(p, p_fp32)

                z = state["z"]

                # Swap x into z buffer temporarily
                self.swap(z, p)

                # Now state['z'] is x and p is z.

        #######
        # Apply step to z
        self.base_optimizer.step()

        ######
        for group in self.param_groups:
            weight_lr_power = self.sf_weight_lr_power
            r = self.sf_r
            # tiny bit of starting LR to avoid divide by zero
            lr = max(group['lr'] * 1.0, 1e-8)
            lr_max = group['lr_max'] = max(lr, group.get('lr_max', 0))
            
            weight = (group['sf_step']**r) * (lr_max**weight_lr_power)
            weight_sum = group['sf_weight_sum'] = group.get('sf_weight_sum', 0.0) + weight

            ckp1 = weight/weight_sum

            for p in group['params']:
                if p.grad is None:
                    continue
                
                state = self.state[p]
                z = state['z']

                # Swap x back out of z buffer, leaving p as x
                self.swap(z, p)

                # Now state['z'] is z and p is x.

                p_fp32 = p

                # unpack
                if p.dtype in {torch.float16, torch.bfloat16}:
                    p_fp32 = p.clone().to(torch.float32)

                # Update x
                p_fp32.lerp_(end=z.to(torch.float32), weight=ckp1)

                # Now set p to y
                p_fp32.lerp_(end=state['z'].to(torch.float32), weight=1-self.sf_momentum)

                # pack
                if p.dtype in {torch.float16, torch.bfloat16}:
                    copy_stochastic_(p, p_fp32)

        return loss
    
    def load_state_dict(self, state_dict: Dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups

class ADOPTScheduleFree(BaseOptimizer):
    r"""Schedule-Free AdamW.
    Arguments:
        params (iterable):
            Iterable of parameters to optimize or dicts defining
            parameter groups.
        lr (float):
            Learning rate parameter (default 2.5e-3).
        betas (float, float):
            coefficients for momentum and exponential moving average squared (default: 0.9, 0.9999).
        eps (float):
            Term the denominator is minimally clamped to, to
            improve numerical stability. (default: 1e-6).
        eps2 (float):
            Term to multiple the RMS of the grad to calculate adaptive eps. (default: 1e-2).
        eps_floor (float):
            Term to set a floor for the eps, to prevent NaNs. (default: None, disabling adaptive eps).
        weight_decay (float):
            Weight decay at y, i.e. a L2 penalty (default: 0.0).
        weight_decouple (bool): 
            the optimizer uses decoupled weight decay as in AdamW.
        centralization (float):
            Center model grad (default: 0.0).
        adaptive_clip (float):
            Adaptive clip value to apply to the gradient first, before any further processing or use by the optimizer. (default: 1.0).
        adaptive_clip_eps (float):
            The eps for adaptive gradient clipping, provides a minimum to avoid parameters 
            not getting updating due to very small gradients being clipped excessively. (default: 1e-3).
        adaptive_clip_type (string):
            The type of clipping, can be unit or global. If done at the unit level can change
            the direction of the gradient, while global only scales down the magnitude of the entire gradient proportionally.
            Traditional adaptive clipping uses unit-wise, while this implementation also supports global.
            Valid values: global, unit (default: global).
        rectify: (bool)
            Rectify variance as per RAdam - https://arxiv.org/abs/1908.03265 (Default: false)
        bias_correction_beta2: (bool)
            Apply bias correction to denominator of updates (adaptive LR). i.e.  (Default: false)
        :param r: float. use polynomial weighting in the average with power r.
        :param weight_lr_power: float. during warmup, the weights in the average will be equal to lr raised to this power.
            set to 0 for no weighting.
        :param warmup_steps: int. enables a linear learning rate warmup.
    """

    def __init__(
        self,
        params: PARAMETERS,
        lr: float = 2.5e-3,
        betas: BETAS = (0.9, 0.9999),
        weight_decay: float = 0.0,
        weight_decouple: bool = False,
        stable_weight_decay: bool = False,
        r: float = 0.0,
        weight_lr_power: float = 2.0,
        warmup_steps: int = 0,
        eps: float = 1e-6,
        eps2: float = 1e-2,
        eps_floor: float = None,
        adaptive_clip: float = 1.0,
        adaptive_clip_eps: float = 1e-3,
        adaptive_clip_type: NORM_TYPE = 'layer',
        rectify: bool = False,
        bias_correction_beta2: bool = False,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_betas(betas)
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_non_negative(eps, 'eps')

        # Override zero to 1e-37, as zero and float32.tiny NaNs
        # Using 1e-37 as 1e-38 NaNs for Flux loras
        if eps_floor is not None and eps_floor < eps and eps_floor <= 0:
            eps_floor = 1e-37

        defaults: DEFAULTS = {
            'lr': lr,
            'betas': betas,
            'weight_decay': weight_decay,
            'weight_decouple':weight_decouple,
            'stable_weight_decay':stable_weight_decay,
            'r': r,
            'weight_lr_power': weight_lr_power,
            'warmup_steps': warmup_steps,
            'eps': eps,
            'eps2': eps2,
            'eps_floor':eps_floor,
            'train_mode': True,
            'weight_sum': 0.0,
            'lr_max': -1.0,
            'adaptive_clip':adaptive_clip,
            'adaptive_clip_eps':adaptive_clip_eps,
            'adaptive_clip_type':adaptive_clip_type,
            'rectify':rectify,
            'bias_correction_beta2':bias_correction_beta2,
        }
        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'ADOPTScheduleFree'

    def eval(self):
        for group in self.param_groups:
            beta1, _ = group['betas']
            if group['train_mode']:
                for p in group['params']:
                    state = self.state[p]
                    if 'z' in state:
                        p.data.lerp_(end=state['z'], weight=1.0 - 1.0 / beta1)
                group['train_mode'] = False

    def train(self):
        for group in self.param_groups:
            beta1, _ = group['betas']
            if not group['train_mode']:
                for p in group['params']:
                    state = self.state[p]
                    if 'z' in state:
                        p.data.lerp_(end=state['z'], weight=1.0 - beta1)
                group['train_mode'] = True

    @torch.no_grad()
    def reset(self):
        for group in self.param_groups:
            group['step'] = 0
            group['exp_avg_mean_sqrt'] = 0.0
            for p in group['params']:
                state = self.state[p]

                state['z'] = p.clone()
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
                group['exp_avg_mean_sqrt'] = 0.0

            param_size: int = 0
            exp_avg_sq_sum: float = 0.0

            warmup_steps: int = group['warmup_steps']
            schedule: float = group['step'] / warmup_steps if group['step'] < warmup_steps else 1.0

            beta1, beta2 = group['betas']

            beta2_t = beta2**group['step']
            bias_correction2 = 1 - beta2_t

            if group['rectify']:
                # maximum length of the approximated SMA
                rho_inf = 2 / (1 - beta2) - 1
                # compute the length of the approximated SMA
                rho_t = rho_inf - 2 * group['step'] * beta2_t / bias_correction2
                rect = (
                    ((rho_t - 4) * (rho_t - 2) * rho_inf / ((rho_inf - 4) * (rho_inf - 2) * rho_t)) ** 0.5
                    if rho_t > 4.0
                    else 0.0
                )
                lr: float = group['lr'] * rect
            else:
                lr: float = group['lr'] * schedule

            if not group['bias_correction_beta2']:
                bias_correction2 = 1.0

            lr_max = group['lr_max'] = max(lr, group['lr_max'])

            weight = (group['step'] ** group['r']) * (lr_max ** group['weight_lr_power'])
            weight_sum = group['weight_sum'] = group['weight_sum'] + weight

            checkpoint: float = weight / weight_sum if weight_sum != 0.0 else 0.0

            adaptive_y_lr: float = lr * (beta1 * (1.0 - checkpoint) - 1)
            adopt_clip: float = (group['step']-1)**0.25

            adaptive_clip = group["adaptive_clip"]
            adaptive_clip_type = group["adaptive_clip_type"]
            adaptive_clip_eps = group["adaptive_clip_eps"]
            eps = group["eps"]
            eps2 = group["eps2"]
            eps_floor = group["eps_floor"]

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise NoSparseGradientError(str(self))

                p_fp32 = p
                state = self.state[p]

                if group["weight_decay"] != 0 and group['weight_decouple'] and group['stable_weight_Decay']:
                    param_size += p.numel()                

                if len(state) == 0:
                    state['z'] = p.clone()
                    state['exp_avg_sq'] = torch.zeros_like(p)

                z, exp_avg_sq = state['z'], state['exp_avg_sq']

                # unpack
                if p.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.to(torch.float32)
                    z, exp_avg_sq = z.to(torch.float32), exp_avg_sq.to(torch.float32)
                    p_fp32 = p.clone().to(torch.float32)

                if adaptive_clip > 0.0:
                    # Apply Adaptive Gradient Clipping (AGC)
                    grad.copy_(agc(p_fp32, grad, adaptive_clip_eps, adaptive_clip, norm_type=adaptive_clip_type))

                if eps_floor is not None and eps_floor < eps:
                    rms_grad = grad.pow(2).mean().sqrt_()
                    curr_eps = max(min(eps, eps2 * rms_grad.item()), eps_floor) # Set a floor for eps to avoid NaN
                else:
                    curr_eps = eps

                if group['step'] == 1:
                    exp_avg_sq.addcmul_(grad, grad.conj())
                else:
                    if not group['rectify'] or rho_t > 4.0:
                        de_nom = exp_avg_sq.div(bias_correction2).sqrt_().clamp_(curr_eps)

                        # Reuse grad buffer for memory efficiency
                        update = grad.div(de_nom)
                        update.clamp_(-adopt_clip, adopt_clip)
                    else:
                        # Fall back to SGD (or nothing)
                        update = grad

                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad.conj(), value=1 - beta2)

                    # Weight decay calculated at y
                    if group["weight_decay"] != 0 and group['weight_decouple']:
                        if group['stable_weight_decay'] and group['exp_avg_mean_sqrt'] > 0:
                            swd_scaling = 1.0 / group['exp_avg_mean_sqrt']
                        else:
                            swd_scaling = 1.0

                        p_fp32.data.mul_(1.0 - group['weight_decay'] * group['lr'] * swd_scaling)
                    elif group["weight_decay"] != 0:
                        update.add_(p_fp32, alpha=group["weight_decay"])

                    p_fp32.lerp_(z, weight=checkpoint)
                    p_fp32.add_(update, alpha=adaptive_y_lr)

                    if group["weight_decay"] != 0 and group['weight_decouple'] and group['stable_weight_decay']:
                        exp_avg_sq_sum += exp_avg_sq.div(bias_correction2).sum()

                    z.sub_(update, alpha=lr)

                if group["weight_decay"] != 0 and group['weight_decouple'] and group['stable_weight_decay']:
                    group['exp_avg_mean_sqrt'] = math.sqrt(exp_avg_sq_sum / param_size)

                # pack
                if p.dtype in {torch.float16, torch.bfloat16}:
                    copy_stochastic_(state["z"], z)
                    copy_stochastic_(state["exp_avg_sq"], exp_avg_sq)
                    copy_stochastic_(p, p_fp32)

        return loss