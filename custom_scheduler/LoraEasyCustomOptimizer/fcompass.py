import torch
from torch.optim import Optimizer
from .utils import copy_stochastic_, quantize, dequantize
import math
from torch.nn.functional import softplus
from typing import Literal, Optional, List

from bitsandbytes.functional import quantize_blockwise, dequantize_blockwise
from pytorch_optimizer.base.exception import NoSparseGradientError, ZeroParameterSizeError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.types import BETAS, CLOSURE, DEFAULTS, LOSS, PARAMETERS
from pytorch_optimizer.optimizer.agc import agc
from pytorch_optimizer.optimizer.gc import centralize_gradient
from pytorch_optimizer.optimizer.utils import normalize_gradient, unit_norm

# Fisher optimizer (FAdam) from https://github.com/lessw2020/FAdam_PyTorch/blob/main/fadam.py by Less Wright (lessw2020), I may not know them, but I am aware of their expertise. Many thanks for your contributing work!
# Original optimizer (Compass) from https://github.com/lodestone-rock/compass_optimizer/blob/main/compass.py by lodestone-rock, many thanks for their optim, help, and ideas!
# FCompass from https://github.com/Clybius/Personalized-Optimizers/blob/main/FCompass.py by Clybius
# Defaults tuned for lora training based on testing
class FCompass(Optimizer):
    r"""
    Fisher Compass: Utilizing approximate fisher information to accelerate training. (Applied onto Compass).
    Arguments:
        params (iterable):
            Iterable of parameters to optimize or dicts defining
            parameter groups.
        lr (float):
            Learning rate parameter (default 7e-05)
        betas (Tuple[float, float], optional):
            coefficients used for computing running averages of
            gradient and its square (default: (0.98, 0.999)).
        amp_fac (float):
            amplification factor for the first moment filter (default: 2).
        eps (float):
            Term added to the denominator outside of the root operation to
            improve numerical stability. (default: 1e-8).
        eps2 (float):
            Term to multiple the RMS of the grad to calculate adaptive eps. (default: 0.01).
        eps_floor (float):
            Term to set a floor for the eps, to prevent NaNs. (default: 1e-30).
        weight_decay (float):
            Weight decay, i.e. a L2 penalty (default: 0.001).
        clip (float):
            Clip gradient to this value (default: 1.0).
        centralization (float):
            Center grad (default: 1.0).
    """

    def __init__(
        self,
        params,
        lr=7e-05, #Original default 1e-3
        betas=(0.98, 0.999), #Original default 0.99, 0.999
        amp_fac=2,
        eps=1e-8,
        eps2=1e-8,
        eps_floor=1e-30,
        weight_decay=0.001, #Original default 0.1
        clip=1.0,
        centralization=1.0,
    ):
        defaults = dict(
            lr=lr,
            betas=betas,
            amp_fac=amp_fac,
            eps=eps,
            eps2=eps2,
            eps_floor=eps_floor,
            weight_decay=weight_decay,
            clip=clip,
            centralization=centralization,
        )

        self.eps = eps
        self.eps2 = eps2
        self.eps_floor = eps_floor
        super(FCompass, self).__init__(params, defaults)

    def __str__(self) -> str:
        return 'FCompass'

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            if 'step' in group:
                group['step'] += 1
            else:
                group['step'] = 1

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("FCompass does not support sparse gradients")

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    # Exponential moving average and squared exponential moving average gradient values
                    state["momentum"] = torch.zeros_like(p.data)
                    state['max_ema_squared'] = torch.zeros_like(p.data)
                    # Fisher Information Matrix
                    state["fim"] = torch.ones_like(p.data)

                # unpack
                if p.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.to(torch.float32).data
                    momentum, fim, max_ema_squared = state["momentum"].to(torch.float32), state["fim"].to(torch.float32), state['max_ema_squared'].to(torch.float32)
                    p_fp32 = p.clone().to(torch.float32)
                else:
                    grad = grad.data
                    momentum, fim, max_ema_squared = state["momentum"], state["fim"], state['max_ema_squared']

                beta1, beta2 = group["betas"]
                amplification_factor = group["amp_fac"]
                lr = group["lr"]
                weight_decay = group["weight_decay"]
                clip = group["clip"]
                centralization = group["centralization"]

                # center the gradient vector
                if centralization != 0 and grad.dim() > 1:
                    grad.sub_(
                        grad.mean(dim=tuple(range(1, grad.dim())), keepdim=True).mul_(centralization)
                    )

                # bias correction step size
                # soft warmup
                curr_beta2 = (beta2**group["step"] - beta2) / (beta2**group["step"] - 1.0)
                bias_correction_sqrt = (1 - curr_beta2 ** group["step"]) ** (1 / 2)

                # Update fim
                fim.mul_(curr_beta2).addcmul_(grad, grad, value=1 - curr_beta2)

                rms_grad = grad.pow(2).mean().sqrt_()
                curr_eps = max(min(rms_grad.item() * self.eps2, self.eps), self.eps_floor) # Set a floor for eps to avoid NaN

                fim_base = fim**0.5 + curr_eps

                # Compute natural gradient
                grad_nat = grad / fim_base

                if clip != 0:
                    rms = grad_nat.pow(2).mean().sqrt_().add_(curr_eps)
                    divisor = max(1, rms) / clip
                    grad_nat.div_(divisor)

                # Momentum + Compass amplification
                momentum.mul_(beta1).add_(grad_nat, alpha=1 - beta1)
                grad_nat.add_(momentum, alpha=amplification_factor)

                # Weight decay
                if p.dtype in {torch.float16, torch.bfloat16}:
                    grad_weights = p_fp32.data / fim_base
                else:
                    grad_weights = p.data / fim_base

                if clip != 0:
                    rms = grad_weights.pow(2).mean().sqrt_().add_(curr_eps)
                    divisor = max(1, rms) / clip
                    grad_weights.div_(divisor)
                
                full_step = grad_nat + (weight_decay * grad_weights)

                # Use the max. for normalizing running avg. of gradient (amsgrad)
                torch.max(max_ema_squared, max_ema_squared.mul(beta2).addcmul_(full_step, full_step, value=1 - beta2), out=max_ema_squared)
                denom = (max_ema_squared.sqrt() / bias_correction_sqrt).add_(curr_eps)

                if p.dtype in {torch.float16, torch.bfloat16}:
                    p_fp32.data.addcdiv_(full_step, denom, value=-lr)
                else:
                    p.data.addcdiv_(full_step, denom, value=-lr)

                # pack
                if p.dtype in {torch.float16, torch.bfloat16}:
                    copy_stochastic_(state["momentum"], momentum)
                    copy_stochastic_(state["fim"], fim)
                    copy_stochastic_(state["max_ema_squared"], max_ema_squared)
                    copy_stochastic_(p, p_fp32)
        return loss

class FCompassPlus(BaseOptimizer):
    r"""
    Fisher Compass: Utilizing approximate fisher information to accelerate training. (Applied onto Compass).
    Arguments:
        params (iterable):
            Iterable of parameters to optimize or dicts defining
            parameter groups.
        lr (float):
            Learning rate parameter (default 7e-05)
        betas (Tuple[float, float], optional):
            coefficients used for computing running averages of
            gradient and its square (default: (0.98, 0.999)).
        amp_fac (float):
            amplification factor for the first moment filter (default: 2).
        eps (float):
            Term added to the denominator outside of the root operation to
            improve numerical stability. (default: 1e-8).
        eps2 (float):
            Term to multiple the RMS of the grad to calculate adaptive eps. (default: 0.01).
        eps_floor (float):
            Term to set a floor for the eps, to prevent NaNs. (default: 1e-30).
        weight_decay (float):
            Weight decay, i.e. a L2 penalty (default: 0.001).
        clip (float):
            Clip gradient to this value (default: 1.0).
        centralization (float):
            Center grad (default: 1.0).
        :param use_lookahead: bool. use lookahead. ADDS 1 STATE
        :param lookahead_merge_time: int. merge time.
        :param lookahead_blending_alpha: float. blending alpha.
    """

    def __init__(
        self,
        params: PARAMETERS,
        lr: float = 7e-05, #Original default 1e-3
        betas: BETAS = (0.98, 0.999), #Original default 0.99, 0.999
        amp_fac: float = 2.0,
        eps: float = 1e-8,
        eps2: float = 1e-8,
        eps_floor: float = 1e-30,
        weight_decay: float = 0.001, #Original default 0.1
        clip: float = 1.0,
        centralization: float = 1.0,
        use_lookahead: bool = False,
        lookahead_merge_time: int = 5,
        lookahead_blending_alpha: float = 0.5,
        use_softplus: bool = True,
        beta_softplus: float = 50.0,
        amsgrad: bool = True,
    ):
        defaults = dict(
            lr=lr,
            betas=betas,
            amp_fac=amp_fac,
            eps=eps,
            eps2=eps2,
            eps_floor=eps_floor,
            weight_decay=weight_decay,
            clip=clip,
            centralization=centralization,
            use_lookahead = use_lookahead,
            lookahead_merge_time = lookahead_merge_time,
            lookahead_blending_alpha = lookahead_blending_alpha,
            use_softplus = use_softplus,
            beta_softplus = beta_softplus,
            amsgrad = amsgrad,
        )

        self.eps = eps
        self.eps2 = eps2
        self.eps_floor = eps_floor
        self.use_lookahead = use_lookahead
        self.lookahead_merge_time = lookahead_merge_time
        self.lookahead_blending_alpha = lookahead_blending_alpha
        self.use_softplus = use_softplus
        self.beta_softplus = beta_softplus
        self.amsgrad = amsgrad
        super(FCompassPlus, self).__init__(params, defaults)

    def __str__(self) -> str:
        return 'FCompassPlus'
    
    @torch.no_grad()
    def reset(self):
        for group in self.param_groups:
            group['step'] = 0
            for p in group['params']:
                state = self.state[p]

                # Exponential moving average and squared exponential moving average gradient values
                state["momentum"] = torch.zeros_like(p)
                state['ema_squared'] = torch.zeros_like(p)
                # Fisher Information Matrix
                state["fim"] = torch.ones_like(p)

                if self.use_lookahead:
                    state['lookahead_params'] = p.clone()

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

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("FCompassPlus does not support sparse gradients")

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    # Exponential moving average and squared exponential moving average gradient values
                    state["momentum"] = torch.zeros_like(p)
                    state['ema_squared'] = torch.zeros_like(p)
                    # Fisher Information Matrix
                    state["fim"] = torch.ones_like(p)

                    if self.use_lookahead:
                        state['lookahead_params'] = p.clone()


                momentum, fim, ema_squared = state["momentum"], state["fim"], state['ema_squared']
                p_fp32 = p

                # unpack
                if p.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.to(torch.float32)
                    momentum, fim, ema_squared = momentum.to(torch.float32), fim.to(torch.float32), ema_squared.to(torch.float32)
                    p_fp32 = p.clone().to(torch.float32)

                beta1, beta2 = group["betas"]
                amplification_factor = group["amp_fac"]
                lr = group["lr"]
                weight_decay = group["weight_decay"]
                clip = group["clip"]
                centralization = group["centralization"]

                # center the gradient vector
                if centralization != 0 and grad.dim() > 1:
                    grad.sub_(
                        grad.mean(dim=tuple(range(1, grad.dim())), keepdim=True).mul_(centralization)
                    )

                # bias correction step size
                # soft warmup
                curr_beta2 = (beta2**group["step"] - beta2) / (beta2**group["step"] - 1.0)
                bias_correction_sqrt = (1 - curr_beta2 ** group["step"]) ** (1 / 2)

                # Update fim
                fim.mul_(curr_beta2).addcmul_(grad, grad, value=1 - curr_beta2)

                rms_grad = grad.pow(2).mean().sqrt_()
                curr_eps = max(min(rms_grad.item() * self.eps2, self.eps), self.eps_floor) # Set a floor for eps to avoid NaN

                fim_base = fim**0.5 + curr_eps

                # Compute natural gradient
                grad_nat = grad / fim_base

                if clip != 0:
                    rms = grad_nat.pow(2).mean().sqrt_().add_(curr_eps)
                    divisor = max(1, rms) / clip
                    grad_nat.div_(divisor)

                # Momentum + Compass amplification
                momentum.mul_(beta1).add_(grad_nat, alpha=1 - beta1)
                grad_nat.add_(momentum, alpha=amplification_factor)

                grad_weights = p_fp32.data / fim_base
  
                if clip != 0:
                    rms = grad_weights.pow(2).mean().sqrt_().add_(curr_eps)
                    divisor = max(1, rms) / clip
                    grad_weights.div_(divisor)
                
                # Weight decay
                full_step = grad_nat + (weight_decay * grad_weights)

                # Use the max. for normalizing running avg. of gradient (amsgrad)
                if self.amsgrad:
                    torch.max(ema_squared, ema_squared.mul(beta2).addcmul_(full_step, full_step, value=1 - beta2), out=ema_squared)
                    de_nom = (ema_squared.sqrt() / bias_correction_sqrt).add_(curr_eps)
                else:
                    ema_squared.mul_(beta2).addcmul_(full_step, full_step, value=1 - beta2)
                    de_nom = (ema_squared.sqrt() / bias_correction_sqrt).add_(curr_eps)

                if self.use_softplus:
                    de_nom = softplus(de_nom, beta=self.beta_softplus, threshold=self.threshold_softplus if self.threshold_softplus != 0 else curr_eps)

                p_fp32.data.addcdiv_(full_step, de_nom, value=-lr)

                # pack
                if p.dtype in {torch.float16, torch.bfloat16}:
                    copy_stochastic_(state["momentum"], momentum)
                    copy_stochastic_(state["fim"], fim)
                    copy_stochastic_(state["ema_squared"], ema_squared)
                    copy_stochastic_(p, p_fp32)

        if self.use_lookahead:
            self.lookahead_process_step()

        return loss
    
    def lookahead_process_step(self):
        self.lookahead_step += 1
        if self.lookahead_step >= self.lookahead_merge_time:
            self.lookahead_step: int = 0
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None:
                        continue

                    state = self.state[p]

                    p_fp32 = p

                    lookahead_params = state['lookahead_params']

                    if p.dtype in {torch.float16, torch.bfloat16}:
                        p_fp32 = p.clone().to(torch.float32)
                        lookahead_params = lookahead_params.to(torch.float32)

                    p_fp32.mul_(self.lookahead_blending_alpha).add_(
                        lookahead_params,
                        alpha=1.0 - self.lookahead_blending_alpha,
                    )

                    # pack
                    if p.dtype in {torch.float16, torch.bfloat16}:
                        copy_stochastic_(p, p_fp32)

                    state['lookahead_params'].copy_(p)