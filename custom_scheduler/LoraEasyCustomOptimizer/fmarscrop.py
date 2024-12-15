# FMARSCrop from https://github.com/Clybius/Personalized-Optimizers by Clybius
import torch
from .utils import copy_stochastic_, agc, NORM_TYPE, schedule_beta, schedule_alpha
from typing import Literal, Optional
import math

from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.types import BETAS, CLOSURE, DEFAULTS, LOSS, PARAMETERS

class FMARSCrop(BaseOptimizer):
    r"""
    FMARSCrop: Fisher-accelerated MARS (https://arxiv.org/abs/2411.10438), with momentum-based Compass-style amplification, with ADOPT's AdamW changes (https://arxiv.org/abs/2411.02853).
    Un-official MARS implementation is credited to Less Wright (lessw2020).
    Arguments:
        params (iterable):
            Iterable of parameters to optimize or dicts defining
            parameter groups.
        lr (float):
            Learning rate parameter (default 0.0001).
        betas (float, float):
            coefficients used for computing running averages of
            gradient difference FIM and approx. natural grad FIM (default: 0.999, 0.9999).
        eps (float):
            Term the denominator is minimally clamped to, to
            improve numerical stability. (default: 1e-8).
        eps2 (float):
            Term to multiple the RMS of the grad to calculate adaptive eps. (default: 1e-2).
        eps_floor (float):
            Term to set a floor for the eps, to prevent NaNs. (default: None, disabling adaptive eps).
        weight_decay (float):
            Weight decay, i.e. a L2 penalty (default: 0.0).
        centralization (float):
            Center model grad (default: 0.0).
        moment_centralization (float):
            Center the slow momentum / EMA (default: 0.0).
        diff_mult (float):
            Multiplier for difference amplification (default: 1.0).
        momentum_lambda (float):
            The lambda value for slow momentum / EMA, controlling how much the momentum is amplified while being added to the update. (default: 2.0).
        momentum_beta (float):
            Beta value for slow momentum / EMA (default: 0.99).
        clip (float):
            Value to clip the grad's RMS at (default: 1.0)
        cautious (bool):
            Use cautious mask on parameter update - https://arxiv.org/abs/2411.16085 (default: True)
        adaptive_clip (float):
            Adaptive clip value to applied to the MARS corrected gradient. (default: 1.0).
        adaptive_clip_eps (float):
            The eps for adaptive gradient clipping, provides a minimum to avoid parameters 
            not getting updating due to very small gradients being clipped excessively. (default: 1e-3).
        adaptive_clip_type (string):
            The type of clipping, can be unit or layer. If done at the unit level can change
            the direction of the gradient, while layer only scales down the magnitude of the entire gradient proportionally.
            Traditional adaptive clipping uses unit-wise, while this implementation also supports layer.
            Valid values: layer, unit (default: layer).
        gamma (float):
            Scaling value for the MARS style correction of the gradient (default: 0.0005)
    """

    def __init__(
        self,
        params: PARAMETERS,
        lr: float = 5e-4,
        betas: BETAS = (0.999, 0.9999),
        eps: float = 1e-6,
        eps2: float = 1e-2,
        eps_floor: float = None,
        weight_decay: float = 0.0,
        centralization: float = 0.0,
        moment_centralization: float = 0.0,
        diff_mult: float = 1.0,
        momentum_lambda: float = 0.1,
        momentum_beta: float = 0.99,
        clip: float = 1.0,
        cautious: bool = True,
        gamma: float = 0.0005,
        adaptive_clip: float = 1.0,
        adaptive_clip_eps: float = 1e-3,
        adaptive_clip_type: NORM_TYPE = 'global',
        stable_weight_decay: bool = False,
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
            'lr':lr,
            'betas':betas,
            'eps':eps,
            'eps2':eps2,
            'eps_floor':eps_floor,
            'weight_decay':weight_decay,
            'centralization':centralization,
            'moment_centralization':moment_centralization,
            'diff_mult':diff_mult,
            'momentum_beta':momentum_beta,
            'momentum_lambda':momentum_lambda,
            'clip':clip,
            'cautious':cautious,
            'gamma': gamma,
            'adaptive_clip':adaptive_clip,
            'adaptive_clip_eps':adaptive_clip_eps,
            'adaptive_clip_type':adaptive_clip_type,
            'stable_weight_decay': stable_weight_decay,
        }

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'FMARSCrop'
    
    @torch.no_grad()
    def reset(self):
        for group in self.param_groups:
            group['step'] = 0
            group['fim_mean_sqrt'] = None
            for p in group["params"]:
                state = self.state[p]

                state["fim"] = torch.ones_like(p.data)
                # Fisher information matrix
                state["momentum"] = torch.zeros_like(p.data)
                # Prev grad
                state["prev_grad"] = torch.zeros_like(p.data).detach()
                if group["diff_mult"] > 0:
                    state["grad_diff_fim"] = torch.ones_like(p.data)

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
                group['fim_mean_sqrt'] = None

            param_size: int = 0
            fim_sum: float = 0.0

            beta1, beta2 = group["betas"]
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            centralization = group["centralization"]
            moment_centralization = group["moment_centralization"]
            diff_mult = group["diff_mult"]
            momentum_beta = group["momentum_beta"]
            momentum_lambda = group["momentum_lambda"]
            clip = group["clip"]
            step = group["step"]
            eps = group["eps"]
            eps2 = group["eps2"]
            eps_floor = group["eps_floor"]
            gamma = group["gamma"]
            adaptive_clip = group["adaptive_clip"]
            adaptive_clip_type = group["adaptive_clip_type"]
            adaptive_clip_eps = group["adaptive_clip_eps"]
            stable_weight_decay = group["stable_weight_decay"]

            clip_lambda = (step - 1)**0.25

            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]

                if stable_weight_decay:
                    param_size += p.numel()

                # State initialization
                if len(state) == 0:
                    state["momentum"] = torch.zeros_like(p)
                    state["fim"] = torch.ones_like(p)
                    state["prev_grad"] = -p.grad.to(dtype=p.dtype, copy=True).detach()
                    if diff_mult > 0:
                        state["grad_diff_fim"] = torch.ones_like(p)

                grad = p.grad

                p_fp32 = p

                prev_grad = state["prev_grad"]
                fim = state["fim"]
                momentum = state["momentum"]

                # Unpack
                if p.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.to(torch.float32)
                    fim = state["fim"].to(torch.float32)
                    momentum = state["momentum"].to(torch.float32)
                    prev_grad = state["prev_grad"].to(torch.float32)
                    p_fp32 = p.clone().to(torch.float32)

                prev_grad = prev_grad.add(grad)

                # Calculate cₜ (gradient with correction term)
                correction = (((1 - beta1) / 2) if gamma is None else gamma) * beta1 / (1 - beta1) * prev_grad
                c_t = grad + correction

                if adaptive_clip > 0.0:
                    # Apply Adaptive Gradient Clipping (AGC)
                    c_t.copy_(agc(p_fp32, c_t, adaptive_clip_eps, adaptive_clip, norm_type=adaptive_clip_type))

                if eps_floor is not None and eps_floor < eps:
                    rms_grad = c_t.pow(2).mean().sqrt_()
                    curr_eps = max(min(eps, eps2 * rms_grad.item()), eps_floor) # Set a floor for eps to avoid NaN
                else:
                    curr_eps = eps

                if diff_mult > 0:
                    # Get previous grad, initialized at 0 (first step is just grad)
                    # grad_diff will contain the difference between prev grad and current grad
                    grad_diff = prev_grad * diff_mult

                    rms = grad_diff.pow(2).mean().sqrt_()
                    divisor = max(clip, rms) / clip
                    grad_diff.div_(divisor)

                    grad_diff_fim = state["grad_diff_fim"]

                    # Unpack
                    if p.dtype in {torch.float16, torch.bfloat16}:
                        grad_diff_fim = state["grad_diff_fim"].to(torch.float32)

                    # Get natural gradient (squared ema, obtained sqrt of ema)
                    diff_fim_base = torch.clamp(grad_diff_fim.sqrt(), curr_eps)

                    grad_diff_fim.mul_(beta1).addcmul_(grad_diff, grad_diff, value=1.0 - beta1).clamp_(-clip_lambda, clip_lambda)

                    # pack
                    if p.dtype in {torch.float16, torch.bfloat16}:
                        copy_stochastic_(state["grad_diff_fim"], grad_diff_fim)
                else:
                    diff_fim_base = 1.0

                approx_grad_nat = c_t.div(diff_fim_base)
                rms = approx_grad_nat.pow(2).mean().sqrt_()
                divisor = max(clip, rms) / clip
                approx_grad_nat.div_(divisor)

                if group['step'] == 1:
                    fim.addcmul_(approx_grad_nat, approx_grad_nat)
                else:
                    fim_base = torch.clamp(fim.sqrt(), curr_eps)

                    grad_nat = approx_grad_nat.div(fim_base).div_(diff_fim_base)
                    rms = grad_nat.pow(2).mean().sqrt_()
                    divisor = max(clip, rms) / clip
                    grad_nat.div_(divisor)

                    momentum.mul_(momentum_beta).add_(grad_nat, alpha=1.0 - momentum_beta)

                    if moment_centralization != 0:
                        momentum_cent = momentum.sub(torch.mean(momentum).mul_(moment_centralization))
                    else:
                        momentum_cent = momentum

                    if group['cautious']:
                        mask = (momentum_cent * grad_nat < 0).to(momentum_cent.dtype)
                        mask.div_(mask.mean().clamp_(min=1e-3))
                        momentum_cent = momentum_cent * mask

                    # Compass-style amplification
                    full_step = grad_nat.add(momentum_cent, alpha=step**momentum_lambda)

                    # center the gradient vector
                    if centralization != 0 and full_step.dim() > 1:
                        full_step.sub_(
                            full_step.mean(dim=tuple(range(1, full_step.dim())), keepdim=True).mul_(
                                centralization
                            )
                        )

                    if weight_decay != 0:
                        # Perform weight decay
                        grad_weights = p_fp32.data.div(fim_base).div_(diff_fim_base)

                        rms = grad_weights.pow(2).mean().sqrt_()
                        divisor = max(clip, rms) / clip
                        grad_weights.div_(divisor)

                        if stable_weight_decay and group['fim_mean_sqrt'] is not None:
                            scale = 1.0 / group['fim_mean_sqrt']
                        else:
                            scale = 1.0

                        p_fp32.data.add_(grad_weights, alpha=-lr * weight_decay * scale)

                    if group["cautious"]:
                        mask = (full_step * grad_nat > 0).to(grad_nat.dtype)
                        mask.div_(mask.mean().clamp_(min=1e-3))
                    else:
                        mask = 1.0

                    # Apply full step
                    p_fp32.data.add_(full_step * mask, alpha=-lr)

                    fim.mul_(beta2).addcmul_(approx_grad_nat, approx_grad_nat, value=1.0 - beta2).clamp_(-clip_lambda, clip_lambda)

                if stable_weight_decay:
                    fim_sum += fim.sum()

                # pack
                if p.dtype in {torch.float16, torch.bfloat16}:
                    copy_stochastic_(state["fim"], fim)
                    copy_stochastic_(state["momentum"], momentum)
                    copy_stochastic_(state["prev_grad"], -grad)
                    copy_stochastic_(p, p_fp32)
                else:
                    # Copy the negative of the current grad (next step diff is -prev_grad + grad, or alternatively grad - prev_grad)
                    state["prev_grad"].copy_(-grad)

            if stable_weight_decay:
                group['fim_mean_sqrt'] = math.sqrt(fim_sum / param_size)

        return loss
