# FMARSCrop from https://github.com/Clybius/Personalized-Optimizers by Clybius
import torch
from .utils import copy_stochastic_, agc, NORM_TYPE
from typing import Literal

from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.types import BETAS, CLOSURE, DEFAULTS, LOSS, PARAMETERS

MASK_GRADS = Literal['grad', 'corrected_grad_pre_clip','corrected_grad', 'approx_grad_nat' 'grad_nat']


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
            improve numerical stability. (default: 1e-6).
        eps2 (float):
            Term to multiple the RMS of the grad to calculate adaptive eps. (default: 1e-2).
        eps_floor (float):
            Term to set a floor for the eps, to prevent NaNs. (default: None, disabling adaptive eps).
        weight_decay (float):
            Weight decay, i.e. a L2 penalty (default: 0.0).
        centralization (float):
            Center model grad (default: 0.0).
        moment_centralization (float):
            Center the slow momentum / EMA (default: 1.0).
        diff_mult (float):
            Multiplier for difference amplification (default: 1.0).
        momentum_beta (float):
            Beta value for slow momentum / EMA (default: 0.9999) (Alternative recommendation: 0.99999).
        momentum_lambda (float):
            Amplification exponent for slow momentum / EMA (default: 0.25) (Alternative recommendation: 0.5).
        clip (float):
            Value to clip the grad's RMS at (default: 1.0)
        cautious (bool):
            Use cautious mask on parameter update - https://arxiv.org/abs/2411.16085 (default: False)
        cautious_grad (str):
            Which form of grad to use for the cautious mask, valid options are 'grad', 'corrected_grad_pre_clip', 'corrected_grad', 'approx_grad_nat' 'grad_nat' (Default: corrected_grad)
        cautious_momentum (bool):
            Only effective if cautious is True, controls if cautious mask is also applied to the momentum update. 
            Experimental, doesn't align with original impl, may not behave as expected or produce better results. (default: False)
        og_approx_grad_nat (bool):
            Enables old, technicially unintended and likely incorrect way of handling approx_grad_nat that would cause it to replace
            the original corrected grad in place, thus resulting in it being used as grad for grad_nat. (Default: False)
        adaptive_clip (float):
            Adaptive clip value to apply to the gradient first, before any further processing or use by the optimizer. (default: 0.0).
        adaptive_clip_eps (float):
            The eps for adaptive gradient clipping, provides a minimum to avoid parameters 
            not getting updating due to very small gradients being clipped excessively. (default: 1e-3).
        adaptive_clip_type (string):
            The type of clipping, can be unit or global. If done at the unit level can change
            the direction of the gradient, while global only scales down the magnitude of the entire gradient proportionally.
            Traditional adaptive clipping uses unit-wise, while this implementation also supports global.
            Valid values: global, unit (default: global).
    """

    def __init__(
        self,
        params: PARAMETERS,
        lr: float = 1e-4,
        betas: BETAS = (0.999, 0.9999),
        eps: float = 1e-6,
        eps2: float = 1e-2,
        eps_floor: float = None,
        weight_decay: float = 0.0,
        centralization: float = 0.0,
        moment_centralization: float = 1.0,
        diff_mult: float = 1.0,
        momentum_beta: float = 0.9999,
        momentum_lambda: float = 0.25,
        clip: float = 1.0,
        cautious: bool = False,
        cautious_grad: MASK_GRADS = 'corrected_grad',
        cautious_momentum: bool = False,
        og_approx_grad_nat: bool = False,
        gamma: float = None,
        adaptive_clip: float = 0.0,
        adaptive_clip_eps: float = 1e-3,
        adaptive_clip_type: NORM_TYPE = 'global',
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_betas(betas)
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_non_negative(eps, 'eps')

        # Override zero to 1e-38, as zero and float32.tiny NaNs
        if eps_floor is not None and eps_floor < eps and eps_floor <= 0:
            eps_floor = 1e-38

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
            'cautious_grad':cautious_grad,
            'cautious_momentum':cautious_momentum,
            'og_approx_grad_nat':og_approx_grad_nat,
            'gamma': gamma,
            'adaptive_clip':adaptive_clip,
            'adaptive_clip_eps':adaptive_clip_eps,
            'adaptive_clip_type':adaptive_clip_type,
        }

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'FMARSCrop'
    
    @torch.no_grad()
    def reset(self):
        for group in self.param_groups:
            group['step'] = 0
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
            og_approx_grad_nat = group["og_approx_grad_nat"]
            cautious_grad = group["cautious_grad"]
            gamma = group["gamma"]
            adaptive_clip = group["adaptive_clip"]
            adaptive_clip_type = group["adaptive_clip_type"]
            adaptive_clip_eps = group["adaptive_clip_eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["momentum"] = torch.zeros_like(p)
                    state["fim"] = torch.ones_like(p)
                    state["prev_grad"] = -p.grad.clone().to(p.dtype).detach()
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

                if adaptive_clip > 0.0:
                    # Apply Adaptive Gradient Clipping (AGC)
                    grad.copy_(agc(p_fp32, grad, adaptive_clip_eps, adaptive_clip, norm_type=adaptive_clip_type))

                prev_grad = prev_grad.add(grad)

                # Calculate cₜ (gradient with correction term)
                #correction = (1 - beta1) / 2 * beta1 / (1 - beta1) * prev_grad
                correction = (((1 - beta1) / 2) if gamma is None else gamma) * beta1 / (1 - beta1) * prev_grad
                c_t = grad + correction
                pre_clip_c_t = c_t.clone().detach()

                # Gradient clipping (if necessary)
                grad_norm = torch.norm(c_t)
                if grad_norm > clip:
                    c_t = c_t * clip / grad_norm

                clip_lambda = step**0.25

                fim_slow_beta = ((beta2**step - beta2) / (beta2**step - 1.0)) ** (1/2)

                if eps_floor is not None and eps_floor < eps:
                    rms_grad = grad.pow(2).mean().sqrt_()
                    curr_eps = max(min(eps, eps2 * rms_grad.item()), eps_floor) # Set a floor for eps to avoid NaN
                else:
                    curr_eps = eps

                if diff_mult > 0:
                    # Get previous grad, initialized at 0 (first step is just grad)
                    #prev_grad = prev_grads[i]
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

                    grad_diff_fim.mul_(beta1).addcmul_(grad_diff, grad_diff, value=1 - beta1).clamp_(-clip_lambda, clip_lambda)

                    # pack
                    if p.dtype in {torch.float16, torch.bfloat16}:
                        copy_stochastic_(state["grad_diff_fim"], grad_diff_fim)
                else:
                    diff_fim_base = 1.0

                clipped_c_t = c_t.clone().detach()

                if og_approx_grad_nat:
                    approx_grad_nat = c_t
                    approx_grad_nat.div_(diff_fim_base)
                else:
                    approx_grad_nat = c_t.div(diff_fim_base)
                rms = approx_grad_nat.pow(2).mean().sqrt_()
                divisor = max(clip, rms) / clip
                approx_grad_nat.div_(divisor)

                fim_base = torch.clamp(fim.sqrt(), curr_eps)

                grad_nat = c_t.div(fim_base).div_(diff_fim_base)
                rms = grad_nat.pow(2).mean().sqrt_()
                divisor = max(clip, rms) / clip
                grad_nat.div_(divisor)

                if moment_centralization != 0:
                    momentum_cent = momentum - torch.mean(momentum)
                else:
                    momentum_cent = momentum

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

                    p_fp32.data.add_(grad_weights, alpha=-lr*weight_decay)

                # Apply full step
                if group["cautious"]:
                    if cautious_grad == 'grad':
                        grad_for_mask = grad
                    elif cautious_grad == 'corrected_grad_pre_clip':
                        grad_for_mask = pre_clip_c_t
                    elif cautious_grad == 'corrected_grad':
                        grad_for_mask = clipped_c_t
                    elif cautious_grad == 'approx_grad_nat':
                        grad_for_mask = approx_grad_nat
                    elif cautious_grad == 'grad_nat':
                        grad_for_mask = grad_nat
                    # compute norm gradient
                    mask = (full_step * grad_for_mask > 0).to(grad.dtype)
                    mask.mul_(mask.numel() / (mask.sum() + 1))
                else:
                    mask = 1.0

                # Apply full step
                p_fp32.data.add_(full_step * mask, alpha=-lr)

                fim.mul_(fim_slow_beta).addcmul_(approx_grad_nat, approx_grad_nat, value=1 - fim_slow_beta).clamp_(-clip_lambda, clip_lambda)

                momentum.mul_(momentum_beta).add_(grad_nat * (mask if group["cautious_momentum"] else 1.0), alpha=1 - momentum_beta)

                # pack
                if p.dtype in {torch.float16, torch.bfloat16}:
                    copy_stochastic_(state["fim"], fim)
                    copy_stochastic_(state["momentum"], momentum)
                    copy_stochastic_(state["prev_grad"], -grad)
                    copy_stochastic_(p, p_fp32)
                else:
                    # Copy the negative of the current grad (next step diff is -prev_grad + grad, or alternatively grad - prev_grad)
                    state["prev_grad"].copy_(-grad)
        return loss