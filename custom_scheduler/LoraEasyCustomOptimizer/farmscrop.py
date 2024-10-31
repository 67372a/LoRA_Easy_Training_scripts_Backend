# FARMSCrop from https://github.com/Clybius/Personalized-Optimizers/blob/main/FARMSCrop.py by Clybius
import torch
from torch.optim import Optimizer
from .utils import copy_stochastic_

class FARMSCrop(Optimizer):
    r"""
    FARMSCrop: Fisher-Accelerated RMSProp, replaced denom with momentum and compass-style amplification.
    Arguments:
        params (iterable):
            Iterable of parameters to optimize or dicts defining
            parameter groups.
        lr (float):
            Learning rate parameter (default 0.0001)
        betas (float, float):
            coefficients used for computing running averages of
            gradient difference FIM and approx. natural grad FIM (default: 0.999, 0.9999).
        eps (float):
            Term added to the denominator outside of the root operation to
            improve numerical stability. (default: 1e-8).
        eps2 (float):
            Term to multiple the RMS of the grad to calculate adaptive eps. (default: 0.01).
        eps_floor (float):
            Term to set a floor for the eps, to prevent NaNs. (default: 1e-30).
        weight_decay (float):
            Weight decay, i.e. a L2 penalty (default: 1e-6).
        centralization (float):
            center model grad (default: 1.0).
        diff_mult (float):
            Multiplier for difference amplification (default: 1.0)
        momentum_beta (float):
            Beta value for slow momentum / EMA (default: 0.9999)
        momentum_amp (float):
            Amplification multiplier for slow momentum / EMA (default: 5.0)
    """

    def __init__(
        self,
        params,
        lr=1e-4,
        betas=(0.999, 0.9999),
        eps=1e-8,
        eps2=0.01,
        eps_floor=1e-16,
        weight_decay=1e-6,
        centralization=1.0,
        diff_mult=1.0,
        momentum_beta=0.9999,
        momentum_amp=5.0,
    ):
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            eps2=eps2,
            eps_floor=eps_floor,
            weight_decay=weight_decay,
            centralization=centralization,
            diff_mult=diff_mult,
            momentum_beta=momentum_beta,
            momentum_amp=momentum_amp,
        )

        self.eps = eps
        self.eps2 = eps2
        self.eps_floor = eps_floor
        super(FARMSCrop, self).__init__(params, defaults)

    def __str__(self) -> str:
        return 'FARMSCrop'

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

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    # Fisher information matrix
                    state["fim"] = torch.ones_like(p.data)
                    # Fisher information matrix
                    state["momentum"] = torch.zeros_like(p.data)
                    # Prev grad
                    state["previous_grad"] = torch.zeros_like(p.data)
                    state["grad_diff_fim"] = torch.ones_like(p.data)
                    
                # Unpack
                if p.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.to(torch.float32)
                    fim = state["fim"].to(torch.float32)
                    momentum = state["momentum"].to(torch.float32)
                    prev_grad = state["previous_grad"].to(torch.float32)
                    grad_diff_fim = state["grad_diff_fim"].to(torch.float32)
                    p_fp32 = p.clone().to(torch.float32)
                else:
                    fim = state["fim"]
                    momentum = state["momentum"]
                    prev_grad = state["previous_grad"]
                    grad_diff_fim = state["grad_diff_fim"]

                beta1, beta2 = group["betas"]
                lr = group["lr"]
                weight_decay = group["weight_decay"]
                centralization = group["centralization"]
                diff_mult = group["diff_mult"]
                momentum_beta = group["momentum_beta"]
                momentum_amp = group["momentum_amp"]

                # bias correction step size
                #bias_correction_sqrt = (1 - beta2 ** group["step"]) ** (1 / 2)
                fim_slow_beta = ((beta2**group["step"] - beta2) / (beta2**group["step"] - 1.0)) ** (1/2)
                step_size = lr

                # Get previous grad, initialized at 0 (first step is just grad)
                # grad_diff will contain the difference between prev grad and current grad
                grad_diff = prev_grad.add(grad) * diff_mult

                grad_diff_fim.mul_(beta1).addcmul_(grad_diff, grad_diff, value=1 - beta1)

                rms_grad = grad.pow(2).mean().sqrt_()
                curr_eps = max(min(rms_grad.item() * self.eps2, self.eps), self.eps_floor) # Set a floor for eps to avoid NaN

                # Get natural gradient (squared ema, obtained sqrt of ema)
                diff_fim_base = grad_diff_fim.sqrt().add_(curr_eps)

                approx_grad_nat = grad.div(diff_fim_base)

                rms = approx_grad_nat.pow(2).mean().sqrt_()
                divisor = max(1, rms)
                approx_grad_nat.div_(divisor)

                fim.mul_(fim_slow_beta).addcmul_(approx_grad_nat, approx_grad_nat, value=1 - fim_slow_beta)
                fim_base = fim.sqrt().add_(curr_eps)

                grad_nat = grad.div(fim_base).mul_(diff_fim_base)
                rms = grad_nat.pow(2).mean().sqrt_()
                divisor = max(1, rms)
                grad_nat.div_(divisor)

                # center the gradient vector
                if centralization != 0 and grad_nat.dim() > 1:
                    grad_nat.sub_(
                        grad_nat.mean(dim=tuple(range(1, grad_nat.dim())), keepdim=True).mul_(
                            centralization
                        )
                    )

                # Compass-style amplification
                momentum.mul_(momentum_beta).add_(grad_nat, alpha=1 - momentum_beta)
                full_step = grad_nat.add(momentum, alpha=momentum_amp)

                if weight_decay != 0:
                    # Perform weight decay
                    if p.dtype in {torch.float16, torch.bfloat16}:
                        grad_weights = p_fp32.data / fim_base * diff_fim_base
                    else:
                        grad_weights = p.data / fim_base * diff_fim_base

                    rms = grad_weights.pow(2).mean().sqrt_()
                    divisor = max(1, rms)
                    grad_weights.div_(divisor)

                    full_step.add_(grad_weights, alpha=weight_decay)

                # Apply full step
                if p.dtype in {torch.float16, torch.bfloat16}:
                    p_fp32.data.add_(full_step, alpha=-step_size)
                else:
                    p.data.add_(full_step, alpha=-step_size)
                    
                # pack
                if p.dtype in {torch.float16, torch.bfloat16}:
                    copy_stochastic_(state["fim"], fim)
                    copy_stochastic_(state["momentum"], momentum)
                    copy_stochastic_(state["previous_grad"], -grad)
                    copy_stochastic_(state["grad_diff_fim"], grad_diff_fim)
                    copy_stochastic_(p, p_fp32)
                else:
                    # Copy the negative of the current grad (next step diff is -prev_grad + grad, or alternatively grad - prev_grad)
                    state['previous_grad'].copy_(-grad)
        return loss
