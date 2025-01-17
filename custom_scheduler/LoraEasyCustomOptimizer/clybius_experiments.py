# MomentusCaution from https://github.com/Clybius/Personalized-Optimizers by Clybius

import torch
from torch.optim import Optimizer

def copy_stochastic_(target: torch.Tensor, source: torch.Tensor):
    # thanks to Nerogar for fast stochastic pytorch implementation
    # https://github.com/pytorch/pytorch/issues/120376#issuecomment-1974828905
    with torch.no_grad():
        # create a random 16 bit integer
        result = torch.randint_like(
            source,
            dtype=torch.int32,
            low=0,
            high=(1 << 16),
        )

        # add the random number to the lower 16 bit of the mantissa
        result.add_(source.view(dtype=torch.int32))

        # mask off the lower 16 bit of the mantissa
        result.bitwise_and_(-65536)  # -65536 = FFFF0000 as a signed int32

        # copy the higher 16 bit into the target tensor
        target.copy_(result.view(dtype=torch.float32))

# From pytorch_optimizer: https://github.com/kozistr/pytorch_optimizer
def unit_norm_func(x: torch.Tensor, norm: float = 2.0) -> torch.Tensor:
    r"""Get norm of unit."""
    keep_dim = True
    dim = None

    x_len: int = len(x.shape)
    if x_len <= 1:
        keep_dim = False
    elif x_len in (2, 3):
        dim = 1
    elif x_len == 4:
        dim = (1, 2, 3)
    else:
        dim = tuple(range(1, x_len))

    return x.norm(p=norm, dim=dim, keepdim=keep_dim)

def agc_global_norm(p: torch.Tensor, grad: torch.Tensor, agc_eps: float, agc_clip_val: float, eps: float = 1e-6, unit_norm: bool = 1) -> torch.Tensor:
    r"""Clip gradient values based on the global norm.
    Scale the entire gradient tensor if its norm exceeds a threshold.

    References:
        [Brock, Smith, De, Simonyan 2021] High-Performance Large-Scale Image
        Recognition Without Normalization.

    :param p: torch.Tensor. Parameter tensor.
    :param grad: torch.Tensor. Gradient tensor.
    :param agc_eps: float. A small epsilon value to prevent division by zero.
    :param agc_clip_val: float. Clipping threshold multiplier.
    :param eps: float. Small value to prevent division by zero in normalization.
    """
    func = unit_norm_func
    if not unit_norm:
        func = torch.linalg.norm
    # Compute the global norm of the parameters and gradients
    p_norm = func(p).clamp_(min=agc_eps)
    g_norm = func(grad)

    # Compute the maximum allowed norm for the gradients
    max_norm = p_norm * agc_clip_val

    clipped_grad = grad * (max_norm / g_norm.clamp_min_(eps))

    return torch.where(g_norm > max_norm, clipped_grad, grad)

class MomentusCaution(Optimizer):
    r"""
    MomentusCaution: MomentusCaution
    """

    def __init__(
        self,
        params,
        lr: float = 1e-4,
        beta: float = 0.9,
        momentum_beta: float = 0.0,
        weight_decay: float = 0.0,
        gamma_ratio: float = 0.5,
        adaptive_clip: float = 0.0,
        cautious: bool = True,
        nesterov: bool = False,
        **kwargs,
    ):

        defaults = dict(
            lr = lr,
            beta = beta,
            momentum_beta = momentum_beta,
            weight_decay = weight_decay,
            gamma_ratio = gamma_ratio,
            adaptive_clip = adaptive_clip,
            cautious = cautious,
            nesterov = nesterov
        )

        super(MomentusCaution, self).__init__(params, defaults)

    @torch.no_grad()
    def reset(self):
        for group in self.param_groups:
            group['step'] = 0
            for p in group["params"]:
                state = self.state[p]

                state["momentum"] = torch.zeros_like(p)
                if group["gamma_ratio"] != 0:
                    state["prev_grad"] = torch.zeros_like(p)
                    copy_stochastic_(state["prev_grad"], -p)
                if group["gamma_ratio"] > 0:
                    state["grad_momentum"] = torch.zeros_like(p)

    @torch.no_grad()
    def step(self, closure = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if 'step' in group:
                group['step'] += 1
            else:
                group['step'] = 1

            lr = group["lr"]
            beta = group["beta"]
            momentum_beta = group["momentum_beta"]
            weight_decay = group["weight_decay"]
            gamma_ratio = group["gamma_ratio"]
            adaptive_clip = group["adaptive_clip"]
            nesterov = group["nesterov"]
            step = group['step']

            curr_beta = beta * (1 - beta ** (step - 1)) / (1 - beta**step)
            curr_gamma = (1. - curr_beta) * gamma_ratio

            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                grad = p.grad.data
                p_fp32 = p

                # State initialization
                if len(state) == 0:
                    state["momentum"] = torch.zeros_like(p)
                    if gamma_ratio != 0:
                        state["prev_grad"] = torch.zeros_like(p)
                        copy_stochastic_(state["prev_grad"], -grad)
                    if momentum_beta > 0:
                        state["grad_momentum"] = torch.zeros_like(p)

                momentum = state["momentum"]

                # Unpack
                if p.dtype == torch.bfloat16:
                    grad = grad.to(torch.float32)
                    momentum = momentum.to(torch.float32)
                    p_fp32 = p.to(dtype=torch.float32, copy=True)

                if gamma_ratio != 0:
                    state['prev_grad'].add_(grad)

                    # Calculate cₜ (gradient with correction term)
                    correction = curr_gamma * curr_beta / (1 - curr_beta) * state['prev_grad']
                    c_t = (grad + correction)
                else:
                    c_t = grad

                # Gradient clipping (if necessary)
                if adaptive_clip > 0.0:
                    c_t = agc_global_norm(p, c_t, 1e-3, adaptive_clip, unit_norm=1)
                else:
                    grad_norm = torch.norm(c_t)
                    if grad_norm > 1.0:
                        c_t = c_t / grad_norm

                var_reduced_grad = c_t
                if momentum_beta > 0:
                    grad_momentum = state["grad_momentum"]

                    # Unpack
                    if p.dtype == torch.bfloat16:
                        grad_momentum = grad_momentum.to(torch.float32)

                    if nesterov:
                        grad_momentum.mul_(momentum_beta).add_(c_t)
                        var_reduced_grad = c_t.add(grad_momentum, alpha=momentum_beta).mul_(1. - momentum_beta)
                    else:
                        grad_momentum.mul_(momentum_beta).add_(c_t, alpha=1 - momentum_beta)
                        var_reduced_grad = grad_momentum
                    
                    # pack
                    if p.dtype == torch.bfloat16:
                        copy_stochastic_(state["grad_momentum"], grad_momentum)

                full_step = var_reduced_grad.div(momentum.sqrt().clamp_min_(1e-6))
                momentum.mul_(curr_beta).addcmul_(c_t, c_t, value=1 - curr_beta)

                if weight_decay != 0:
                    # Perform weight decay
                    full_step = full_step.add(p_fp32, alpha=weight_decay)

                # Apply full step
                if group['cautious']:
                    # Apply caution as per 'Cautious Optimizers' + 'Grams' - https://arxiv.org/abs/2411.16085 + https://arxiv.org/abs/2412.17107
                    mask = (full_step * grad > 0).to(full_step.dtype)
                    mask.div_(mask.mean().clamp_(min=1e-3))
                    full_step = torch.sign(c_t) * full_step.abs() * mask.clamp_min_(1)

                p_fp32.add_(full_step, alpha=-lr)

                # pack
                if p.dtype == torch.bfloat16:
                    copy_stochastic_(state["momentum"], momentum)
                    if gamma_ratio != 0:
                        copy_stochastic_(state["prev_grad"], -grad)
                    copy_stochastic_(p, p_fp32)
                elif gamma_ratio != 0:
                    # Copy the negative of the current grad (next step diff is -prev_grad + grad, or alternatively grad - prev_grad)
                    state["prev_grad"].copy_(-grad)
        return loss