import torch
from typing import Any, Dict, List, Tuple, Union, Type, Literal, Optional
import torch.nn.functional as F
from torch.optim import Optimizer
from einops import rearrange
from pytorch_optimizer.optimizer.utils import unit_norm
import math

OPTIMIZER = Type[Optimizer]

NORM_TYPE = Literal['unit','global','layer']

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

# @torch.compile
def quantize(tensor, group_size=8, eps=1e-8, factor=3.2):
    shape = tensor.shape
    numel = tensor.numel()

    # just in case it's not divisible by group size
    padding = numel % group_size

    if padding != 0:
        tensor = rearrange(
            F.pad(tensor.flatten(), (0, padding), "constant", 0), "(r g) -> r g", g=2
        )
    else:
        tensor = rearrange(tensor.flatten(), "(r g) -> r g", g=group_size)
    scale = tensor.abs().max(dim=-1).values.unsqueeze(dim=-1)
    tensor /= scale + eps
    sign = tensor.sign()

    tensor = (
        ((torch.pow(tensor.abs(), 1 / factor) * sign + 1) * 127.5)
        .round()
        .to(dtype=torch.uint8)
    )
    if padding != 0:
        tensor = tensor.flatten()[:-padding]
    tensor = tensor.view(shape)
    return tensor, (scale, group_size, eps, factor, padding)


# @torch.compile
def dequantize(tensor, details, dtype=torch.float32):
    scale, group_size, eps, factor, padding = details
    shape = tensor.shape

    if padding != 0:
        tensor = rearrange(
            F.pad(tensor.flatten(), (0, padding), "constant", 0), "(r g) -> r g", g=2
        )
    else:
        tensor = rearrange(tensor.flatten(), "(r g) -> r g", g=group_size)
    tensor = tensor.to(dtype=dtype) / 127.5 - 1
    sign = tensor.sign()
    tensor = torch.pow(tensor.abs(), factor) * sign * scale
    if padding != 0:
        tensor = tensor.flatten()[:-padding]
    tensor = tensor.view(shape)

    return tensor
    
def agc(p: torch.Tensor, grad: torch.Tensor, agc_eps: float, agc_clip_val: float, eps: float = 1e-6, norm_type: NORM_TYPE = 'unit') -> torch.Tensor:
    r"""Clip gradient values in excess of the norm.
        Clip updates to be at most clipping * parameter_norm.

    References:
        [Brock, Smith, De, Simonyan 2021] High-Performance Large-Scale Image
        Recognition Without Normalization.
        
    :param p: torch.Tensor. parameter.
    :param grad: torch.Tensor, gradient.
    :param agc_eps: float. agc epsilon to clip the norm of parameter.
    :param agc_clip_val: float. norm clip.
    :param eps: float. simple stop from div by zero and no relation to standard optimizer eps.
    """
    if norm_type in {'global','layer'}:
        # Compute the global norm of the parameters and gradients
        p_norm = torch.norm(p).clamp_(min=agc_eps)
        g_norm = torch.norm(grad)

        # Compute the maximum allowed norm for the gradients
        max_norm = p_norm * agc_clip_val

        # Compute the clipping coefficient
        clip_coef = max_norm / g_norm.clamp(min=eps)

        # If the gradient norm exceeds the maximum allowed norm, scale the gradients
        if g_norm > max_norm:
            # Scale the gradients holistically
            grad = grad * clip_coef

        return grad
    elif norm_type == 'unit':
        p_norm = unit_norm(p).clamp_(agc_eps)
        g_norm = unit_norm(grad)

        max_norm = p_norm * agc_clip_val

        clipped_grad = grad * (max_norm / g_norm.clamp_min_(eps))

        return torch.where(g_norm > max_norm, clipped_grad, grad)
    else:
        raise ValueError(f"'{norm_type}' is not a supported value for norm_type.")


def schedule_alpha(t_alpha: Optional[float], step: int, alpha: float) -> float:
    if t_alpha is None:
        return alpha
    return min(step * alpha / t_alpha, alpha)


def schedule_beta(t_beta: Optional[float], step: int, beta_initial: float, beta_final: float, eps: float = 1e-8) -> float:
    if t_beta is None:
        return beta_initial

    # Add eps to prevent log 0
    log_beta_intial, log_beta_final = math.log(max(beta_initial, eps)), math.log(beta_final)

    return min(
        math.exp(
            log_beta_intial * log_beta_final / ((1.0 - step / t_beta) * log_beta_final + (step / t_beta) * log_beta_intial)
        ),
        beta_final,
    )

# Modified Adafactor factorisation implementation by Ross Wightman 
# https://github.com/huggingface/pytorch-image-models/pull/2320
@torch.no_grad()
def create_factored_dims(
    shape,
    factored,
    min_dim_size_to_factor):
    r"""Whether to use a factored second moment estimator.
    This function returns a tuple with the two largest axes to reduce over.
    If all dimensions have size < min_dim_size_to_factor, return None.
    Args:
    shape: an input shape
    factored: whether to use factored second-moment estimator for > 2d vars.
    min_dim_size_to_factor: only factor accumulator if all array dimensions are greater than this size.
    Returns:
    None or a tuple of ints
    """
    if not factored or len(shape) < 2:
        return None
    if all(dim < min_dim_size_to_factor for dim in shape):
        return None
    sorted_dims = sorted(((x, i) for i, x in enumerate(shape)))
    return int(sorted_dims[-2][1]), int(sorted_dims[-1][1])
    
# https://github.com/LoganBooker/prodigy-plus-schedule-free/blob/23f752a3901686d270dfdcb9b29823541ad1c3c7/prodigyplus/core_optimiser.py#L389
@torch.no_grad()
def get_denom(second_moment, eps: float = 1e-30):
    # Get denom
    if isinstance(second_moment, list):
        row_var, col_var, _, _, reduce_dc = second_moment

        row_col_mean = row_var.mean(dim=reduce_dc, keepdim=True).clamp_(eps)
        row_factor = row_var.div(row_col_mean).sqrt_()
        col_factor = col_var.sqrt()
        denom = row_factor * col_factor
    else:
        denom = second_moment.sqrt()

    return denom
    
# https://github.com/LoganBooker/prodigy-plus-schedule-free/blob/23f752a3901686d270dfdcb9b29823541ad1c3c7/prodigyplus/core_optimiser.py#L411
@torch.no_grad()
def update_second_moment(second_moment, grad, beta2, adopt_first: bool = False):
    # EMA updates
    if isinstance(second_moment, list):
        row_var, col_var, dr, dc, _ = second_moment
        if adopt_first:
            row_var.copy_(
                grad.norm(dim=dr, keepdim=True).square_().div_(grad.shape[dr])
            )
            col_var.copy_(
                grad.norm(dim=dc, keepdim=True).square_().div_(grad.shape[dc])
            )
        else:
            row_var.lerp_(
                grad.norm(dim=dr, keepdim=True).square_().div_(grad.shape[dr]),
                weight=1 - beta2
            )
            col_var.lerp_(
                grad.norm(dim=dc, keepdim=True).square_().div_(grad.shape[dc]),
                weight=1 - beta2
            )
    else:
        if adopt_first:
            second_moment.addcmul_(grad, grad)
        else:
            second_moment.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

    return second_moment

# From: https://github.com/KellerJordan/Muon/blob/master/muon.py
@torch.no_grad()
def newton_schulz_(grad, steps=6, eps=1e-7):
    # Inline reshaping step within the method itself.
    original_shape = None
    if len(grad.shape) > 2:
        original_shape = grad.shape
        grad = grad.view(grad.size(0), -1)
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = grad.bfloat16()
    if grad.size(0) > grad.size(1):
        X = X.T

    # Ensure spectral norm is at most 1
    X = X / (X.norm() + eps)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A # adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X

    if grad.size(0) > grad.size(1):
        X = X.T
    if X is not grad:
        grad.copy_(X)
        del X
    if original_shape is not None:
        grad = grad.view(*original_shape)
    return grad