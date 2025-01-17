import torch
from typing import Tuple, Union, Type, Literal, Optional
from torch.optim import Optimizer
import math

OPTIMIZER = Type[Optimizer]

NORM_TYPE = Literal['unit','global','layer']

CLIP_TYPE = Literal['unit','layer']

STATE_PRECISION = Literal['parameter', 'q4bit', 'q8bit', 'qfp8']

UPDATE_STRATEGY = Literal['unmodified','cautious','grams','both']

def unit_norm(x: torch.Tensor, norm: float = 2.0) -> torch.Tensor:
    r"""Get norm of unit."""
    keep_dim: bool = True
    dim: Optional[Union[int, Tuple[int, ...]]] = None

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
    
def agc(p: torch.Tensor, 
        grad: torch.Tensor, 
        agc_clip_val: float, 
        agc_eps: float = 1e-3, 
        eps: float = 1e-16, 
        norm_type: NORM_TYPE = 'layer') -> torch.Tensor:
    r"""Clip gradient values in excess of the norm.
        Clip updates to be at most clipping * parameter_norm.

    References:
        [Brock, Smith, De, Simonyan 2021] High-Performance Large-Scale Image
        Recognition Without Normalization.
        
    :param p: torch.Tensor. parameter.
    :param grad: torch.Tensor, gradient.
    :param agc_eps: float. Effectively sets a floor for the p_norm, as such, any gradients smaller than this will be clipped
        as though their parameter is at least agc_eps. This helps prevent vanishing gradients and excessive clipping early in training
        for small parameters.
    :param agc_clip_val: float. The desired clipping ratio, e.x. 0.5 would mean any gradient would be clipped to be no greater than half it's
        associated parameter.
    :param eps: float. simple stop from div by zero, as such should be as small as possible to avoid skewing clipping.
    """
    if norm_type in {'global','layer'}:
        # Compute the global norm of the parameters and gradients
        p_norm = torch.norm(p).clamp_(min=agc_eps)
        g_norm = torch.norm(grad)

        # Compute the maximum allowed norm for the gradients
        max_norm = p_norm * agc_clip_val

        # Compute the clipping coefficient
        clip_coef = max(1, max_norm / g_norm.clamp(min=eps))

        # Scale the gradients holistically
        grad = grad * clip_coef

        return grad
    elif norm_type == 'unit':
        p_norm = unit_norm(p).clamp_(min=agc_eps)
        g_norm = unit_norm(grad)

        max_norm = p_norm * agc_clip_val

        clipped_grad = grad * (max_norm / g_norm.clamp_(min=eps))

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

def schedule_beta_tc(t_beta: Optional[float], step: int, beta_initial: float, beta_final: float, eps: float = 1e-8) -> float:
    if t_beta is None:
        return beta_initial

    # Add eps to prevent log 0
    log_beta_intial, log_beta_final = math.log(max(beta_initial, eps)), math.log(beta_final)

    return min(
        torch.exp(
            log_beta_intial * log_beta_final / ((1.0 - step / t_beta) * log_beta_final + (step / t_beta) * log_beta_intial)
        ),
        beta_final,
    )

@torch.no_grad()
def spam_grad_clipping(grad: torch.Tensor, second_moment: torch.Tensor, clip_threshold: float, clip_type: CLIP_TYPE = 'unit') -> torch.Tensor:
    if clip_type == 'unit':
        # Calculate the clipping condition
        second_momentum_threshold = second_moment.mul(clip_threshold)
        second_momentum_threshold_sqrt = torch.sqrt(second_momentum_threshold)
        sign_grad = grad.sign()

        # Use torch.where instead of boolean masking
        return torch.where(
            grad.square() > second_momentum_threshold,
            sign_grad * second_momentum_threshold_sqrt,
            grad
        )
    elif clip_type == 'layer':
        # Calculate the global gradient norm
        max_norm = torch.norm(torch.sqrt(second_moment * clip_threshold))
        grad_norm = torch.norm(grad)

        # Calculate scaling factor for clipping
        scale = torch.where(
            grad_norm > max_norm,
            max_norm / grad_norm,
            torch.ones_like(grad_norm)
        )

        # Apply scaling to gradient
        return grad * scale
    

# Modified Adafactor factorisation implementation by Ross Wightman 
# https://github.com/huggingface/pytorch-image-models/pull/2320
@torch.no_grad()
def create_factored_dims(
    shape,
    factored: bool,
    min_dim_size_to_factor: int):
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
def get_denom(second_moment: torch.tensor, eps: float = 1e-16):
    # Get denom
    if isinstance(second_moment, list):
        row_var, col_var, _, _, reduce_dc = second_moment

        row_col_mean = row_var.mean(dim=reduce_dc, keepdim=True).add_(eps)
        row_factor = row_var.div(row_col_mean).sqrt_()
        col_factor = col_var.sqrt()
        denom = row_factor * col_factor
    else:
        denom = second_moment.sqrt()

    return denom
    
# https://github.com/LoganBooker/prodigy-plus-schedule-free/blob/23f752a3901686d270dfdcb9b29823541ad1c3c7/prodigyplus/core_optimiser.py#L411
@torch.no_grad()
def update_second_moment(second_moment: torch.tensor, grad: torch.tensor, beta2: float, adopt_first: bool = False) -> torch.tensor:
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
def newton_schulz(grad: torch.tensor, steps: int = 6, eps: float = 1e-7) -> torch.tensor:
    # Inline reshaping step within the method itself.
    original_shape = None
    original_type = grad.dtype
    working_grad = grad.clone()
    if len(working_grad.shape) > 2:
        original_shape = working_grad.shape
        working_grad = working_grad.view(working_grad.size(0), -1)
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = working_grad.bfloat16()
    if original_type in {torch.float32}:
        copy_stochastic_(X, working_grad)
    if working_grad.size(0) > working_grad.size(1):
        X = X.T

    # Ensure spectral norm is at most 1
    X = X / (X.norm() + eps)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A # adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X

    if working_grad.size(0) > working_grad.size(1):
        X = X.T
    if X is not working_grad:
        working_grad = working_grad.copy_(X)
        del X
    if original_shape is not None:
        working_grad = working_grad.view(*original_shape)

    return working_grad.to(dtype=original_type)

# Implementation from: https://github.com/LucasPrietoAl/grokking-at-the-edge-of-numerical-stability/blob/main/orthograd.py
def orthograd(param: torch.tensor, grad: torch.tensor, eps: float = 1e-30):
    w = param.view(-1)
    og_grad_shape = grad.shape
    grad = grad.view(-1)

    proj = torch.dot(w, grad) / (torch.dot(w, w) + eps)
    g_orth = grad.to(dtype=torch.float32, copy=True).add_(w, alpha=-proj)
    g_orth_scaled = g_orth.mul_(grad.norm(2) / (g_orth.norm(2) + eps))

    return g_orth_scaled.view(og_grad_shape)