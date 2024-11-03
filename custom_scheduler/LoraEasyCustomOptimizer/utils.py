import torch
from typing import Any, Dict, List, Tuple, Union, Type, Literal
import torch.nn.functional as F
from torch.optim import Optimizer
from einops import rearrange
from pytorch_optimizer.optimizer.utils import unit_norm

OPTIMIZER = Type[Optimizer]

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
    
def agc(p: torch.Tensor, grad: torch.Tensor, agc_eps: float, agc_clip_val: float, eps: float = 1e-6) -> torch.Tensor:
    r"""Clip gradient values in excess of the unit wise norm.
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
    p_norm = unit_norm(p).clamp_(agc_eps)
    g_norm = unit_norm(grad)

    max_norm = p_norm * agc_clip_val

    clipped_grad = grad * (max_norm / g_norm.clamp_min_(eps))

    return torch.where(g_norm > max_norm, clipped_grad, grad)