import torch
from typing import Any, Dict, List, Tuple, Union, Type
import torch.nn.functional as F
from torch.optim import Optimizer
from einops import rearrange

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