# FFTDescent from https://github.com/Clybius/Personalized-Optimizers by Clybius

import torch
from torch.optim import Optimizer
from math import sqrt
from typing import Callable, Tuple
import math
import logging

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

# Original Spectral Clipping code by leloykun (https://leloykun.github.io/ponder/spectral-clipping/ https://github.com/leloykun/spectral_clip)

"""
@misc{cesista2025spectralclipping,
  author = {Franz Louis Cesista},
  title = {"Fast, Numerically Stable, and Auto-Differentiable Spectral Clipping Via Newton-Schulz Iteration"},
  year = {2025},
  url = {http://leloykun.github.io/ponder/spectral-clipping/},
}
"""
NS_COEFFS = [
    (3.5318, -4.7911, 1.9388),
    (3.3274, -4.0557, 1.5782),
    (3.0809, -3.5160, 1.3464),
    (2.7476, -2.8484, 1.0775),
    (2.2948, -2.0951, 0.7895),
    (2.1535, -1.8338, 0.6869),
]
# New coeffs from https://kexue.fm/archives/11059, may enable later.
"""
NS_COEFFS = [
    (8.287212018145622, -23.59588651909882, 17.300387312530923),
    (4.107059111542197, -2.9478499167379084, 0.54484310829266),
    (3.9486908534822938, -2.908902115962947, 0.5518191394370131),
    (3.3184196573706055, -2.488488024314878, 0.5100489401237208),
    (2.3006520199548186, -1.6689039845747518, 0.4188073119525678),
    (1.8913014077874002, -1.2679958271945908, 0.37680408948524996),
    (1.875, -1.25, 0.375)
]
"""
@torch.no_grad()
def orthogonalize(M: torch.Tensor, num_ns_steps=len(NS_COEFFS), ortho_dtype=None, adaptive=False) -> torch.Tensor:
    """Orthogonalize a matrix via 5th order Newton-Schulz iteration."""
    if ortho_dtype is not None:
        orig_dtype = M.dtype
        M = M.to(ortho_dtype)
    if adaptive:
        M_orig = M.clone()
    transpose = M.shape[0] < M.shape[1]
    if transpose:
        M = M.T
    M = M / (torch.linalg.norm(M) + 1e-20)
    for a, b, c in NS_COEFFS[:num_ns_steps]:
        A = M.T @ M
        I = torch.eye(A.shape[0], dtype=M.dtype, device=M.device)
        M = M @ (a * I + b * A + c * A @ A)
    if transpose:
        M = M.T
    if adaptive:
        M = torch.einsum('ij,ij,ab->ab', M_orig.type_as(M), M, M)
    if ortho_dtype is not None:
        M = M.to(orig_dtype)
    return M

def _spectral_clip(W: torch.Tensor, sigma_min: float=-1., sigma_max: float=1., ortho_dtype=torch.float32, num_ns_steps=len(NS_COEFFS), adaptive=False):
    if adaptive:
        W_orig = W.clone()
    orig_dtype = W.dtype
    W = W.to(ortho_dtype)
    OW = orthogonalize(W, num_ns_steps)
    eye_m = torch.eye(W.shape[0], dtype=W.dtype, device=W.device)
    result = (1/2) * (
        (sigma_min + sigma_max) * eye_m
        + (sigma_min * OW - W) @ orthogonalize(sigma_min * OW - W, num_ns_steps).T
        - (sigma_max * OW - W) @ orthogonalize(sigma_max * OW - W, num_ns_steps).T
    ) @ OW
    if adaptive:
        result = torch.einsum('ij,ij,ab->ab', W_orig.type_as(result), result, result)
    return result.to(orig_dtype)

@torch.no_grad()
def spectral_clip_func(W: torch.Tensor, sigma_min: float=-1., sigma_max: float=1., ortho_dtype=None, num_ns_steps=len(NS_COEFFS), adaptive=False):
    if ortho_dtype is None:
        ortho_dtype = torch.float32
    return  _spectral_clip(W, sigma_min=sigma_min, sigma_max=sigma_max, ortho_dtype=ortho_dtype, num_ns_steps=num_ns_steps, adaptive=adaptive)

@torch._dynamo.utils.disable_cache_limit()
@torch.compile(fullgraph=True, mode="reduce-overhead")
def spectral_clip_compiled_func(W: torch.Tensor, sigma_min: float=-1., sigma_max: float=1., ortho_dtype=None, num_ns_steps=len(NS_COEFFS), adaptive=False):
    if ortho_dtype is None:
        ortho_dtype = torch.float32
    return  _spectral_clip(W, sigma_min=sigma_min, sigma_max=sigma_max, ortho_dtype=ortho_dtype, num_ns_steps=num_ns_steps, adaptive=adaptive)

@torch.no_grad()
def filter_grad(grad, fft_alpha=1.0):
    # 1. Apply n-dimensional FFT
    grad_freq = torch.fft.fftn(grad, norm='ortho')
    
    # 2. Create a radial low-pass filter
    # Create a grid of frequency coordinates
    freq_dims = [torch.fft.fftfreq(s, device=grad.device) for s in grad.shape]
    # Center the grid for radial calculation
    shifted_freq_dims = [torch.fft.ifftshift(d) for d in freq_dims]
    
    # Create a meshgrid of coordinates
    coords = torch.stack(torch.meshgrid(*shifted_freq_dims, indexing='ij'))
    
    # Calculate the radial distance (L2 norm) from the center (zero frequency)
    # Normalize by the max possible frequency radius for scale invariance
    max_radius = 0.5 * math.sqrt(len(grad.shape))
    radius = torch.linalg.norm(coords, dim=0) / max_radius
    
    # Create a Gaussian low-pass filter.
    # Higher alpha means sharper decay, i.e., more aggressive filtering
    filter_weights = torch.exp(-fft_alpha * (radius ** 2))
    
    # 3. Apply the filter
    filtered_grad_freq = grad_freq * filter_weights
    
    # 4. Apply inverse n-dimensional FFT
    modified_grad = torch.fft.ifftn(filtered_grad_freq, norm='ortho')
    
    # The result should be real, but take .real to discard negligible imaginary parts
    return modified_grad.real

class FFTDescent(Optimizer):
    r"""
    FFTDescent: ***TEMPORARY NAME***

    Arguments:
        params (iterable):
            Iterable of parameters to optimize or dicts defining
            parameter groups.
        lr (float):
            Learning rate parameter (default 0.0001).
        beta (float, float, float):
            Coefficient used for computing the running average (default: 0.95)
        weight_decay (float):
            AdamW-like weight decay, i.e. a L2 penalty (default: 0.0).
        weight_decay_rate (float):
            Decay the multiplier at which rate weight decay is applied, weight_decay * weight_decay_rate**step (default: 0.995).
        spectral_clip (bool):
            Utilize six optimized Newton-Schulz iterations per step to clip the spectral norm to a max of 1. - https://leloykun.github.io/ponder/spectral-clipping/ - https://github.com/leloykun/spectral_clip (default: True, recommended to keep on True if possible / slowdown is negligible).
        spectral_clip_compile (bool):
            Compile the spectral clip function (Highly recommended for a large speed increase). (default: True).
        spectral_clip_dtype (torch.dtype or None):
            Compute spectral clipping in this dtype. (default: None, is determined based on spectral_clip_compile (float16 if uncompiled, float32 if compiled)).
        spectral_min (float):
            The minimum value of the spectral magnitude. Ought to be lower than spectral_max. (default: -1.0).
        spectral_max (float):
            The maximum value of the spectral magnitude. (default: 1.0).
        spectral_adaptive (bool):
            Adapt the result of spectral clipping to adapt to the scale of the gradients - https://github.com/leloykun/adaptive-muon (default: False).
        lowpass_grad (float):
            Pre-condition the gradient with a lowpass filter via FFT (default: 1.0).
        sign_momentum (float):
            Decouple the momentum from the sign/direction, value is the coefficient used for computing the sign's running average (default: 0.9).
        stochastic_fp (bool):
            Utilize stochastic rounding for bf16 and fp16 tensors. (default: True).
    """

    def __init__(
        self,
        params,
        lr: float = 1e-4,
        beta: float = 0.95,
        weight_decay: float = 0.0,
        weight_decay_rate: float = 0.995,
        spectral_clip: bool = True,
        spectral_clip_compile: bool = True,
        spectral_clip_dtype = None, # Can be set to torch.bfloat16, torch.float16, torch.float32, or even torch.float64 if you're insane in the membrane.
        spectral_min: float = -1.,
        spectral_max: float = 1.,
        spectral_adaptive: bool = False,
        lowpass_grad: float = 1.0,
        sign_momentum: float = 0.9,
        stochastic_fp: bool = True,
        **kwargs,
    ):
        
        # Loop over the keys in the kwargs dictionary
        for key in kwargs:
            logging.warning(
                f"Optimizer argument '{key}' passed into FFTDescent. It will be ignored."
            )


        self._init_lr = lr

        if spectral_clip:
            self.clip_func = spectral_clip_compiled_func if spectral_clip_compile else spectral_clip_func

        defaults = dict(
            lr = lr,
            beta = beta,
            weight_decay = weight_decay,
            weight_decay_rate = weight_decay_rate,
            spectral_clip = spectral_clip,
            spectral_clip_compile = spectral_clip_compile,
            spectral_clip_dtype = spectral_clip_dtype,
            spectral_min = spectral_min,
            spectral_max = spectral_max,
            spectral_adaptive = spectral_adaptive,
            lowpass_grad = lowpass_grad,
            sign_momentum = sign_momentum,
            stochastic_fp = stochastic_fp,
        )

        super(FFTDescent, self).__init__(params, defaults)

    @torch.no_grad()
    def reset(self):
        pass

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
            weight_decay = group["weight_decay"]
            weight_decay_rate = group["weight_decay_rate"]
            step = group['step']

            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]

                grad = p.grad.data

                dimcount = grad.ndim

                # State initialization
                if len(state) == 0:
                    # Exponential moving average of gradient values
                    state["momentum"] = torch.zeros_like(grad)
                    # Exponential moving average of sign
                    if group["sign_momentum"] != 0:
                        state["sign_momentum"] = torch.zeros_like(grad)

                # Detach
                p_fp32 = p.detach().clone()
                momentum = state["momentum"].detach().clone()
                if group["sign_momentum"] != 0:
                    sign_momentum = state["sign_momentum"].detach().clone()

                # Unpack
                if p.dtype in {torch.float16, torch.bfloat16} and group["stochastic_fp"]:
                    grad = grad.to(torch.float32)
                    momentum = state['momentum'].detach().clone().to(torch.float32)
                    if group["sign_momentum"] != 0:
                        sign_momentum = state['sign_momentum'].detach().clone().to(torch.float32)
                    p_fp32 = p.detach().clone().to(torch.float32)

                # Low-pass filter via FFT
                if dimcount > 0:
                    grad = filter_grad(grad, fft_alpha=group["lowpass_grad"]).abs().mul_(grad.sign())

                if step == 1:
                    grad.clamp_(-0, 0)

                # Decouple momentum from direction if using sign_momentum parameter (highly recommended)
                if group["sign_momentum"] != 0:
                    momentum = momentum.mul(beta).add_(grad.abs(), alpha=1. - beta)
                else:
                    momentum = momentum.mul(beta).add_(grad, alpha=1. - beta)

                # Update sign momentum
                if group["sign_momentum"] != 0:
                    sign_momentum = sign_momentum.mul(group["sign_momentum"]).add_(grad.sign(), alpha=1 - group["sign_momentum"])
                    c_t = grad.abs().lerp(momentum, weight=beta) # Nesterov-like momentum
                else:
                    c_t = grad.lerp(momentum, weight=beta) # Nesterov-like momentum

                # Spectral Clipping / Newton Schulz iters or RMS normalization
                if dimcount >= 2 and group["spectral_clip"]:
                    if dimcount > 2:
                        c_t_2d = c_t.reshape(len(c_t), -1) # Make 2D if conv or 1 dim
                    else:
                        c_t_2d = c_t

                    flip = c_t_2d.shape[0] > c_t_2d.shape[1]
                    if flip:
                        c_t_2d = c_t_2d.T # Flip if first dim is larger

                    full_step = self.clip_func(c_t_2d, sigma_min=group["spectral_min"], sigma_max=group["spectral_max"], adaptive=group["spectral_adaptive"], ortho_dtype=group["spectral_clip_dtype"])

                    if flip:
                        full_step = full_step.T

                    full_step = full_step.view_as(c_t).atan2(momentum.abs()).mul_(1.27323954474)
                else:
                    # Utilize momentum as denom with atan2
                    full_step = c_t.atan2(momentum.abs()).mul_(1.27323954474)

                # Apply sign if using sign_momentum
                if group["sign_momentum"] != 0:
                    full_step = full_step.mul(sign_momentum)

                # Perform weight decay
                if weight_decay != 0:
                    grad_weights = p_fp32.data

                    full_step = full_step.add(grad_weights, alpha=weight_decay * weight_decay_rate**group["step"])

                p_fp32.data.add_(full_step, alpha=-lr)

                if p.dtype in {torch.float16, torch.bfloat16} and group["stochastic_fp"]:
                    copy_stochastic_(state["momentum"], momentum)
                    if group["sign_momentum"] != 0:
                        copy_stochastic_(state["sign_momentum"], sign_momentum)
                    copy_stochastic_(p, p_fp32)
                else:
                    state["momentum"].copy_(momentum)
                    if group["sign_momentum"] != 0:
                        state["sign_momentum"].copy_(sign_momentum)
                    p.copy_(p_fp32)
        return loss