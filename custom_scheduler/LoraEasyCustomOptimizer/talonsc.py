# TALON from https://github.com/Clybius/Personalized-Optimizers by Clybius

import torch
from torch.optim import Optimizer
from typing import Tuple
from spectral_funcs import spectral_clip

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

NS_COEFFS = [
    (3.5318, -4.7911, 1.9388),
    (3.3274, -4.0557, 1.5782),
    (3.0809, -3.5160, 1.3464),
    (2.7476, -2.8484, 1.0775),
    (2.2948, -2.0951, 0.7895),
    (2.1535, -1.8338, 0.6869),
]

def block_matmul(
    P1: torch.Tensor, Q1: torch.Tensor, R1: torch.Tensor,
    P2: torch.Tensor, Q2: torch.Tensor, R2: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Performs block matrix multiplication elements of the (linear) sub-algebra
    of matrices of the form:
        [P   Q]
        [Q.T R]
    where Q is a MxN matrix, and P and R are symmetric matrices of size MxM and NxN respectively.
    This function is batched and operates on tensors of shape [B, ...].
    """
    P = P1 @ P2   + Q1 @ Q2.transpose(-2, -1)
    Q = P1 @ Q2   + Q1 @ R2
    R = Q1.transpose(-2, -1) @ Q2 + R1 @ R2
    return P, Q, R


def newton_schulz_iter(
    P: torch.Tensor, Q: torch.Tensor, R: torch.Tensor,
    a: float, b: float, c: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """5th order blockwise Newton-Schulz iteration for orthogonalization.
    This function is batched.
    """
    P2, Q2, R2 = block_matmul(P, Q, R, P, Q, R)
    P4, Q4, R4 = block_matmul(P2, Q2, R2, P2, Q2, R2)

    # Create 2D identity matrices that will broadcast with the batched tensors.
    I_P = a * torch.eye(P.shape[-2], dtype=P.dtype, device=P.device)
    I_R = a * torch.eye(R.shape[-2], dtype=R.dtype, device=R.device)

    Ppoly = I_P + b * P2 + c * P4
    Qpoly =       b * Q2 + c * Q4
    Rpoly = I_R + b * R2 + c * R4
    return block_matmul(P, Q, R, Ppoly, Qpoly, Rpoly)


def orthogonalize(M: torch.Tensor, num_ns_steps=len(NS_COEFFS)) -> torch.Tensor:
    """Orthogonalize a matrix via 5th order Newton-Schulz iteration."""
    transpose = M.shape[0] < M.shape[1]
    if transpose:
        M = M.T
    M = M / (torch.linalg.norm(M) + 1e-12)
    for a, b, c in NS_COEFFS[:num_ns_steps]:
        A = M.T @ M
        I = torch.eye(A.shape[0], dtype=M.dtype, device=M.device)
        M = M @ (a * I + b * A + c * A @ A)
    if transpose:
        M = M.T
    return M


def orthogonalize_blockwise(
    W: torch.Tensor, ortho_dtype: torch.dtype = torch.float16, num_ns_steps: int = len(NS_COEFFS)
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Orthogonalize a matrix via 5th order blockwise Newton-Schulz iteration.
    Assumes W is a batched tensor of shape (B, M, N).

    Tighter spectral norm bound:
    => Matrices of the form [I_m, W; W.T, I_n] have spectral norm 1 + ||W||_2
    => We can estimate ||W||_2 via power iteration or Gram iteration.
    => However, we can also use the fact that ||W||_2 <= ||W||_F and the latter is much cheaper to compute.
    """
    orig_dtype = W.dtype
    b, m, n = W.shape

    # Frobenius norm for each matrix in the batch. keepdim=True for broadcasting.
    norm = 1 + torch.linalg.norm(W, dim=(-2, -1), keepdim=True)

    # Create batched identity matrices
    I_m = torch.eye(m, dtype=ortho_dtype, device=W.device).expand(b, -1, -1)
    I_n = torch.eye(n, dtype=ortho_dtype, device=W.device).expand(b, -1, -1)

    # Scale and cast
    P = (I_m / (norm + 1e-12)).to(ortho_dtype)
    Q = (W / (norm + 1e-12)).to(ortho_dtype)
    R = (I_n / (norm + 1e-12)).to(ortho_dtype)

    for a, b, c in NS_COEFFS[:num_ns_steps]:
        P, Q, R = newton_schulz_iter(P, Q, R, a=a, b=b, c=c)

    return P.to(orig_dtype), Q.to(orig_dtype), R.to(orig_dtype)


def _spectral_hardcap_blockwise(
    W: torch.Tensor, sigma_max: float = 1.0, ortho_dtype: torch.dtype = torch.float16, num_ns_steps: int = len(NS_COEFFS)
) -> torch.Tensor:
    """Internal helper to apply spectral hardcap to a batch of 2D matrices."""
    W_scaled = W / sigma_max

    transpose = W_scaled.shape[-2] > W_scaled.shape[-1]
    if transpose:
        W_scaled = W_scaled.transpose(-2, -1)

    orig_dtype = W.dtype
    W_ortho = W_scaled.to(ortho_dtype)

    P, Q, _ = orthogonalize_blockwise(W_ortho, ortho_dtype, num_ns_steps)
    
    # The matrix multiplication uses the ortho_dtype version of W, matching original logic.
    result = Q + P @ W_ortho

    if transpose:
        result = result.transpose(-2, -1)

    return sigma_max * result.to(orig_dtype)

@torch.no_grad()
def _spectral_hardcap_fully_materialized(
    W: torch.Tensor,
    sigma_max: float = 1.0,
    ortho_dtype: torch.dtype = torch.float16,
    num_ns_steps: int = len(NS_COEFFS)
) -> torch.Tensor:
    """Helper function to clip the singular values of a single matrix."""
    
    # The original JAX function scales the input before clipping and the output after.
    W_scaled = W / sigma_max
    
    if (transpose := W_scaled.shape[0] > W_scaled.shape[1]):
        W_scaled = W_scaled.T
        
    orig_dtype = W.dtype
    W_casted = W_scaled.to(ortho_dtype)
    
    m, n = W_casted.shape
    I_m = torch.eye(m, dtype=W_casted.dtype, device=W_casted.device)
    I_n = torch.eye(n, dtype=W_casted.dtype, device=W_casted.device)
    
    # Replicate jnp.block using torch.cat
    top_row = torch.cat([I_m, W_casted], dim=1)
    bottom_row = torch.cat([W_casted.T, I_n], dim=1)
    H = torch.cat([top_row, bottom_row], dim=0)
    
    OH = orthogonalize(H, num_ns_steps)
    
    # Extract blocks P and Q from the orthogonalized H
    P, Q = OH[:m, :m], OH[:m, m:]
    result = Q + P @ W_casted
    
    if transpose:
        result = result.T
        
    # Scale result back up and cast to original dtype
    return (sigma_max * result).to(orig_dtype)


def spectral_hardcap_blockwise(
    W: torch.Tensor, sigma_max: float = 1.0, ortho_dtype: torch.dtype = torch.float16, num_ns_steps: int = len(NS_COEFFS)
) -> torch.Tensor:
    """
    Applies a spectral norm hard cap to a tensor of weights `W`.

    This function reshapes an input tensor `W` of shape [..., fan_out, fan_in]
    into a batch of 2D matrices, applies the spectral hard cap to each matrix,
    and then reshapes it back to the original shape.

    The JAX equivalent uses `vmap` over a function for a single matrix. This
    PyTorch version is implemented to be batch-native for efficiency.
    """
    orig_shape = W.shape
    matrix_shape = W.shape[-2:]

    W_flat = W.reshape(-1, *matrix_shape)

    W_projected_flat = _spectral_hardcap_blockwise(
        W_flat,
        sigma_max=sigma_max,
        ortho_dtype=ortho_dtype,
        num_ns_steps=num_ns_steps
    )

    batch_size = W_flat.shape[0]
    if batch_size == 0:
        return W_projected_flat.reshape(orig_shape)

    W_projected = W_projected_flat.reshape(orig_shape)
    
    return W_projected / batch_size

# https://github.com/kozistr/pytorch_optimizer/blob/6397d56279ad80b26c4bba7fb4b04852b517fdeb/pytorch_optimizer/optimizer/shampoo_utils.py#L533
def zero_power_via_newton_schulz_6(
    g: torch.Tensor, eps: float = 1e-16
) -> torch.Tensor:
    r"""Compute the zeroth power / orthogonalization of G.

    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a quintic iteration
    whose coefficients are selected to maximize the slope at zero. For the purpose of minimizing steps, it turns out
    to be empirically effective to keep increasing the slope at zero even beyond the point where the iteration no
    longer converges all the way to one everywhere on the interval. This iteration therefore does not produce UV^T but
    rather something like US'V^T where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt
    model performance at all relative to UV^T, where USV^T = G is the SVD.

    :param g: torch.Tensor. matrix.
    :param num_steps: int. number of iterations.
    :param eps: float. add this times I to G, to make is positive definite. For scaling, we multiply it by the largest
        eigenvalue of G.
    :param weights: Tuple[int, int, int]. weights.
    """
    if len(g.shape) != 2:
        raise ValueError('shape of g must be 2-dimensional')

    abc_list = [
      (3955/1024, -8306/1024, 5008/1024),
      (3735/1024, -6681/1024, 3463/1024),
      (3799/1024, -6499/1024, 3211/1024),
      (4019/1024, -6385/1024, 2906/1024),
      (2677/1024, -3029/1024, 1162/1024),
      (2172/1024, -1833/1024,  682/1024)
   ]

    x = g.float()
    x = x.div(x.norm().add_(eps))

    if g.size(0) > g.size(1):
        x = x.T

    for weight in abc_list:
        a = x @ x.T
        b = weight[1] * a + weight[2] * a @ a
        x = weight[0] * x + b @ x

    if g.size(0) > g.size(1):
        x = x.T

    x = torch.einsum('ij,ij,ab->ab', g.type_as(x), x, x)

    return x

class TALONSC(Optimizer):
    r"""
    TALON: Temporal Adaptation via Level and Orientation Normalization. 
    
    Cuts through noise by decoupling the gradient's sign and magnitude into two different momentum states, with a denominator for adaptive learning.

    Arguments:
        params (iterable):
            Iterable of parameters to optimize or dicts defining
            parameter groups.
        lr (float):
            Learning rate parameter (default 0.0001).
        betas (float, float, float):
            Coefficient used for computing the sign momentum, running average, and the long-term squared running average (default: 0.9, 0.99, 0.9999999)
        weight_decay (float):
            AdamW-like weight decay, i.e. a L2 penalty (default: 0.0).
        weight_decay_rate (float):
            Decay the multiplier at which rate weight decay is applied, weight_decay * weight_decay_rate**step (default: 0.995).
        denom_atan2 (bool):
            Divide the smooth gradient using .atan2 instead of .div for stability and scale-invariance, removes epsilon/eps - https://arxiv.org/abs/2407.05872 (default: True).
        invariant (bool):
            Scale the latent into -1 to 1 space via .arctan().sin(), then later divide by the original grad's .arctan().cos(). Its been tested a bit, with the general result of speeding up descent. (default: False).
        adaptive_muon (bool):
            Utilize six optimized Newton-Schulz iterations per step to compute the orthogonalization of the gradient, and adapt to the gradient norm - https://arxiv.org/abs/2410.21265 - https://github.com/leloykun/adaptive-muon (default: True).
        orthograd (bool):
            Modify the gradient to apply an orthogonal gradient update, - https://arxiv.org/abs/2501.04697 - extended with atan2 in place of epsilon - https://arxiv.org/abs/2407.05872 (default: False).
        stochastic_fp (bool):
            Utilize stochastic rounding for bf16 and fp16 tensors. (default: True).
    """

    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: tuple = (0.9, 0.99, 1. - 1e-7),
        weight_decay: float = 0.0,
        weight_decay_rate: float = 0.995,
        denom_atan2: bool = True,
        invariant: bool = False,
        adaptive_muon: bool = True,
        orthograd: bool = False,
        stochastic_fp: bool = True,
    ):

        self._init_lr = lr

        defaults = dict(
            lr = lr,
            betas = betas,
            weight_decay = weight_decay,
            weight_decay_rate = weight_decay_rate,
            denom_atan2 = denom_atan2,
            invariant = invariant,
            adaptive_muon = adaptive_muon,
            orthograd = orthograd,
            stochastic_fp = stochastic_fp,
        )

        super(TALONSC, self).__init__(params, defaults)

    @torch.no_grad()
    def orthograd_atan2sin(self, p, grad):
        w = p.view(-1)
        g = grad.view(-1)

        dot_product = torch.dot(w, g).atan2_(torch.dot(w, w))
        sin_dot_product = torch.sin(dot_product)

        g_atansin = g.to(dtype=torch.float32, copy=True).atan().sin_()
        g_atancos = g.to(dtype=torch.float32, copy=True).atan().cos_()

        g_orth = g_atansin.sub(w.atan().sin_(), alpha=sin_dot_product).div(g_atancos)

        g_orth_scaled = g_orth.mul(g.norm(2).div_(g_orth.norm(2).clamp_min_(1e-16)))

        grad.copy_(g_orth_scaled.view_as(grad))
    
    @torch.no_grad()
    def invariance(self, grad, degrad = None):
        if degrad is None:
            g_atansin = grad.atan().sin_()
            g_atancos = grad.atan().cos_()

            return g_atansin, g_atancos
        else:
            return grad.atan2(degrad).mul_(1.27323954474)

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
            betas = group["betas"]
            weight_decay = group["weight_decay"]
            weight_decay_rate = group["weight_decay_rate"]
            step = group['step']

            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]

                grad = p.grad.data

                # State initialization
                if len(state) == 0:
                    # Exponential moving average of gradient values
                    state["value_momentum"] = torch.ones_like(p.data)
                    # Exponential moving average of squared gradient values
                    state["stage2_emasq"] = torch.ones_like(p.data)
                    state["sign_momentum"] = torch.zeros_like(grad)

                # Detach
                p_fp32 = p.detach().clone()
                value_momentum = state["value_momentum"].detach().clone()
                stage2_emasq = state["stage2_emasq"].detach().clone()
                sign_momentum = state["sign_momentum"].detach().clone()

                # Unpack
                if p.dtype in {torch.float16, torch.bfloat16} and group["stochastic_fp"]:
                    grad = grad.to(torch.float32)
                    value_momentum = state['value_momentum'].detach().clone().to(torch.float32)
                    stage2_emasq = state['stage2_emasq'].detach().clone().to(torch.float32)
                    sign_momentum = state['sign_momentum'].detach().clone().to(torch.float32)
                    p_fp32 = p.detach().clone().to(torch.float32)

                # Create betas
                slow_beta1 = ((betas[1]**(step) - betas[1]) / (betas[1]**(step) - 1.0)) # Short-term bias-correctionless squared EMA beta
                slow_beta2 = ((betas[2]**(step) - betas[2]) / (betas[2]**(step) - 1.0)) # Long-term bias-correctionless squared EMA beta

                # Absmax clip value for early stability
                clip_lambda = step**0.25

                rms = grad.pow(2).mean().sqrt_().clamp_min_(1)
                grad = grad.div(rms)

                # Orthograd
                if group["orthograd"] and p_fp32.data.nelement() > 1: # Might just be me, but I've had the most success via ndim > 1
                    self.orthograd_atan2sin(p_fp32, grad)

                # Update sign momentum
                sign_momentum = sign_momentum.lerp(grad.sign(), weight=1. - betas[0])

                # Clip grad to prevent INF
                grad = torch.where(
                    grad.abs() > 255,
                    grad.mul(255 / grad.abs()),
                    grad
                )

                # ADOPT-style update squared momentum (Stage 1)
                value_momentum = value_momentum.mul(slow_beta1).add_(grad.abs(), alpha=1 - slow_beta1)

                # Adaptive Muon / Newton Schulz iters
                dimcount = value_momentum.ndim
                if dimcount > 0 and group["adaptive_muon"]:
                    if dimcount > 1:
                        c_t = spectral_clip(value_momentum.mul(sign_momentum))
                    else:
                        c_t = spectral_clip(value_momentum.mul(sign_momentum).view(len(value_momentum), -1)).view(value_momentum.shape)
                else:
                    c_t = value_momentum.mul(sign_momentum)
                
                if group["invariant"] and c_t.nelement() > 0:
                    c_t, degrad = self.invariance(c_t)

                # Denom (Stage 2)
                if group["denom_atan2"]:
                    full_step = c_t.atan2(stage2_emasq.sqrt()).mul_(1.27323954474)
                else:
                    stage2_denom = torch.clamp(stage2_emasq.sqrt(), 1e-16)
                    full_step = c_t.div(stage2_denom).clamp_(-clip_lambda, clip_lambda)

                # ADOPT-style update squared momentum (Stage 2)
                stage2_emasq = stage2_emasq.mul(slow_beta2).addcmul_(grad, grad, value=1 - slow_beta2)

                if group["invariant"] and grad.nelement() > 0:
                    full_step = self.invariance(full_step, degrad)

                # Perform weight decay
                if weight_decay != 0:
                    grad_weights = p_fp32.data

                    full_step = full_step.add(grad_weights, alpha=weight_decay * weight_decay_rate**group["step"])

                p_fp32.data.add_(full_step, alpha=-lr)
                if p.dtype in {torch.float16, torch.bfloat16} and group["stochastic_fp"]:
                    copy_stochastic_(state["value_momentum"], value_momentum)
                    copy_stochastic_(state["stage2_emasq"], stage2_emasq)
                    copy_stochastic_(state["sign_momentum"], sign_momentum)
                    copy_stochastic_(p, p_fp32)
                else:
                    state["value_momentum"].copy_(value_momentum)
                    state["stage2_emasq"].copy_(stage2_emasq)
                    state["sign_momentum"].copy_(sign_momentum)
                    p.copy_(p_fp32)
        return loss