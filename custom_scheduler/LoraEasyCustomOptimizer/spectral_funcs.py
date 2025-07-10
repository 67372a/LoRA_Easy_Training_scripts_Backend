# from https://github.com/Clybius/Personalized-Optimizers by Clybius
from typing import Callable

import torch

NS_COEFFS = [
    (3.5318, -4.7911, 1.9388),
    (3.3274, -4.0557, 1.5782),
    (3.0809, -3.5160, 1.3464),
    (2.7476, -2.8484, 1.0775),
    (2.2948, -2.0951, 0.7895),
    (2.1535, -1.8338, 0.6869),
]

@torch.no_grad()
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

@torch.no_grad()
def block_matmul(
    P1: torch.Tensor, Q1: torch.Tensor, R1: torch.Tensor,
    P2: torch.Tensor, Q2: torch.Tensor, R2: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Performs block matrix multiplication elements of the (linear) sub-algebra
    of matrices of the form:
        [P   Q]
        [Q.T R]
    where Q is a MxN matrix, and P and R are symmetric matrices of size MxM and NxN respectively.
    """
    P = P1 @ P2   + Q1 @ Q2.T
    Q = P1 @ Q2   + Q1 @ R2
    R = Q1.T @ Q2 + R1 @ R2
    return P, Q, R

@torch.no_grad()
def newton_schulz_iter(
    P: torch.Tensor, Q: torch.Tensor, R: torch.Tensor,
    a: float, b: float, c: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """5th order blockwise Newton-Schulz iteration for orthogonalization."""
    P2, Q2, R2 = block_matmul(P, Q, R, P, Q, R)
    P4, Q4, R4 = block_matmul(P2, Q2, R2, P2, Q2, R2)
    I_P = a * torch.eye(P.shape[0], dtype=P.dtype, device=P.device)
    I_R = a * torch.eye(R.shape[0], dtype=R.dtype, device=R.device)
    Ppoly = I_P + b * P2 + c * P4
    Qpoly =       b * Q2 + c * Q4
    Rpoly = I_R + b * R2 + c * R4
    return block_matmul(P, Q, R, Ppoly, Qpoly, Rpoly)

@torch.no_grad()
def orthogonalize_blockwise(
    W: torch.Tensor, ortho_dtype=torch.float32, num_ns_steps: int=len(NS_COEFFS)
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Orthogonalize a matrix via 5th order blockwise Newton-Schulz iteration.

    Tighter spectral norm bound:
    => Matrices of the form [I_m, W; W.T, I_n] have spectral norm 1 + ||W||_2
    => We can estimate ||W||_2 via power iteration or Gram iteration.
    => However, we can also use the fact that ||W||_2 <= ||W||_F and the latter is much cheaper to compute.
    """
    orig_dtype = W.dtype
    m, n = W.shape
    I_m, I_n = torch.eye(m, device=W.device), torch.eye(n, device=W.device)
    # norm = 1 + _power_iterate(W, torch.manual_seed(0), num_iters=16)[1]
    norm = 1 + torch.linalg.norm(W)
    P = (I_m / (norm + 1e-12)).to(ortho_dtype)
    Q = (W   / (norm + 1e-12)).to(ortho_dtype)
    R = (I_n / (norm + 1e-12)).to(ortho_dtype)
    for a, b, c in NS_COEFFS[:num_ns_steps]:
        P, Q, R = newton_schulz_iter(P, Q, R, a=a, b=b, c=c)
    return P.to(orig_dtype), Q.to(orig_dtype), R.to(orig_dtype)

@torch.no_grad()
def _spectral_hardcap_blockwise(W: torch.Tensor, sigma_max=1., ortho_dtype=torch.float32, num_ns_steps=len(NS_COEFFS)):
    def _spectral_hardcap_blockwise_util(W: torch.Tensor):
        transpose = W.shape[0] > W.shape[1]
        if transpose:
            W = W.T
        orig_dtype = W.dtype
        W = W.to(ortho_dtype)
        # _, Q, R = orthogonalize_blockwise(W, ortho_dtype, num_ns_steps)
        # result = Q + W @ R
        P, Q, _ = orthogonalize_blockwise(W, ortho_dtype, num_ns_steps)
        result = Q + P @ W
        if transpose:
            result = result.T
        return result.to(orig_dtype)
    return sigma_max * _spectral_hardcap_blockwise_util(W / sigma_max)

@torch.no_grad()
def _spectral_hardcap_fully_materialized(W: torch.Tensor, sigma_max: float=1., ortho_dtype=torch.float32, num_ns_steps: int=len(NS_COEFFS)):
    def _spectral_clip_fully_materialized_utl(W: torch.Tensor):
        transpose = W.shape[0] > W.shape[1]
        if transpose:
            W = W.T
        orig_dtype = W.dtype
        W = W.to(ortho_dtype)
        m, n = W.shape
        I_m = torch.eye(m, dtype=W.dtype, device=W.device)
        I_n = torch.eye(n, dtype=W.dtype, device=W.device)
        row1 = torch.cat([I_m, W], dim=1)
        row2 = torch.cat([W.T, I_n], dim=1)
        H = torch.cat([row1, row2], dim=0)
        OH = orthogonalize(H, num_ns_steps)
        # Q, R = OH[:m, m:], OH[m:, m:]
        # W_clipped = Q + W @ R
        P, Q = OH[:m, :m], OH[:m, m:]
        result = Q + P @ W
        if transpose:
            result = result.T
        return result.to(orig_dtype)
    return sigma_max * _spectral_clip_fully_materialized_utl(W / sigma_max)

@torch.no_grad()
def _spectral_hardcap_nested(W: torch.Tensor, sigma_max: float=1., ortho_dtype=torch.float32, num_ns_steps: int=len(NS_COEFFS)):
    def _spectral_hardcap_util(W: torch.Tensor):
        transpose = W.shape[0] > W.shape[1]
        if transpose:
            W = W.T
        orig_dtype = W.dtype
        W = W.to(ortho_dtype)
        OW = orthogonalize(W, num_ns_steps)
        aW = OW - W
        result = ((OW + W) - aW @ orthogonalize(aW, num_ns_steps).T @ OW) / 2.
        if transpose:
            result = result.T
        return result.to(orig_dtype)
    return sigma_max * _spectral_hardcap_util(W / sigma_max)

@torch.no_grad()
def _spectral_clip(W: torch.Tensor, sigma_min: float=-1., sigma_max: float=1., ortho_dtype=torch.float32, num_ns_steps=len(NS_COEFFS)):
    flip = W.shape[0] > W.shape[1]
    if flip:
        W = W.T
    orig_dtype = W.dtype
    W = W.to(ortho_dtype)
    OW = orthogonalize(W, num_ns_steps)
    eye_m = torch.eye(W.shape[0], dtype=W.dtype, device=W.device)
    result = (1/2) * (
        (sigma_min + sigma_max) * eye_m
        + (sigma_min * OW - W) @ orthogonalize(sigma_min * OW - W, num_ns_steps).T
        - (sigma_max * OW - W) @ orthogonalize(sigma_max * OW - W, num_ns_steps).T
    ) @ OW
    if flip:
        result = result.T
    return result.to(orig_dtype)

@torch.no_grad()
def _spectral_relu(W: torch.Tensor, sigma_min: float=1., ortho_dtype=torch.float32, num_ns_steps=len(NS_COEFFS)):
    def _spectral_relu_util(W: torch.Tensor):
        flip = W.shape[0] > W.shape[1]
        if flip:
            W = W.T
        orig_dtype = W.dtype
        W = W.to(ortho_dtype)
        OW = orthogonalize(W, num_ns_steps)
        aW = OW - W
        result = (1/2) * (OW + W + aW @ orthogonalize(aW, num_ns_steps).T @ OW)
        if flip:
            result = result.T
        return result.to(orig_dtype)
    return sigma_min * _spectral_relu_util(W / sigma_min)

@torch.no_grad()
def batch_project(M: torch.Tensor, project_fn: Callable) -> torch.Tensor:
    """Batch project tensors of shape [..., fanout, fanin]. Taken from Modula library."""
    matrix_shape = M.shape[-2:]
    M_flattened = M.reshape(-1, *matrix_shape)
    
    # Replicate jax.vmap with a simple loop and stack.
    projected_list = [project_fn(m) for m in M_flattened]
    M_projected = torch.stack(projected_list, dim=0)
    
    # The division by batch size is unusual but preserved from the original JAX code.
    return M_projected.reshape(M.shape) / len(M_flattened)


@torch._dynamo.utils.disable_cache_limit()
@torch.compile(fullgraph=True, mode="reduce-overhead")
def spectral_hardcap(W: torch.Tensor, sigma_max=1., ortho_dtype=torch.float32, num_ns_steps=len(NS_COEFFS)):
    # return batch_project(W, lambda x: _spectral_hardcap_fully_materialized(x, sigma_max=sigma_max, ortho_dtype=ortho_dtype, num_ns_steps=num_ns_steps))
    # return batch_project(W, lambda x: _spectral_hardcap_nested(x, sigma_max=sigma_max, ortho_dtype=ortho_dtype, num_ns_steps=num_ns_steps))
    # return batch_project(W, lambda x: _spectral_clip(x, sigma_min=0., sigma_max=sigma_max))
    return batch_project(W, lambda x: _spectral_hardcap_blockwise(x, sigma_max=sigma_max, ortho_dtype=ortho_dtype, num_ns_steps=num_ns_steps))


@torch._dynamo.utils.disable_cache_limit()
@torch.compile(fullgraph=True, mode="reduce-overhead")
def spectral_clip(W: torch.Tensor, sigma_min: float=-1., sigma_max: float=1., ortho_dtype=torch.float32, num_ns_steps=len(NS_COEFFS)):
    return batch_project(W, lambda x: _spectral_clip(x, sigma_min=sigma_min, sigma_max=sigma_max, ortho_dtype=ortho_dtype, num_ns_steps=num_ns_steps))


@torch._dynamo.utils.disable_cache_limit()
@torch.compile(fullgraph=True, mode="reduce-overhead")
def spectral_relu(W: torch.Tensor, sigma_min: float=1., ortho_dtype=torch.float32, num_ns_steps=len(NS_COEFFS)):
    return batch_project(W, lambda x: _spectral_relu(x, sigma_min=sigma_min, ortho_dtype=ortho_dtype, num_ns_steps=num_ns_steps))