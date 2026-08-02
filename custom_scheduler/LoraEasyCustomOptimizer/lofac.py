# Source: https://github.com/Clybius/Personalized-Optimizers

"""LoFac: Low-rank Factorized, momentum-free optimizer.

Designed as a highly memory-efficient, momentum-free alternative to
Adafactor-style optimizers. Instead of Adafactor's row/col (rank-1 outer
product) second-moment fit, the second moment is tracked *inside a low-rank
factorization of the gradient* -- LoRA, LoHa, or LoKr -- updated by one
EMA-weighted alternating least-squares (ALS) sweep per optimizer step. The
full-size second moment is never stored as state; it is only materialized
transiently during the step.

Per step (2D+ parameters, shape m x n):

1. Factorized second-moment update with a single poly-beta debiased EMA
   (default ``polybeta=0.99999``): fit the low-rank factors to ``T = g^2``
   with one alternating ALS half-iteration, smoothed with EMA weight
   ``w = 1 - polybeta`` (``w = 1`` at step 1, so the estimate is exact at
   the first step -- no compounding bias-correction terms):

   - **LoRA** (default): two rank-r factors B (m x r), A (r x n) with
     ``v = B @ A``. A and B are alternately re-fit via r x r solves:
     ``A <- (B^T B + eps I)^{-1} B^T T``, ``B <- T A^T (A A^T + eps I)^{-1}``.
   - **LoHa**: two LoRA pairs with ``v = (B1 A1) o (B2 A2)`` (Hadamard
     product), giving an effective rank of r^2. Pair 1 is fit to
     ``T / (B2 A2 + eps)``, then pair 2 to ``T / (B1 A1 + eps)``.
   - **LoKr**: Kronecker factorization ``v = kron(A, B)`` with A (a x c),
     B (b x d) and m = a*b, n = c*d (dims auto-split closest to sqrt, or
     user-specified). The ALS update is a closed-form weighted mean:
     ``A <- sum_{j,l} T4[i,j,k,l] B[j,l] / ||B||^2`` (T viewed as a x b x c x d).

2. Precondition: ``u = g / (sqrt(v) + eps)``. No momentum buffer exists --
   momentum's stabilizing role is filled by the heavy conditioning, per the
   SinkFactor rationale (arXiv:2507.07101).

3. Optional whole-tensor RMS normalization of ``u`` to 1.0.

4. Optional post-processing: 2-step Gram-Newton-Schulz orthogonalization of
   ``u`` (pre-optimized cubic coefficients, bf16 compute).

5. Optional rescaling of the orthogonalized update back to the RMS norm of
   the *non-orthogonalized* update (Muon-family norm-preserving trick).

6. Optional RMS-clip (shrink-only), cautious update, cautious decoupled
   weight decay, and stochastic rounding for BF16 parameters.

Memory: state per m x n matrix is r(m+n) (LoRA), 2r(m+n) (LoHa), or
ac + bd (LoKr) fp32 scalars -- far less than one gradient. E.g. a 4096 x 4096
weight with rank 8 uses 65K elements of state (0.4%) versus 16.7M for Adam.
If the factorization is not smaller than the parameter (small tensors, low
effective rank), the optimizer transparently falls back to a plain
elementwise second moment (state == gradient size, like Adafactor's
non-factorized path). 1D/0D tensors always use the plain path.
"""

import math
from typing import Optional, Tuple

import torch
from torch.optim import Optimizer


def copy_stochastic_(target: torch.Tensor, source: torch.Tensor) -> None:
    """Stochastically round a float32 source into bfloat16.

    This prevents dying parameters in bf16 training: when a bf16 update is smaller
    than one ULP, deterministic rounding would truncate it to zero, whereas stochastic
    rounding gives it a non-zero probability of surviving.
    """
    assert source.dtype is torch.float32, f"source must be float32, got {source.dtype}"
    assert target.dtype is torch.bfloat16, f"target must be bfloat16, got {target.dtype}"
    with torch.no_grad():
        result = torch.randint_like(source, dtype=torch.int32, low=0, high=(1 << 16))
        result.add_(source.view(dtype=torch.int32))
        result.bitwise_and_(-65536)
        target.copy_(result.view(dtype=torch.float32).to(target.dtype))


def _reshape_to_2d(t: torch.Tensor) -> torch.Tensor:
    """Reshape tensor to 2D: [N, -1] for >2D, [1, -1] for 1D, identity for 2D."""
    if t.ndim > 2:
        return t.reshape(t.shape[0], -1)
    if t.ndim < 2:
        return t.reshape(1, -1)
    return t


GRAM_NEWTON_SCHULZ_2STEP_COEFFS = [
    (1.4897216394163149, -0.5798724169434551, 0.0831346315615072),
    (2.0181598271548000, -1.5523232773433393, 0.5343894201774000),
]


@torch.no_grad()
def gram_newton_schulz_2step(
    M: torch.Tensor,
    eps: float = 1e-7,
    ortho_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """2-step Gram Newton-Schulz with pre-optimized unconstrained coefficients.

    Uses a 2-step accumulated iteration with coefficients derived from pure
    optimization (no h(1)=1 or h'(1)=-0.5 constraints enforced), optimized
    for spectral range [0.2, 1.8] with convergence target
    ||r_final - 1|| < 1e-4. Includes the AOL-Gram folding for a cheap
    spectral pre-conditioning.

    Args:
        M: Input matrix [n, m] to orthogonalize
        eps: Numerical stability constant
        ortho_dtype: Data type for orthogonalization computation

    Returns:
        Orthonormal matrix [n, m]
    """
    X = M.to(ortho_dtype)
    transposed = False
    if X.size(0) > X.size(1):
        X = X.mT
        transposed = True

    # AOL-Gram folding
    A = X @ X.mT
    rescaling = A.abs().sum(dim=-1).clamp_min_(eps)
    s = rescaling.rsqrt().unsqueeze(-1)
    X = X * s
    R = s * A * s.mT

    n, m = X.shape
    I = torch.eye(n, dtype=X.dtype, device=X.device)
    # Q = I is safe here: Q = Q @ z rebinds Q to a new tensor (not in-place),
    # so I is never mutated.
    Q = I

    for a, b, c in GRAM_NEWTON_SCHULZ_2STEP_COEFFS:
        # Cubic polynomial on Gram matrix
        R2 = R @ R
        z = a * I + b * R + c * R2

        # Accumulated updates
        Q = Q @ z
        R = z @ R @ z

    out = Q @ X

    if transposed:
        out = out.mT

    return out.to(M.dtype)


def _factor_dims(size: int) -> Tuple[int, int]:
    """Split ``size`` into (a, b) with a*b == size and a closest to sqrt(size)."""
    best = 1
    for i in range(1, int(size ** 0.5) + 1):
        if size % i == 0:
            best = i
    return best, size // best


def _poly_beta(polybeta: float, step: int) -> float:
    """Poly-beta debiased EMA weight: 0 at step <= 1, asymptotes to polybeta."""
    if step <= 1:
        return 0.0
    beta_pow = polybeta ** step
    return (beta_pow - polybeta) / (beta_pow - 1.0)


class LoFac(Optimizer):
    r"""LoFac: Low-rank Factored, momentum-free optimizer.

    The second moment of the gradient is tracked inside a low-rank
    factorization (LoRA / LoHa / LoKr), updated by one EMA-weighted ALS
    sweep per step with a single poly-beta (default 0.99999). The update is
    ``u = g / (sqrt(v) + eps)`` with optional RMS normalization, optional
    2-step Gram-Newton-Schulz orthogonalization, and optional rescaling of
    the orthogonalized update to the norm of the non-orthogonalized update.

    Args:
        params: iterable of parameters to optimize.
        lr (float): learning rate (default: 1e-2). Larger than a typical
            Adam-style default since the normalization pipeline keeps the
            pre-lr update RMS near O(1); treat this as a tuning starting point.
        polybeta (float): single poly-beta decay rate governing the EMA
            smoothing of the low-rank factor fits (default: 0.99999).
        rank (int): rank of the low-rank factors for LoRA/LoHa modes,
            capped at min(m, n) per tensor (default: 8).
        factorization (str): "lora" (default), "loha", or "lokr". Falls back
            to LoRA (then plain) if the requested factorization's state is not
            smaller than the parameter.
        lokr_dims (tuple, optional): explicit LoKr factor dims (a, c) with
            m = a*b and n = c*d; must divide the parameter shape. If None,
            dims are auto-split closest to sqrt.
        weight_decay (float): decoupled weight decay coefficient (default: 0.0).
        orthogonalize (bool): apply 2-step Gram-Newton-Schulz to the 2D+
            update after preconditioning (default: False -- expensive,
            opt-in post-processing).
        rescale_ortho (bool): rescale the orthogonalized update back to the
            RMS norm of the non-orthogonalized update (only meaningful when
            ``orthogonalize`` is on; default: True).
        rms_normalize (bool): whole-tensor RMS-normalize the preconditioned
            update to 1.0 (default: True).
        rms_clip (bool): shrink-only RMS clip on the final update, never
            boosts (default: True).
        cautious_update (bool): mask update components whose sign disagrees
            with the raw gradient (default: True).
        cautious_wd (bool): mask weight decay components whose sign disagrees
            with the parameter (default: True).
        stochastic_fp (bool): use stochastic rounding when writing back to
            bf16 parameters (default: True).
        compile_step (bool): torch.compile the per-mode core step functions
            (default: False).
        ortho_dtype (torch.dtype): dtype for Gram-Newton-Schulz computation
            (default: torch.bfloat16).
        eps (float): numerical stability epsilon (default: 1e-8).
    """

    def __init__(
        self,
        params,
        lr: float = 1e-2,
        polybeta: float = 0.99999,
        rank: int = 8,
        factorization: str = "lokr",
        lokr_dims: Optional[Tuple[int, int]] = None,
        weight_decay: float = 0.0,
        orthogonalize: bool = True,
        rescale_ortho: bool = True,
        rms_normalize: bool = False,
        rms_clip: bool = False,
        cautious_update: bool = True,
        cautious_wd: bool = True,
        stochastic_fp: bool = True,
        compile_step: bool = False,
        ortho_dtype: torch.dtype = torch.bfloat16,
        eps: float = 1e-8,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= polybeta < 1.0:
            raise ValueError(f"Invalid polybeta parameter: {polybeta}")
        if rank < 1:
            raise ValueError(f"Invalid rank: {rank}")
        if factorization not in ("lora", "loha", "lokr"):
            raise ValueError(
                f"Invalid factorization: {factorization}. "
                "Must be one of: 'lora', 'loha', 'lokr'"
            )
        if lokr_dims is not None:
            if len(lokr_dims) != 2 or lokr_dims[0] < 1 or lokr_dims[1] < 1:
                raise ValueError(f"Invalid lokr_dims: {lokr_dims}. Must be (a, c) with a, c >= 1")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if eps < 0.0:
            raise ValueError(f"Invalid eps value: {eps}")

        defaults = dict(
            lr=lr,
            polybeta=polybeta,
            rank=rank,
            factorization=factorization,
            lokr_dims=lokr_dims,
            weight_decay=weight_decay,
            orthogonalize=orthogonalize,
            rescale_ortho=rescale_ortho,
            rms_normalize=rms_normalize,
            rms_clip=rms_clip,
            cautious_update=cautious_update,
            cautious_wd=cautious_wd,
            stochastic_fp=stochastic_fp,
            ortho_dtype=ortho_dtype,
            eps=eps,
        )
        super().__init__(params, defaults)

        self._compile_step = compile_step

        if self._compile_step:
            try:
                torch._dynamo.config.recompile_limit = max(
                    torch._dynamo.config.recompile_limit, 64
                )
                self._compiled_lora = torch.compile(
                    self._core_lora, fullgraph=True, dynamic=False,
                )
                self._compiled_loha = torch.compile(
                    self._core_loha, fullgraph=True, dynamic=False,
                )
                self._compiled_lokr = torch.compile(
                    self._core_lokr, fullgraph=True, dynamic=False,
                )
                self._compiled_plain = torch.compile(
                    self._core_plain, fullgraph=True, dynamic=False,
                )
            except Exception as e:
                import logging
                logging.warning(
                    f"torch.compile failed to initialize: {e}. Falling back to uncompiled step."
                )
                self._compiled_lora = self._core_lora
                self._compiled_loha = self._core_loha
                self._compiled_lokr = self._core_lokr
                self._compiled_plain = self._core_plain
        else:
            self._compiled_lora = self._core_lora
            self._compiled_loha = self._core_loha
            self._compiled_lokr = self._core_lokr
            self._compiled_plain = self._core_plain

    @staticmethod
    def _resolve_mode(
        m: int,
        n: int,
        factorization: str,
        rank: int,
        lokr_dims: Optional[Tuple[int, int]],
    ) -> Tuple[str, Optional[Tuple[Tuple[int, int], Tuple[int, int]]]]:
        """Pick the cheapest mode whose state is smaller than the parameter.

        Returns (mode, lokr_dims_or_None). Falls back LoKr -> LoRA -> plain
        and LoHa/LoRA -> plain when the factored state would not be smaller
        than m*n (small tensors, low effective rank).
        """
        if m < 2 or n < 2:
            return "plain", None

        if factorization == "lokr":
            if lokr_dims is not None:
                a, c = lokr_dims
                if m % a != 0 or n % c != 0:
                    raise ValueError(
                        f"lokr_dims (a={a}, c={c}) must divide shape ({m}, {n})"
                    )
                b, d = m // a, n // c
            else:
                a, b = _factor_dims(m)
                c, d = _factor_dims(n)
            if a > 1 and c > 1 and a * c + b * d < m * n:
                return "lokr", ((a, c), (b, d))
            # Fall through to LoRA sizing
            r = min(rank, m, n)
            if r * (m + n) < m * n:
                return "lora", None
            return "plain", None

        if factorization == "loha":
            r = min(rank, m, n)
            if 2 * r * (m + n) < m * n:
                return "loha", None
            if r * (m + n) < m * n:
                return "lora", None
            return "plain", None

        # factorization == "lora"
        r = min(rank, m, n)
        if r * (m + n) < m * n:
            return "lora", None
        return "plain", None

    def _init_state(self, state: dict, p: torch.Tensor) -> None:
        """Create per-parameter state: factored buffers or plain second moment."""
        state["step"] = 0
        state["w_t"] = torch.zeros((), dtype=torch.float32, device=p.device)
        grad = p.grad
        if grad.ndim < 2:
            state["mode"] = "plain"
            # Stored 2D (1, n) so the compiled plain core has a single code
            # path with no rank-dependent branching.
            state["exp_avg_sq"] = torch.zeros(
                (1, grad.numel()), dtype=torch.float32, device=p.device
            )
            return

        g2d = _reshape_to_2d(grad)
        m, n = g2d.shape
        mode, lokr_dims = self._resolve_mode(
            m, n, self.defaults["factorization"], self.defaults["rank"], self.defaults["lokr_dims"]
        )
        state["mode"] = mode

        if mode == "plain":
            state["exp_avg_sq"] = torch.zeros(
                (m, n), dtype=torch.float32, device=p.device
            )
        elif mode == "lora":
            r = min(self.defaults["rank"], m, n)
            state["B"] = torch.randn(m, r, dtype=torch.float32, device=p.device) / math.sqrt(m)
            state["A"] = torch.randn(r, n, dtype=torch.float32, device=p.device) / math.sqrt(n)
        elif mode == "loha":
            r = min(self.defaults["rank"], m, n)
            for key in ("B1", "A1", "B2", "A2"):
                rows, cols = (m, r) if key.startswith("B") else (r, n)
                state[key] = torch.randn(rows, cols, dtype=torch.float32, device=p.device) / math.sqrt(rows)
        else:  # "lokr"
            (a, c), (b, d) = lokr_dims
            state["A"] = torch.randn(a, c, dtype=torch.float32, device=p.device) / math.sqrt(c)
            state["B"] = torch.randn(b, d, dtype=torch.float32, device=p.device) / math.sqrt(d)
            state["a"], state["b"], state["c"], state["d"] = a, b, c, d

    @staticmethod
    def _finish(
        u: torch.Tensor,
        g_raw: torch.Tensor,
        p_raw: torch.Tensor,
        weight_decay: float,
        eps: float,
        rms_normalize: bool,
        orthogonalize: bool,
        rescale_ortho: bool,
        ortho_dtype: torch.dtype,
        rms_clip: bool,
        cautious_update: bool,
        cautious_wd: bool,
    ) -> torch.Tensor:
        """Shared tail: RMS normalize, optional GNS ortho + norm-rescale, RMS
        clip, cautious update, cautious weight decay."""
        if rms_normalize:
            u = u / (u.pow(2).mean().sqrt() + eps)

        if orthogonalize and u.ndim >= 2 and u.shape[0] >= 2 and u.shape[1] >= 2:
            pre_rms = u.pow(2).mean().sqrt()
            u = gram_newton_schulz_2step(u, eps=eps, ortho_dtype=ortho_dtype)
            if rescale_ortho:
                # Rescale the orthogonalized update back to the RMS norm of
                # the non-orthogonalized update (norm-preserving).
                u = u * (pre_rms / (u.pow(2).mean().sqrt() + eps))

        if rms_clip:
            # Shrink-only: never boosts.
            u = u / u.pow(2).mean().sqrt().clamp_min(1.0)

        update = u

        if cautious_update:
            # `>= 0` (not `> 0`) so that coordinates with an exactly-zero
            # gradient (u == 0) count as agreeing: they carry no signal and
            # must not shrink mask_mean, which would amplify every surviving
            # element by 1/mask_mean and destabilize convergence.
            mask = (update * g_raw >= 0).to(update.dtype)
            mask_mean = mask.mean().clamp_min(1e-3)
            update = update * mask / mask_mean

        if weight_decay != 0:
            if cautious_wd:
                wd_mask = (update.sign() == p_raw.sign()).to(update.dtype)
                update = update + weight_decay * p_raw * wd_mask
            else:
                update = update + weight_decay * p_raw

        return update

    @staticmethod
    def _core_lora(
        g2d: torch.Tensor,
        B: torch.Tensor,
        A: torch.Tensor,
        p2d: torch.Tensor,
        w: torch.Tensor,
        weight_decay: float,
        eps: float,
        rms_normalize: bool,
        orthogonalize: bool,
        rescale_ortho: bool,
        ortho_dtype: torch.dtype,
        rms_clip: bool,
        cautious_update: bool,
        cautious_wd: bool,
    ) -> torch.Tensor:
        """LoRA-mode core: rank-r fit of g^2 via EMA-weighted ALS, then
        precondition ``u = g / sqrt(v)``."""
        r = A.shape[0]
        eye = torch.eye(r, dtype=B.dtype, device=B.device)
        T = g2d.pow(2)

        # ALS half-iterations, smoothed with EMA weight w:
        # A <- (B^T B + eps I)^{-1} B^T T   (least squares in A given B)
        # B <- T A^T (A A^T + eps I)^{-1}   (least squares in B given A)
        A.lerp_(torch.linalg.solve(B.mT @ B + eps * eye, B.mT @ T), w)
        B.lerp_(
            torch.linalg.solve((A @ A.mT + eps * eye).mT, A @ T.mT).mT, w
        )

        u = g2d / ((B @ A).clamp_min(eps).sqrt() + eps)
        return LoFac._finish(
            u, g2d, p2d, weight_decay, eps, rms_normalize, orthogonalize,
            rescale_ortho, ortho_dtype, rms_clip, cautious_update, cautious_wd,
        )

    @staticmethod
    def _core_loha(
        g2d: torch.Tensor,
        B1: torch.Tensor,
        A1: torch.Tensor,
        B2: torch.Tensor,
        A2: torch.Tensor,
        p2d: torch.Tensor,
        w: torch.Tensor,
        weight_decay: float,
        eps: float,
        rms_normalize: bool,
        orthogonalize: bool,
        rescale_ortho: bool,
        ortho_dtype: torch.dtype,
        rms_clip: bool,
        cautious_update: bool,
        cautious_wd: bool,
    ) -> torch.Tensor:
        """LoHa-mode core: two rank-r pairs, v = (B1 A1) o (B2 A2)."""
        r = A1.shape[0]
        eye = torch.eye(r, dtype=B1.dtype, device=B1.device)
        T = g2d.pow(2)

        # Alternate: fit pair 1 to T/(B2 A2 + eps), then pair 2 to T/(B1 A1 + eps).
        t1 = T / ((B2 @ A2) + eps)
        A1.lerp_(torch.linalg.solve(B1.mT @ B1 + eps * eye, B1.mT @ t1), w)
        B1.lerp_(
            torch.linalg.solve((A1 @ A1.mT + eps * eye).mT, A1 @ t1.mT).mT, w
        )

        t2 = T / ((B1 @ A1) + eps)
        A2.lerp_(torch.linalg.solve(B2.mT @ B2 + eps * eye, B2.mT @ t2), w)
        B2.lerp_(
            torch.linalg.solve((A2 @ A2.mT + eps * eye).mT, A2 @ t2.mT).mT, w
        )

        v = ((B1 @ A1) * (B2 @ A2)).clamp_min(eps)
        u = g2d / (v.sqrt() + eps)
        return LoFac._finish(
            u, g2d, p2d, weight_decay, eps, rms_normalize, orthogonalize,
            rescale_ortho, ortho_dtype, rms_clip, cautious_update, cautious_wd,
        )

    @staticmethod
    def _core_lokr(
        g2d: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        a: int,
        b: int,
        c: int,
        d: int,
        p2d: torch.Tensor,
        w: torch.Tensor,
        weight_decay: float,
        eps: float,
        rms_normalize: bool,
        orthogonalize: bool,
        rescale_ortho: bool,
        ortho_dtype: torch.dtype,
        rms_clip: bool,
        cautious_update: bool,
        cautious_wd: bool,
    ) -> torch.Tensor:
        """LoKr-mode core: v = kron(A, B) with T viewed as (a, b, c, d).

        ALS closed forms are weighted means: given B, the best A is
        A[i,k] = sum_{j,l} T4[i,j,k,l] B[j,l] / ||B||^2 (and vice versa).
        """
        T4 = g2d.pow(2).view(a, b, c, d)

        A.lerp_(torch.einsum("ijkl,jl->ik", T4, B) / ((B * B).sum() + eps), w)
        B.lerp_(torch.einsum("ijkl,ik->jl", T4, A) / ((A * A).sum() + eps), w)

        u = g2d / (torch.kron(A, B).clamp_min(eps).sqrt() + eps)
        return LoFac._finish(
            u, g2d, p2d, weight_decay, eps, rms_normalize, orthogonalize,
            rescale_ortho, ortho_dtype, rms_clip, cautious_update, cautious_wd,
        )

    @staticmethod
    def _core_plain(
        g2d: torch.Tensor,
        exp_avg_sq: torch.Tensor,
        p2d: torch.Tensor,
        w: torch.Tensor,
        weight_decay: float,
        eps: float,
        rms_normalize: bool,
        orthogonalize: bool,
        rescale_ortho: bool,
        ortho_dtype: torch.dtype,
        rms_clip: bool,
        cautious_update: bool,
        cautious_wd: bool,
    ) -> torch.Tensor:
        """1D/0D (or non-factorizable) path: plain elementwise second moment.

        Always computed on the 2D reshape (1D/0D params become (1, n)) so the
        compiled core has a single code path with no rank-dependent branching,
        and the optional orthogonalization gets a proper matrix.
        """
        exp_avg_sq.lerp_(g2d.pow(2), w)
        u = g2d / (exp_avg_sq.clamp_min(eps).sqrt() + eps)
        return LoFac._finish(
            u, g2d, p2d, weight_decay, eps, rms_normalize, orthogonalize,
            rescale_ortho, ortho_dtype, rms_clip, cautious_update, cautious_wd,
        )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            polybeta = group["polybeta"]
            weight_decay = group["weight_decay"]
            rms_normalize = group["rms_normalize"]
            orthogonalize = group["orthogonalize"]
            rescale_ortho = group["rescale_ortho"]
            rms_clip = group["rms_clip"]
            cautious_update = group["cautious_update"]
            cautious_wd = group["cautious_wd"]
            stochastic_fp = group["stochastic_fp"]
            ortho_dtype = group["ortho_dtype"]
            eps = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if len(state) == 0:
                    self._init_state(state, p)
                state["step"] += 1
                w = 1.0 - _poly_beta(polybeta, state["step"])
                state["w_t"].fill_(w)

                grad = p.grad.data
                if grad.dtype in (torch.bfloat16, torch.float16):
                    grad = grad.float()
                p_fp32 = (
                    p.data.float()
                    if p.dtype in (torch.bfloat16, torch.float16)
                    else p.data.clone()
                )

                mode = state["mode"]
                if mode == "plain":
                    update = self._compiled_plain(
                        _reshape_to_2d(grad), state["exp_avg_sq"],
                        _reshape_to_2d(p_fp32), state["w_t"], weight_decay, eps,
                        rms_normalize, orthogonalize, rescale_ortho, ortho_dtype,
                        rms_clip, cautious_update, cautious_wd,
                    )
                    update = update.view_as(p_fp32)
                else:
                    g2d = _reshape_to_2d(grad)
                    p2d = _reshape_to_2d(p_fp32)
                    if mode == "lora":
                        update = self._compiled_lora(
                            g2d, state["B"], state["A"], p2d, state["w_t"], weight_decay, eps,
                            rms_normalize, orthogonalize, rescale_ortho, ortho_dtype,
                            rms_clip, cautious_update, cautious_wd,
                        )
                    elif mode == "loha":
                        update = self._compiled_loha(
                            g2d, state["B1"], state["A1"], state["B2"], state["A2"],
                            p2d, state["w_t"], weight_decay, eps, rms_normalize, orthogonalize,
                            rescale_ortho, ortho_dtype, rms_clip, cautious_update,
                            cautious_wd,
                        )
                    else:  # "lokr"
                        update = self._compiled_lokr(
                            g2d, state["A"], state["B"], state["a"], state["b"],
                            state["c"], state["d"], p2d, state["w_t"], weight_decay, eps,
                            rms_normalize, orthogonalize, rescale_ortho, ortho_dtype,
                            rms_clip, cautious_update, cautious_wd,
                        )
                    update = update.view_as(p_fp32)

                p_fp32.add_(update, alpha=-lr)

                if p.dtype is torch.bfloat16 and stochastic_fp:
                    copy_stochastic_(p.data, p_fp32)
                else:
                    p.data.copy_(p_fp32)

        return loss
