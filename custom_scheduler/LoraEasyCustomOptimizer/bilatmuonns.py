# Source: https://github.com/Clybius/Personalized-Optimizers

"""BilatMuonNS: bilateral Newton-Schulz with an NS-only randomized factorization.

A copy of BilatMuonFast focused on per-step latency: the SVD / randomized-SVD
(QR + small SVD) / ALS-solve decomposition that seeds the two momentum factors
is replaced by a Newton-Schulz-based randomized factorization, so the only
linear-algebra primitive left in the step is matrix multiplication. Everything
else -- the bilateral (mutual) Newton-Schulz on the factor pair, the compose
tail, the cautious sign-flip / weight-decay, the poly-beta momentum, the
stochastic bf16 state and the single torch.compile'd fused graph -- is the
BilatMuonFast machinery, unchanged.

The key identity is the same one that makes BilatMuon's mutual NS exact: for
ANY factorization M = A B, the polynomial identity p(XY)X = Xp(YX) gives

    p(M M^T) M = A p(G_B G_A) B = M p(M^T M),   G_A = A^T A, G_B = B B^T,

so the bilateral substep (shared r x r polynomial in the cross-Gramian C =
G_B G_A applied to both sides) is exactly one Newton-Schulz step on the full
product M. The expensive decomposition was never needed for the NS itself --
only to *seed* the pair (A, B) each step. This file seeds it with a
randomized range finder whose orthogonalization step is itself a Newton-Schulz
iteration (not a QR), and whose rank truncation is a greedy row-energy
selection (not a small SVD):

    1. M <- beta(t) M + (1 - beta(t)) g      (poly-beta momentum, bf16 state)
    2. Sketch  Y = M @ Omega                 (Omega a fixed per-shape n x k draw)
    3. Concentrate  Y <- M (M^T Y)  x `power` (range-finder power iterations)
    4. Orthogonalize  Q = NS(Y)              (one-sided Gram-NS, replaces QR)
    5. Project  B = Q^T M                    (k x n coordinates of M in span(Y))
    6. Select the top-r rows of B by energy   (greedy truncation, replaces the
       small SVD; only active when oversample > 0)
    7. Mutual NS on the pair (Q_r, B_r), rebalance, compose delta = Q_r @ B_r,
       rescale (unit energy or pre-ortho norm), cautious tail.

Steps 2-6 are the "alternative bilateral scheme": a factored polar of the
momentum computed entirely by matmuls and norms. With rank=None (full rank)
there is no factorization and no pair -- the one-sided Gram-NS is run directly
on the m x n momentum. That is exactly the bilateral fixed point in the
degenerate limit (the mutual NS with the identity factor: C = G_B G_A = I G_A
= G_A, one shared polynomial on the momentum's own Gramian), so it is the
same algorithm family, Muon-style.

Measured per-step cost (CUDA, 512 x 512 momentum, rank 64, torch.compile'd):
full/svd 107 ms, rSVD 8.6 ms, ALS 1.6 ms, this file's NS factorization
0.93 ms at full rank / 1.37 ms at rank 64 -- the step is one compiled graph, so
the remaining overhead is kernel time, and the NS work runs as bf16 matmuls.

Robustness notes (inherited from BilatMuonFast):

- Non-finite gradients are zeroed in-device: the momentum decays one step and
  the cautious flip emits a zero update, instead of a host-sync skip.
- Sinkhorn preconditioning uses eps-additive denominators (zero marginals stay
  zero) instead of falling back to whole-tensor RMS.
- No SVD / QR / linalg.solve means no LinAlgError retry path: a compile or
  graph failure degrades that key to the raw eager function.

Compile strategy (mirrors BilatMuonFast): one compiled graph per (mode, shape,
rank, hyperparameter) key, `fullgraph=True` by default so any accidental graph
break surfaces as a compile error and degrades that key to eager. lr and step
are runtime tensors so graphs never recompile for schedulers.

Self-contained: `GRAM_NEWTON_SCHULZ_2STEP_COEFFS`, `_poly_beta`,
`_power_lambda_max`, `_rebalance_pair` and `_stoch_round_fp32` are vendored
below (repo convention, cf. BilatMuon.py) -- no imports from the other
optimizer files.
"""

import math
from typing import Dict, Optional, Tuple

import torch
from torch.optim import Optimizer


GRAM_NEWTON_SCHULZ_2STEP_COEFFS = [
    (1.4897216394163149, -0.5798724169434551, 0.0831346315615072),
    (2.0181598271548000, -1.5523232773433393, 0.5343894201774000),
]


def _stoch_round_fp32(x: torch.Tensor) -> torch.Tensor:
    """Stochastically round an fp32 tensor to bf16 precision, kept in fp32."""
    result = torch.randint_like(x, dtype=torch.int32, low=0, high=(1 << 16))
    result.add_(x.view(dtype=torch.int32))
    result.bitwise_and_(-65536)
    return result.view(dtype=torch.float32)


def _poly_beta(polybeta: float, step: int) -> float:
    """Poly-beta debiased EMA weight: 0 at step <= 1, asymptotes to polybeta."""
    if step <= 1:
        return 0.0
    beta_pow = polybeta ** step
    return (beta_pow - polybeta) / (beta_pow - 1.0)


def _power_lambda_max(C: torch.Tensor, iters: int = 6) -> torch.Tensor:
    """Power-iteration estimate of the largest eigenvalue of C.

    C is a product of two PSD Gramians (or a single Gramian), so its
    eigenvalues are real and >= 0. Scaling C by this value before the NS
    polynomial puts the spectrum inside [0, 1] (the polynomial's contraction
    region), which the AOL-Gram folding alone cannot guarantee for a product.
    Deterministic (ones-initialized vector) so the step is reproducible.
    """
    v = torch.ones(C.shape[0], dtype=C.dtype, device=C.device)
    v = v / (v.norm() + 1e-30)
    for _ in range(iters):
        v = C @ v
        v = v / (v.norm() + 1e-30)
    return ((C @ v).norm() / (v.norm() + 1e-30)).clamp_min(1e-30)


def _rebalance_pair(
    a: torch.Tensor,
    b: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Norm-rebalance the pair so ||a||_F == ||b||_F (NaN-safe).

    The composed product and cross-Gramian are invariant under the
    pair-symmetric scaling, but the balance pins the factor scale. Degenerate
    pairs (a factor norm exactly 0, or a non-finite ratio from subnormal-regime
    norm computations in compiled graphs) are left balanced at the clamped
    floor rather than dividing by zero: c = 0 (or NaN) would poison the
    persistent factors with NaN/Inf.
    """
    na = a.norm().clamp_min(1e-30).double()
    nb = b.norm().clamp_min(1e-30).double()
    c = (nb / na).sqrt().float()
    c = torch.where(torch.isfinite(c), c, torch.ones_like(c))
    return a * c, b / c.clamp_min(1e-30)


def _precond_fused(
    g: torch.Tensor,
    precondition: str,
    sinkhorn_iters: int,
    eps: float,
) -> torch.Tensor:
    """Branchless RMS-Sinkhorn / RMS gradient preconditioning.

    Same math as ``_precondition_grad`` with the data-dependent pieces removed:
    non-finite entries are zeroed, and degenerate row/column marginals (zero
    rows/cols) are absorbed by eps-clamped denominators (a zero marginal stays
    zero) instead of the whole-tensor RMS fallback. 1D/0D tensors (static shape
    checks) and "rms" fall through to the scale-only division.
    """
    g = torch.where(torch.isfinite(g), g, torch.zeros_like(g))
    if precondition == "none":
        return g
    if precondition == "rms" or g.shape[0] < 2 or g.shape[1] < 2:
        return g / g.pow(2).mean().sqrt().clamp_min(eps)
    h = g
    for _ in range(sinkhorn_iters):
        h = h / h.pow(2).mean(dim=1, keepdim=True).sqrt().add_(eps)
        h = h / h.pow(2).mean(dim=0, keepdim=True).sqrt().add_(eps)
    return h


def _ns_fused(
    X: torch.Tensor,
    ns_substeps: int,
    eps: float,
    ortho_dtype: torch.dtype,
) -> torch.Tensor:
    """One-sided Gram Newton-Schulz with AOL-Gram folding, branchless.

    Applies the same fold + cubic polynomial substeps the mutual NS applies to
    each factor, but to a single matrix. The Gramian is always the *smaller*
    side (the transpose is taken when X is taller than wide, a static shape
    branch that inductor folds), so the cost is O(ns * min(m,n)^2 * max(m,n)).
    The degenerate-Gramian guard (power-iterated lambda_max ~ 0) skips the
    polynomial, leaving X untouched bit-for-bit. Deterministic and
    compiled-graph-safe.
    """
    dtype = X.dtype
    X = X.to(ortho_dtype)
    transposed = X.size(0) > X.size(1)
    if transposed:
        X = X.mT
    for i in range(ns_substeps):
        G = X @ X.mT
        s = G.abs().sum(dim=-1).clamp_min(eps).rsqrt()
        X = X * s.unsqueeze(-1)
        G = X @ X.mT
        lam = _power_lambda_max(G)
        G = torch.where(
            lam > 1e-10 * G.abs().sum().clamp_min(1e-30),
            G / lam,
            G,
        )
        a, b, c = GRAM_NEWTON_SCHULZ_2STEP_COEFFS[i % 2]
        I = torch.eye(G.shape[0], dtype=G.dtype, device=G.device)
        P = a * I + b * G + c * (G @ G)
        X = P @ X
    if transposed:
        X = X.mT
    return X.to(dtype)


def _rns_pair_fused(
    M: torch.Tensor,
    rank: int,
    omega: torch.Tensor,
    oversample: int,
    power: int,
    ns_substeps: int,
    eps: float,
    ortho_dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Randomized Newton-Schulz factorization M = A B, all matmul.

    The alternative to the SVD/rsvd split: sketch Y = M Omega, concentrate with
    `power` range-finder iterations, orthogonalize the sketch with Newton-Schulz
    (in place of QR), project B = Q^T M, and keep the top-`rank` rows of B by
    row energy (in place of the small SVD's truncation). `omega` is the cached
    fixed n x k draw; the selection is a no-op when oversample makes k == rank.
    """
    k = omega.shape[1]
    Y = M @ omega
    for _ in range(power):
        Y = M @ (M.mT @ Y)
    Q = _ns_fused(Y, ns_substeps, eps, ortho_dtype)
    B = Q.mT @ M
    if k > rank:
        e = B.pow(2).sum(dim=1)
        idx = e.topk(rank).indices
        Q = Q[:, idx]
        B = B[idx, :]
    return Q, B


def _bns_fused(
    A: torch.Tensor,
    B: torch.Tensor,
    ns_substeps: int,
    eps: float,
    ortho_dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Mutual Newton-Schulz (AOL-Gram folded), branchless.

    Identical to ``bilateral_newton_schulz`` with fold=True, including the
    exact degenerate cross-Gramian guard: the polynomial is skipped when the
    power-iterated lambda_max is ~0 (the `torch.where` then leaves C untouched,
    bit-for-bit what the Python branch did). The scaling uses `C / lam` -- not
    `C * (1/lam)` -- so the compiled arithmetic matches the original's
    bit-for-bit. The shared identity is hoisted out of the substep loop.
    """
    dtype = A.dtype
    A = A.to(ortho_dtype)
    B = B.to(ortho_dtype)
    I = torch.eye(A.shape[1], dtype=A.dtype, device=A.device)
    for i in range(ns_substeps):
        GA = A.mT @ A
        GB = B @ B.mT
        sA = GA.abs().sum(dim=-1).clamp_min(eps).rsqrt()
        sB = GB.abs().sum(dim=-1).clamp_min(eps).rsqrt()
        A = A * sA.unsqueeze(0)
        B = B * sB.unsqueeze(-1)
        GA = A.mT @ A
        GB = B @ B.mT
        C = GB @ GA
        lam = _power_lambda_max(C)
        C = torch.where(
            lam > 1e-10 * C.abs().sum().clamp_min(1e-30),
            C / lam,
            C,
        )
        a, b, c = GRAM_NEWTON_SCHULZ_2STEP_COEFFS[i % 2]
        P = a * I + b * C + c * (C @ C)
        A = A @ P
        B = P @ B
    return A.to(dtype), B.to(dtype)


def _compose_tail_fused(
    A: torch.Tensor,
    B: torch.Tensor,
    rank: int,
    eps: float,
    ortho_dtype: torch.dtype,
    ns_substeps: int,
    rescale_ortho: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Mutual NS -> compose -> scale -> rebalance -> finite guard, branchless.

    Same semantics as ``_compose_tail``; the degenerate-product guard
    `denom > 1e-10 * ...` becomes a `torch.where` scale (1.0 when degenerate)
    instead of a Python branch. Returns the balanced pair (U, V) and the
    rescaled delta, all finite-guarded.
    """
    pre_norm = torch.trace((A.mT @ A) @ (B @ B.mT)).clamp_min(0.0).double().sqrt().float()
    u_hat, v_hat = _bns_fused(A, B, ns_substeps, eps, ortho_dtype)
    u_hat, v_hat = _rebalance_pair(u_hat, v_hat)
    delta = u_hat @ v_hat
    denom = delta.norm()
    thresh = 1e-10 * (u_hat.norm() * v_hat.norm()).clamp_min(1e-30)
    target = pre_norm if rescale_ortho else math.sqrt(rank)
    scale = torch.where(denom > thresh, target / denom.clamp_min(1e-30),
                        torch.ones_like(denom))
    delta = delta * scale
    return (
        torch.where(torch.isfinite(u_hat), u_hat, torch.zeros_like(u_hat)),
        torch.where(torch.isfinite(v_hat), v_hat, torch.zeros_like(v_hat)),
        torch.where(torch.isfinite(delta), delta, torch.zeros_like(delta)),
    )


def _compose_single_fused(
    M: torch.Tensor,
    rank: int,
    eps: float,
    ortho_dtype: torch.dtype,
    ns_substeps: int,
    rescale_ortho: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Full-rank one-sided compose: NS on the momentum -> scale -> finite guard.

    The degenerate limit of the bilateral scheme (the mutual NS with the
    identity factor), for rank=None. Returns the orthogonalized factor (as both
    U and V, for a uniform API) and the rescaled delta.
    """
    pre_norm = M.norm().clamp_min(0.0).float()
    u_hat = _ns_fused(M, ns_substeps, eps, ortho_dtype)
    denom = u_hat.norm()
    thresh = 1e-10 * u_hat.norm().clamp_min(1e-30)
    target = pre_norm if rescale_ortho else math.sqrt(rank)
    scale = torch.where(denom > thresh, target / denom.clamp_min(1e-30),
                        torch.ones_like(denom))
    delta = u_hat * scale
    return (
        torch.where(torch.isfinite(u_hat), u_hat, torch.zeros_like(u_hat)),
        torch.where(torch.isfinite(u_hat), u_hat, torch.zeros_like(u_hat)),
        torch.where(torch.isfinite(delta), delta, torch.zeros_like(delta)),
    )


def _finalize(
    delta: torch.Tensor,
    g: torch.Tensor,
    p: torch.Tensor,
    lr: torch.Tensor,
    wd: float,
    cautious_update: bool,
    cautious_wd: bool,
) -> None:
    """Cautious sign flip, cautious weight decay, in-place param update.

    `cautious_update`, `cautious_wd` and `wd` are compile-time constants
    (baked by dynamo), so wd == 0.0 emits no decay kernels at all. The
    parameter update uses `addcmul_(..., value=-1)` -- bit-identical to the
    original's `p.add_(delta, alpha=-lr)` (verified). The parameter is mutated
    in place at the very end, after every read.
    """
    if cautious_update:
        delta = g.sign() * delta.abs()
    if wd != 0.0:
        p2 = p.reshape(g.shape)
        if cautious_wd:
            s = torch.where(p2.sign() == delta.sign(),
                            torch.ones_like(delta), -torch.ones_like(delta))
            decay = wd * s * p2
        else:
            decay = wd * p2
        delta = delta + decay
    p.addcmul_(delta.reshape(p.shape), lr, value=-1.0)


def _write_state(buf: torch.Tensor, x: torch.Tensor, stochastic_bf16: bool) -> None:
    """Stochastic-round (bf16) or plain copy of x into the state buffer.

    `stochastic_bf16` is a compile-time constant. The randint / add / truncate
    / convert / copy chain fuses into a single kernel inside the compiled graph;
    the original eager `copy_stochastic_` ran the same ops as separate kernels.
    """
    if stochastic_bf16:
        r = torch.randint_like(x, dtype=torch.int32, low=0, high=(1 << 16))
        y = (r + x.view(dtype=torch.int32)).bitwise_and_(-65536).view(
            dtype=torch.float32)
        buf.copy_(y.to(dtype=torch.bfloat16))
    else:
        buf.copy_(x)


def _fused_plain(
    p: torch.Tensor,
    m_buf: torch.Tensor,
    g: torch.Tensor,
    lr: torch.Tensor,
    wd: float,
    poly: torch.Tensor,
    quantize_grad: bool,
    cautious_update: bool,
    cautious_wd: bool,
    stochastic_bf16: bool,
    precondition: str,
    sinkhorn_iters: int,
    eps: float,
) -> None:
    with torch.no_grad():
        """1D/0D fallback: precondition, poly-beta momentum, cautious tail."""
        g = _precond_fused(g, precondition, sinkhorn_iters, eps)
        if quantize_grad:
            g = _stoch_round_fp32(g)
        m = m_buf.float() * poly + (1.0 - poly) * g
        m = torch.where(torch.isfinite(m), m, g)
        _finalize(m, g, p, lr, wd, cautious_update, cautious_wd)
        _write_state(m_buf, m, stochastic_bf16)


def _fused_bilat(
    p: torch.Tensor,
    M_buf: torch.Tensor,
    g: torch.Tensor,
    lr: torch.Tensor,
    wd: float,
    poly: torch.Tensor,
    omega: Optional[torch.Tensor],
    rank: int,
    quantize_grad: bool,
    eps: float,
    ortho_dtype: torch.dtype,
    ns_substeps: int,
    rescale_ortho: bool,
    cautious_update: bool,
    cautious_wd: bool,
    stochastic_bf16: bool,
    precondition: str,
    sinkhorn_iters: int,
    oversample: int,
    power: int,
    use_rns: bool,
) -> None:
    with torch.no_grad():
        """Fused 2D+ step. `use_rns` is a compile-time constant baked by
        dynamo: the low-rank branch runs the randomized NS factorization +
        mutual NS pair, the full-rank branch runs the one-sided NS compose."""
        g = _precond_fused(g, precondition, sinkhorn_iters, eps)
        if quantize_grad:
            g = _stoch_round_fp32(g)
        M = M_buf.float() * poly + (1.0 - poly) * g
        M = torch.where(torch.isfinite(M), M, g)
        if use_rns:
            A, B = _rns_pair_fused(M, rank, omega, oversample, power,
                                   ns_substeps, eps, ortho_dtype)
            u_hat, v_hat, delta = _compose_tail_fused(
                A, B, rank, eps, ortho_dtype, ns_substeps, rescale_ortho)
        else:
            u_hat, v_hat, delta = _compose_single_fused(
                M, rank, eps, ortho_dtype, ns_substeps, rescale_ortho)
        _finalize(delta, g, p, lr, wd, cautious_update, cautious_wd)
        _write_state(M_buf, M, stochastic_bf16)


class BilatMuonNS(Optimizer):
    r"""BilatMuonNS: bilateral Newton-Schulz with an NS-only factorization.

    Same algorithm family, state and tail as :class:`BilatMuonFast` (see the
    module docstring for the full description); the difference is entirely in
    how the momentum is factored: the per-step SVD / randomized-SVD (QR + small
    SVD) / ALS decomposition is replaced by a randomized Newton-Schulz range
    finder, so the step is all matmuls. With ``rank`` set below ``min(m, n)``
    the update is the mutual-NS composition of the pair (Q_r, Q_r^T M) built
    from the sketch; with ``rank=None`` (default) the full-rank momentum is
    orthogonalized directly by the one-sided Gram-NS (the degenerate-limit
    bilateral, Muon-style).

    Args:
        params: iterable of parameters to optimize.
        lr (float): learning rate (default: 1e-2). The update is the
            orthogonalized composition with a flat unit spectrum (all nonzero
            singular values ~1), so its RMS stays near O(1) regardless of
            gradient scale; the step size lives in lr (unless ``rescale_ortho``
            restores the pre-orthogonalization norm).
        betas (Tuple[float, float]): poly-beta momentum rate for the momentum
            buffer (default: (0.99, 0.99); only betas[0] is used).
        weight_decay (float): decoupled weight decay (default: 0.0). Applied
            cautiously (sign-flip variant) by default.
        rank (int, optional): rank of the factorization (default: None = full
            rank min(m, n)). With rank < min(m, n) the randomized NS bilateral
            scheme engages (fast path; the update is truncated to the top rank
            sketch directions).
        oversample (int): oversampling for the randomized range finder
            (default: 8). Oversampled directions are pruned by row energy, so
            the composed update has exactly ``rank`` active directions.
        power (int): power iterations for the randomized range finder
            (default: 1). Concentrates the sketch on the dominant directions.
        ns_substeps (int): Newton-Schulz substeps, cycling the repo's
            pre-optimized cubic coefficient pairs, used both for the sketch
            orthogonalization and the mutual orthogonalization (default: 4;
            empirical Rosenbrock optimum -- 2 substeps leaves the polar
            partially converged, 0.089 vs 0.022 at 4, the SVD path's 0.024).
        rescale_ortho (bool): rescale the composed update back to the
            pre-orthogonalization momentum norm (norm-preserving Muon). With
            False the update is normalized to unit Frobenius energy and lr
            carries the magnitude (default: True).
        cautious_update (bool): flip the sign of every update component that
            disagrees with the raw gradient instead of masking it (default: True).
        cautious_wd (bool): apply weight decay with a flipped sign wherever the
            decay would fight the update (default: True).
        stochastic_bf16 (bool): store the momentum state in bf16 with
            stochastic rounding on write (default: True). fp32 master weights
            are kept.
        quantize_grad (bool): stochastically round the raw gradient to bf16
            precision before the momentum (default: False; unbiased, halves the
            matmul cost).
        precondition (str): gradient preconditioner applied before the momentum
            (default: "rms"). "sinkhorn" -- RMS Sinkhorn row/col balancing
            (``sinkhorn_iters`` iterations; falls back to "rms" where it cannot
            apply). "rms" -- divide the whole gradient by its RMS. "none".
        sinkhorn_iters (int): iterations for the "sinkhorn" preconditioner
            (default: 3).
        eps (float): stability constant for the Newton-Schulz folding
            (default: 1e-8).
        ortho_dtype (torch.dtype): compute dtype for the Newton-Schulz work
            (default: torch.bfloat16).
        compile_step (bool): torch.compile the per-shape fused step with eager
            fallback (default: True).
        compile_mode (str): torch.compile mode, "default" (default) or
            "max-autotune".
        fullgraph (bool): require a break-free graph (default: False). If the
            environment cannot compile the step break-free, that key falls back
            to eager.

    State per 2D+ parameter: one momentum buffer (m x n, bf16 by default) plus
    a step counter. 1D/0D parameters keep a single momentum buffer.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-2,
        betas: Tuple[float, float] = (0.99, 0.99),
        weight_decay: float = 0.0,
        rank: Optional[int] = 64,
        oversample: int = 8,
        power: int = 1,
        ns_substeps: int = 4,
        rescale_ortho: bool = True,
        cautious_update: bool = True,
        cautious_wd: bool = True,
        stochastic_bf16: bool = True,
        quantize_grad: bool = False,
        precondition: str = "rms",
        sinkhorn_iters: int = 3,
        eps: float = 1e-8,
        ortho_dtype: torch.dtype = torch.bfloat16,
        compile_step: bool = True,
        compile_mode: str = "default",
        fullgraph: bool = False,
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid lr: {lr}")
        if not (0.0 <= betas[0] < 1.0 and 0.0 <= betas[1] < 1.0):
            raise ValueError(f"Invalid betas: {betas}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        if rank is not None and rank < 1:
            raise ValueError(f"Invalid rank: {rank}")
        if oversample < 1 or power < 0:
            raise ValueError(
                f"Invalid oversample/power: {oversample}, {power}")
        if ns_substeps < 1:
            raise ValueError(f"Invalid ns_substeps: {ns_substeps}")
        if precondition not in ("none", "sinkhorn", "rms"):
            raise ValueError(f"Invalid precondition: {precondition}")
        if sinkhorn_iters < 1:
            raise ValueError(f"Invalid sinkhorn_iters: {sinkhorn_iters}")
        if compile_mode not in ("default", "max-autotune"):
            raise ValueError(f"Invalid compile_mode: {compile_mode}")
        defaults = dict(
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
            rank=rank,
            oversample=oversample,
            power=power,
            ns_substeps=ns_substeps,
            rescale_ortho=rescale_ortho,
            cautious_update=cautious_update,
            cautious_wd=cautious_wd,
            stochastic_bf16=stochastic_bf16,
            quantize_grad=quantize_grad,
            precondition=precondition,
            sinkhorn_iters=sinkhorn_iters,
            eps=eps,
            ortho_dtype=ortho_dtype,
            compile_step=compile_step,
            compile_mode=compile_mode,
            fullgraph=fullgraph,
        )
        super().__init__(params, defaults)
        self._compiled: Dict[tuple, callable] = {}
        self._omegas: Dict[tuple, torch.Tensor] = {}
        self._scalars: Dict[str, tuple] = {}
        self._polys: Dict[tuple, tuple] = {}
        self._precision_set = False

    def _scalar(self, key: str, value: float, device: torch.device) -> torch.Tensor:
        """Cached 0-dim fp32 tensor for a per-step Python scalar (lr)."""
        if isinstance(value, torch.Tensor):
            return value.to(device=device, dtype=torch.float32)
        entry = self._scalars.get(key)
        if entry is not None and entry[0] == device and entry[1] == value:
            return entry[2]
        t = torch.tensor(value, dtype=torch.float32, device=device)
        self._scalars[key] = (device, value, t)
        return t

    def _poly(self, beta: float, step: int, device: torch.device) -> torch.Tensor:
        """Cached poly-beta debiasing weight, bit-identical to the original's
        host-side ``_poly_beta`` (fp64 pow -> fp32 tensor). Computed once per
        (device, beta, step) and shared by every parameter at that step."""
        entry = self._polys.get((device, beta))
        if entry is not None and entry[0] == step:
            return entry[1]
        t = torch.full((), _poly_beta(beta, step), dtype=torch.float32,
                       device=device)
        self._polys[(device, beta)] = (step, t)
        return t

    def _set_precision(self) -> None:
        try:
            torch.set_float32_matmul_precision('high')
        except Exception:
            pass

    def _init_state(self, p: torch.Tensor, group: dict) -> None:
        state = self.state[p]
        state["step"] = 0
        state["step_t"] = torch.zeros((), dtype=torch.int64, device=p.device)
        dtype = torch.bfloat16 if group["stochastic_bf16"] else torch.float32
        m, n = (p.shape[0], p.numel() // p.shape[0]) if p.ndim > 1 else (1, p.numel())
        if p.ndim <= 1:
            state["m"] = torch.zeros((1, n), dtype=dtype, device=p.device)
        else:
            state["M"] = torch.zeros((m, n), dtype=dtype, device=p.device)

    def _run(
        self,
        key: tuple,
        fn: callable,
        args: tuple,
        compile_mode: str,
        fullgraph: bool,
    ) -> None:
        """Run the fused step (compiled, break-free), with an eager fallback.

        There is no LinAlgError retry (no SVD/solve in the graph). If the first
        compiled execution fails to compile or execute for any reason, that key
        permanently degrades to the eager function.
        """
        core = self._compiled.get(key)
        if core is None:
            if not self._precision_set:
                self._set_precision()
                self._precision_set = True
            try:
                core = torch.compile(fn, mode=compile_mode, fullgraph=fullgraph)
            except Exception:
                core = fn
            self._compiled[key] = core
            if core is not fn:
                try:
                    core(*args)
                except Exception:
                    try:
                        fn(*args)
                    except Exception:
                        raise
                    self._compiled[key] = fn
                return
        if core is fn:
            fn(*args)
            return
        try:
            core(*args)
        except Exception:
            fn(*args)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            wd = float(group["weight_decay"])
            betas = group["betas"]
            rank_opt = group["rank"]
            oversample = group["oversample"]
            power = group["power"]
            ns_substeps = group["ns_substeps"]
            rescale_ortho = group["rescale_ortho"]
            cautious_update = group["cautious_update"]
            cautious_wd = group["cautious_wd"]
            stochastic_bf16 = group["stochastic_bf16"]
            quantize_grad = group["quantize_grad"]
            precondition = group["precondition"]
            sinkhorn_iters = group["sinkhorn_iters"]
            eps = group["eps"]
            ortho_dtype = group["ortho_dtype"]
            compile_step = group["compile_step"]
            compile_mode = group["compile_mode"]
            fullgraph = group["fullgraph"]
            for p in group["params"]:
                g = p.grad
                if g is None:
                    continue
                if g.is_sparse:
                    raise RuntimeError(
                        "BilatMuonNS does not support sparse gradients")
                if not g.is_contiguous():
                    g = g.contiguous()
                state = self.state[p]
                if not state:
                    self._init_state(p, group)
                state["step"] += 1
                step_t = state["step"]
                state["step_t"].add_(1)
                lr_t = self._scalar("lr", lr, p.device)
                if p.ndim <= 1:
                    g2 = g.reshape(1, g.numel())
                    poly = self._poly(betas[0], step_t, p.device)
                    baked = (quantize_grad, cautious_update,
                             cautious_wd, stochastic_bf16, precondition,
                             sinkhorn_iters, eps, wd)
                    key = ("plain", g2.numel()) + baked
                    fn = _fused_plain
                    args = (p, state["m"], g2, lr_t, wd, poly,
                            quantize_grad, cautious_update, cautious_wd,
                            stochastic_bf16, precondition, sinkhorn_iters, eps)
                else:
                    m = g.shape[0]
                    n = g.numel() // m
                    r_full = min(m, n)
                    r = r_full if rank_opt is None else min(rank_opt, m, n)
                    r = max(r, 1)
                    # The randomized NS bilateral scheme engages when the rank
                    # dial r < min(m, n); at full rank the one-sided NS compose
                    # is used (the degenerate-limit bilateral).
                    use_rns = r < r_full
                    eff_rank = r if use_rns else r_full
                    g2 = g.reshape(m, n)
                    omega = None
                    if use_rns:
                        k = min(r + oversample, m, n)
                        omega = self._omegas.get((n, k))
                        if omega is None:
                            gen = torch.Generator(device=p.device).manual_seed(0)
                            omega = torch.randn(n, k, dtype=torch.float32,
                                                device=p.device, generator=gen)
                            self._omegas[(n, k)] = omega
                    poly = self._poly(betas[0], step_t, p.device)
                    baked = (eff_rank, use_rns, quantize_grad, eps,
                             ortho_dtype, ns_substeps, rescale_ortho,
                             cautious_update, cautious_wd, stochastic_bf16,
                             precondition, sinkhorn_iters, oversample,
                             power, wd)
                    key = ("bilat", m, n) + baked
                    fn = _fused_bilat
                    args = (p, state["M"], g2, lr_t, wd, poly, omega,
                            eff_rank, quantize_grad, eps, ortho_dtype,
                            ns_substeps, rescale_ortho, cautious_update,
                            cautious_wd, stochastic_bf16, precondition,
                            sinkhorn_iters, oversample, power, use_rns)
                if compile_step:
                    self._run(key, fn, args, compile_mode, fullgraph)
                else:
                    fn(*args)
        return loss
