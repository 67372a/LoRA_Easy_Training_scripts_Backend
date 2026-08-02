# Source: https://github.com/Clybius/Personalized-Optimizers

"""BilatMuon: mutual Newton-Schulz over two momentumized factors.

A gradient-descent optimizer that momentumizes the gradient and decomposes
the momentum into two matrices with different properties -- a left/column
factor A (m x r) and a right/row factor B (r x n) with an asymmetric
singular-value split M = (U Sigma^p)(Sigma^q V^T), p + q = 1, so one factor
carries more of the magnitude and the other more of the geometry -- then
crafts the update by orthogonalizing the two factors *against each other*
with a bilateral (mutual) Newton-Schulz iteration on the shared r x r
cross-Gramian, and finally composes the full update tensor.

The mutual orthogonalization is the novel part: for A in R^{m x r},
B in R^{r x n}, M = A B, the polynomial identity p(XY)X = Xp(YX) gives

    p(M M^T) M = A p(G_B G_A) B  =  M p(M^T M),   G_A = A^T A, G_B = B B^T,

so one joint substep with the *same* r x r polynomial P = p(G_B G_A) applied
on both sides -- A <- A P, B <- P B -- is exactly a Newton-Schulz step on the
full product M, computed at O(r^2(m+n)) instead of Muon's O(m^2 n). The
fixed point is G_A G_B = I (the factors are mutually balanced, not
individually orthonormal) and the composed update has a flat spectrum
(sigma = 1). AOL-Gram row-sum folding of each factor per substep stabilizes
the iteration and normalizes the top of the spectrum.

Two momentum architectures:

- mode "full" (default): a full m x n momentum buffer M <- beta M + (1-beta) G
  is the state; the momentum itself is decomposed fresh each step (exact SVD,
  randomized SVD -- 10-100x cheaper when the rank dial r < min(m, n) -- or an
  ALS warm-started half-sweep). With p = q = 1/2 the composed direction is the
  polar factor polar(M) -- a factored polar decomposition of the momentum;
  the update is lr * polar(M) (Muon-style unit magnitude: lr carries the
  step size, the momentum's scale steers the direction only, so the update
  cannot inherit the momentum's retained energy).
- mode "factored": the momentum lives in the two factor buffers (FacMuon
  style: the gradient is decomposed, each factor momentumized with its own
  poly-beta rate), and the mutual NS replaces FacMuon's independent
  per-factor orthogonalization.

Per step (2D+ parameters, gradient g in R^{m x n}, rank r):

0. Optional gradient preconditioning (eager, before the momentum):
   "sinkhorn" -- RMS Sinkhorn row/col balancing (each iteration divides
   every row by its RMS, then every column by its RMS, driving all row and
   column RMS values to 1), falling back to "rms" (whole-tensor division by
   the RMS) wherever Sinkhorn cannot apply: 1D/0D tensors or degenerate
   row/column marginals.

1. Full mode: M <- beta(t) M + (1 - beta(t)) g (poly-beta debiased).
   Factored mode: (a, b) = decompose(g) (SVD split or ALS half-sweep),
   sign-canonicalized, then m_A <- beta_A(t) m_A + (1 - beta_A(t)) a and
   m_B likewise with its own rate.

2. Decompose the momentum (full mode only): A = U Sigma^p, B = Sigma^q V^T
   with p + q = 1 (split hyperparam; the two factors have different
   properties by construction).

3. Mutual orthogonalization (ns_substeps = 2, the repo's pre-optimized
   cubic coefficient pairs): fold both factors, then per substep
   P = p((B B^T)(A^T A)), A <- A P, B <- P B.

4. Composition: delta = A @ B rescaled by a single scalar. Default: unit
   magnitude (Muon convention) -- the composed update is normalized to unit
   Frobenius energy (flat unit spectrum) and the step size lives entirely
   in lr, so retained momentum energy cannot sustain overshoot. Optional
   rescale_ortho=True: the update is rescaled back to the
   pre-orthogonalization product norm ||A @ B||_F (norm-preserving Muon;
   the norm comes from the r x r Gramians, ||A B||_F^2 = trace((A^T A)(B B^T)),
   so the m x n product is never materialized twice).

5. Cautious sign-flip update (novel variant): delta_i = sign(g_i)|delta_i|.

6. Cautious weight decay (flip variant): s_i = +1 where sign(p_i) ==
   sign(delta_i), else -1; decay = wd * s * p.

7. Stochastic bf16: the momentum/factor state is stored in bf16 with
   stochastic rounding on write (copy_stochastic_); fp32 master weights.

Memory: full mode = one m x n momentum buffer (bf16) + step (+ two r(m+n)
ALS warm-start factors); factored mode = exactly two factor tensors of
r(m + n) elements in bf16 + step. 1D/0D tensors use a plain poly-beta
momentum fallback with the same cautious machinery. All
heavy work lives in pure per-mode core functions that can be torch.compiled
(max-autotune) with an eager fallback; everything that changes per step
(betas, step) is passed as a tensor so the compiled graph never recompiles.
"""

import math
from typing import Dict, Optional, Tuple

import torch
from torch.optim import Optimizer

from LoraEasyCustomOptimizer.lofac import (
    GRAM_NEWTON_SCHULZ_2STEP_COEFFS,
    copy_stochastic_,
    _poly_beta,
    _reshape_to_2d,
)


def _stoch_round_fp32(x: torch.Tensor) -> torch.Tensor:
    """Stochastically round an fp32 tensor to bf16 precision, kept in fp32."""
    result = torch.randint_like(x, dtype=torch.int32, low=0, high=(1 << 16))
    result.add_(x.view(dtype=torch.int32))
    result.bitwise_and_(-65536)
    return result.view(dtype=torch.float32)


def _precondition_grad(
    g2: torch.Tensor,
    precondition: str,
    sinkhorn_iters: int,
    eps: float,
) -> torch.Tensor:
    """RMS-Sinkhorn or whole-tensor RMS preconditioning of the gradient.

    "sinkhorn" (repo convention, cf. SinkFactor/SGDF): alternating row/col
    RMS balancing -- each iteration divides every row by its RMS, then every
    column by its RMS -- driving all row and column RMS values to 1 (a
    doubly-stochastic-like equilibrium on the squared gradient). "rms":
    divide the whole tensor by its RMS (scale-only). Sinkhorn falls back to
    "rms" whenever it cannot be applied: 1D/0D tensors (a single row has no
    row/col structure) or any exactly-degenerate row/column marginal
    (Sinkhorn's fixed point requires positive marginals).
    """
    if precondition == "none":
        return g2
    if precondition == "rms" or g2.shape[0] < 2 or g2.shape[1] < 2:
        return g2 / g2.pow(2).mean().sqrt().clamp_min(eps)
    w = g2.pow(2)
    if not (w.mean(dim=1).sqrt() > eps).all() or not (w.mean(dim=0).sqrt() > eps).all():
        return g2 / w.mean().sqrt().clamp_min(eps)
    g = g2
    for _ in range(sinkhorn_iters):
        g = g / g.pow(2).mean(dim=1, keepdim=True).sqrt().add_(eps)
        g = g / g.pow(2).mean(dim=0, keepdim=True).sqrt().add_(eps)
    return g


def _robust_svd(
    A: torch.Tensor,
    gesvd: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""SVD that survives degenerate/clustered singular spectra.

    torch.linalg.svd (LAPACK gesdd / cuSOLVER) can fail to converge on
    matrices whose singular values are clustered or nearly repeated; the
    compiled rsvd core hit this in training, where TF32 matmul noise
    smears the momentum's tiny tail singular values into a degenerate
    cluster. With gesvd=True (the rsvd's small projected SVD; CUDA only,
    but supported inside torch.compile) cuSOLVER's robust gesvd driver
    is used directly. Otherwise gesdd is tried first, and a convergence
    failure is retried in float64 (which resolves fp32-level clustering)
    and finally with scipy's gesvd (Jacobi) when available. Inside a
    compiled graph the try/except is inert -- step() re-runs the raw
    eager core on a failure, which is where these fallbacks execute.
    Non-finite entries (a NaN/Inf gradient poisons the momentum, and the
    SVD drivers report "failed to converge" on such inputs) are zeroed up
    front so the decomposition itself can never crash.
    """
    A = torch.where(torch.isfinite(A), A, torch.zeros_like(A))
    try:
        if gesvd and A.is_cuda:
            return torch.linalg.svd(A, full_matrices=False, driver="gesvd")
        return torch.linalg.svd(A, full_matrices=False)
    except torch.linalg.LinAlgError:
        pass
    try:
        u, s, vh = torch.linalg.svd(A.double(), full_matrices=False)
        return u.float(), s.float(), vh.float()
    except torch.linalg.LinAlgError:
        pass
    try:
        import numpy as np
        from scipy.linalg import svd as _scipy_svd
        u, s, vh = _scipy_svd(
            A.detach().to(torch.float64).cpu().numpy(),
            full_matrices=False,
            lapack_driver="gesvd",
        )
        return (
            torch.from_numpy(u).to(device=A.device, dtype=A.dtype),
            torch.from_numpy(s).to(device=A.device, dtype=A.dtype),
            torch.from_numpy(vh).to(device=A.device, dtype=A.dtype),
        )
    except ImportError:
        raise torch.linalg.LinAlgError(
            "linalg.svd failed to converge and no robust fallback "
            "(scipy) is available") from None


def _svd_split(
    M: torch.Tensor,
    rank: int,
    p: float,
    q: float,
    canonicalize: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Asymmetric-SVD factors (a, b) with a @ b ~= M and p + q = 1.

    a = U[:, :rank] * Sigma[:rank]^p (m x r), b = Sigma[:rank]^q * Vh[:rank]
    (r x n); at full rank the product reconstructs M exactly. Column signs
    of a are canonicalized to +1 at their largest-magnitude entry and
    mirrored onto the rows of b (needed wherever the factors persist).
    """
    u, s, vh = _robust_svd(M)
    # fp64 powers: inductor special-cases pow(x, 0.5) to tl.sqrt_rn, which
    # returns NaN for subnormal x on some (AMD) backends; the momentum's
    # tiny tail singular values do land subnormal in training.
    sp = s[:rank].clamp_min(0.0).double().pow(p).float()
    sq = s[:rank].clamp_min(0.0).double().pow(q).float()
    a = u[:, :rank] * sp.unsqueeze(0)
    b = sq.unsqueeze(-1) * vh[:rank]
    if canonicalize:
        idx = a.abs().argmax(dim=0)
        cs = a.gather(0, idx.unsqueeze(0)).squeeze(0)
        cs = torch.where(cs == 0.0, torch.ones_like(cs), cs).sign()
        a = a * cs
        b = b * cs.unsqueeze(-1)
    return a, b


def _rsvd_split(
    M: torch.Tensor,
    rank: int,
    p: float,
    q: float,
    omega: torch.Tensor,
    oversample: int,
    power: int,
    canonicalize: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Randomized-SVD factors (a, b) with a @ b ~= M and p + q = 1.

    Standard randomized range finder: Y = (M M^T)^power M Omega (Omega an
    n x (rank + oversample) fixed random projection), QR of Y, project
    B = Q^T M, exact SVD of the k x n projection, compose the left factor
    as Q @ u. Near-optimal rank-r fit (1-3% above the exact truncated-SVD
    error, measured at rank 64) at O(m n k + m k^2) instead of the full
    SVD's O(m n min(m, n)). The mutual NS downstream corrects the
    conditioning, so the approximation error is harmless. Same (p, q)
    split and sign canonicalization as ``_svd_split``; the projection
    Omega is fixed per shape, so the factors are deterministic.
    """
    M = torch.where(torch.isfinite(M), M, torch.zeros_like(M))
    m, n = M.shape
    k = min(rank + oversample, m, n)
    Y = M @ omega
    for _ in range(power):
        Y = M @ (M.mT @ Y)
    Q, _ = torch.linalg.qr(Y)
    B = Q.mT @ M
    u, s, vh = _robust_svd(B, gesvd=True)
    sp = s[:rank].clamp_min(0.0).double().pow(p).float()
    sq = s[:rank].clamp_min(0.0).double().pow(q).float()
    u = (Q @ u[:, :rank]) * sp.unsqueeze(0)
    b = sq.unsqueeze(-1) * vh[:rank]
    if canonicalize:
        idx = u.abs().argmax(dim=0)
        cs = u.gather(0, idx.unsqueeze(0)).squeeze(0)
        cs = torch.where(cs == 0.0, torch.ones_like(cs), cs).sign()
        u = u * cs
        b = b * cs.unsqueeze(-1)
    return u, b


def _canonicalize(
    a: torch.Tensor, b: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sign-canonicalize the column signs of a (mirrored onto b)."""
    idx = a.abs().argmax(dim=0)
    cs = a.gather(0, idx.unsqueeze(0)).squeeze(0)
    cs = torch.where(cs == 0.0, torch.ones_like(cs), cs).sign()
    return a * cs, b * cs.unsqueeze(-1)


def _power_lambda_max(C: torch.Tensor, iters: int = 6) -> torch.Tensor:
    """Power-iteration estimate of the largest eigenvalue of C.

    C = G_B G_A is a product of two PSD Gramians, so its eigenvalues are
    real and >= 0. Scaling C by this value before the NS polynomial puts
    the spectrum inside [0, 1] (the polynomial's contraction region),
    which the AOL-Gram folding alone cannot guarantee for a product.
    Deterministic (ones-initialized vector) so the step is reproducible.
    """
    v = torch.ones(C.shape[0], dtype=C.dtype, device=C.device)
    v = v / (v.norm() + 1e-30)
    for _ in range(iters):
        v = C @ v
        v = v / (v.norm() + 1e-30)
    return ((C @ v).norm() / (v.norm() + 1e-30)).clamp_min(1e-30)


def _als_solve(
    G: torch.Tensor,
    rhs: torch.Tensor,
    als_eps: float,
) -> torch.Tensor:
    r"""Solve (G + ridge I) X = rhs with a noise-proof spectral ridge.

    G is an r x r Gramian (a factor self-product). The ridge is the
    als_eps-relative trace term floored at 1e-2 of G's power-iterated
    largest eigenvalue, so the regularized matrix is positive definite
    with condition number <= ~100 even when the *computed* Gramian is
    indefinite from TF32/bf16 rounding noise or exactly rank-deficient --
    a mathematically-PSD Gramian computed with low-precision matmuls can
    land its smallest eigenvalue at or below zero, which linalg.solve
    reports as "singular" (seen in training). Non-finite entries are
    zeroed up front so the solve can never see NaN. A failed fp32 solve
    is retried in fp64 (the eager fallback; inside a compiled graph the
    error propagates to step()'s _run_core retry instead).
    """
    G = torch.where(torch.isfinite(G), G, torch.zeros_like(G))
    lam = _power_lambda_max(G)
    lam = torch.where(torch.isfinite(lam), lam, torch.zeros_like(lam))
    ridge = als_eps * G.abs().sum().clamp_min(1.0)
    ridge = torch.maximum(ridge, 1e-2 * lam)
    eye = torch.eye(G.shape[0], dtype=G.dtype, device=G.device)
    H = G + ridge * eye
    try:
        return torch.linalg.solve(H, rhs)
    except torch.linalg.LinAlgError:
        return torch.linalg.solve(H.double(), rhs.double()).float()


def _rebalance_pair(
    a: torch.Tensor,
    b: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Norm-rebalance the pair so ||a||_F == ||b||_F (NaN-safe).

    The composed product and cross-Gramian are invariant under the
    pair-symmetric scaling, but the balance pins the ALS scale ambiguity.
    Degenerate pairs (a factor norm exactly 0, or a non-finite ratio
    from subnormal-regime norm computations in compiled graphs) are left
    balanced at the clamped floor rather than dividing by zero: c = 0
    (or NaN) would make b / c poison the persistent factors with
    NaN/Inf, which then collapse the next step's ALS solve.
    """
    na = a.norm().clamp_min(1e-30).double()
    nb = b.norm().clamp_min(1e-30).double()
    c = (nb / na).sqrt().float()
    c = torch.where(torch.isfinite(c), c, torch.ones_like(c))
    return a * c, b / c.clamp_min(1e-30)


def bilateral_newton_schulz(
    A: torch.Tensor,
    B: torch.Tensor,
    ns_substeps: int = 2,
    eps: float = 1e-8,
    ortho_dtype: torch.dtype = torch.bfloat16,
    fold: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Mutual Newton-Schulz: orthogonalize A and B against each other.

    Each substep applies the same r x r polynomial P = p(C) in the shared
    cross-Gramian C = (B B^T)(A^T A) to both sides, A <- A P and B <- P B,
    which is exactly one Newton-Schulz step on the full product A @ B (by
    p(XY)X = Xp(YX)). The fixed point is G_A G_B = I -- the factors are
    mutually balanced, not individually orthonormal -- and the composed
    product has a flat spectrum (all nonzero singular values 1). Optional
    AOL-Gram row-sum folding of each factor normalizes the top of the
    spectrum and stabilizes the iteration (distorts the fixed point
    slightly; with fold=False the update equals the full-matrix NS exactly,
    and the polynomial is applied unscaled). When folding, C is scaled by
    its power-iterated largest eigenvalue before each polynomial so the
    spectrum lies inside the contraction region (a folded Gram product has
    no bounded row-sum guarantee).

    Args:
        A: left factor (m x r), B: right factor (r x n).
        ns_substeps: number of polynomial substeps, cycling the repo's
            pre-optimized cubic coefficient pairs (default: 2).
        eps: numerical stability constant for the folding.
        ortho_dtype: compute dtype for the r x r Gramian work.
        fold: apply AOL-Gram folding each substep (default: True).

    Returns:
        The mutually orthogonalized pair (A, B) in the input dtype.
    """
    dtype = A.dtype
    A = A.to(ortho_dtype)
    B = B.to(ortho_dtype)
    for i in range(ns_substeps):
        GA = A.mT @ A
        GB = B @ B.mT
        if fold:
            sA = GA.abs().sum(dim=-1).clamp_min_(eps).rsqrt()
            sB = GB.abs().sum(dim=-1).clamp_min_(eps).rsqrt()
            A = A * sA.unsqueeze(0)
            B = B * sB.unsqueeze(-1)
            GA = A.mT @ A
            GB = B @ B.mT
        C = GB @ GA
        apply_poly = True
        if fold:
            lam = _power_lambda_max(C)
            if lam > 1e-10 * C.abs().sum().clamp_min(1e-30):
                C = C / lam
            else:
                # degenerate cross-Gramian (momentum ~ 0 or disjoint row
                # spaces): nothing to contract -- keep the folded factors
                apply_poly = False
        if apply_poly:
            a, b, c = GRAM_NEWTON_SCHULZ_2STEP_COEFFS[i % 2]
            I = torch.eye(C.shape[0], dtype=C.dtype, device=C.device)
            P = a * I + b * C + c * (C @ C)
            A = A @ P
            B = P @ B
    return A.to(dtype), B.to(dtype)


def _compose_tail(
    A: torch.Tensor,
    B: torch.Tensor,
    rank: int,
    eps: float,
    ortho_dtype: torch.dtype,
    ns_substeps: int,
    rescale_ortho: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Mutual NS -> compose -> scale -> rebalance. Pure.

    Returns (U, V, delta) with delta = U @ V rescaled by a single scalar:
    with rescale_ortho=True the update is rescaled back to the
    pre-orthogonalization product norm ||A @ B||_F (norm-preserving Muon:
    orthogonalize the direction, restore the magnitude; the norm comes from
    the r x r Gramians, ||A B||_F^2 = trace((A^T A)(B B^T)), so the m x n
    product is never materialized twice). With rescale_ortho=False the
    update is normalized to unit Frobenius energy (flat unit spectrum, all
    nonzero singular values ~1) and lr carries the magnitude. Degenerate
    products (~0) are left unscaled either way. The returned pair (U, V) is
    norm-rebalanced (||U||_F == ||V||_F) -- the pair-symmetric scaling
    leaves the composed product and the cross-Gramian unchanged, but pins
    the factor balance so persistent (momentumized or warm-started) factors
    cannot drift apart and ill-condition their ALS solves.
    """
    pre_norm = torch.trace((A.mT @ A) @ (B @ B.mT)).clamp_min(0.0).double().sqrt().float()
    u_hat, v_hat = bilateral_newton_schulz(
        A, B, ns_substeps, eps, ortho_dtype
    )
    u_hat, v_hat = _rebalance_pair(u_hat, v_hat)
    delta = u_hat @ v_hat
    denom = delta.norm()
    if denom > 1e-10 * (u_hat.norm() * v_hat.norm()).clamp_min(1e-30):
        if rescale_ortho:
            delta = delta * (pre_norm / denom.clamp_min(1e-30))
        else:
            delta = delta * (math.sqrt(rank) / denom.clamp_min(1e-30))
    return (
        torch.where(torch.isfinite(u_hat), u_hat, torch.zeros_like(u_hat)),
        torch.where(torch.isfinite(v_hat), v_hat, torch.zeros_like(v_hat)),
        torch.where(torch.isfinite(delta), delta, torch.zeros_like(delta)),
    )


def _bilatmuon_full_svd_core(
    M: torch.Tensor,
    g: torch.Tensor,
    poly: torch.Tensor,
    rank: int,
    p: float,
    q: float,
    quantize_grad: bool,
    eps: float,
    ortho_dtype: torch.dtype,
    ns_substeps: int,
    rescale_ortho: bool,
    step: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Full mode, SVD: momentumize M, decompose it, mutually orthogonalize,
    compose. Returns (M, A, B, delta) with (A, B) the balanced pair."""
    if quantize_grad:
        g = _stoch_round_fp32(g)
    M = M * poly + (1.0 - poly) * g
    M = torch.where(torch.isfinite(M), M, g)
    A, B = _svd_split(M, rank, p, q)
    u_hat, v_hat, delta = _compose_tail(
        A, B, rank, eps, ortho_dtype, ns_substeps, rescale_ortho,
    )
    return M, u_hat, v_hat, delta


def _bilatmuon_full_als_core(
    M: torch.Tensor,
    g: torch.Tensor,
    w_A: torch.Tensor,
    w_B: torch.Tensor,
    poly: torch.Tensor,
    rank: int,
    p: float,
    q: float,
    quantize_grad: bool,
    eps: float,
    ortho_dtype: torch.dtype,
    ns_substeps: int,
    rescale_ortho: bool,
    als_eps: float,
    step: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Full mode, ALS: momentumize M, fit the factors with one SVD-free
    warm-started half-sweep, mutually orthogonalize, compose."""
    if quantize_grad:
        g = _stoch_round_fp32(g)
    M = M * poly + (1.0 - poly) * g
    M = torch.where(torch.isfinite(M), M, g)
    w_A = torch.where(torch.isfinite(w_A), w_A, torch.zeros_like(w_A))
    w_B = torch.where(torch.isfinite(w_B), w_B, torch.zeros_like(w_B))
    if M.abs().sum() > 1e-6:
        A = _als_solve(w_B @ w_B.mT, (M @ w_B.mT).mT, als_eps).mT
        B = _als_solve(A.mT @ A, A.mT @ M, als_eps)
        A, B = _canonicalize(A, B)
        A, B = _rebalance_pair(A, B)
        u_hat, v_hat, delta = _compose_tail(
            A, B, rank, eps, ortho_dtype, ns_substeps, rescale_ortho,
        )
        return M, u_hat, v_hat, delta
    # Dead momentum (gradient ~ 0 long enough for the EMA to decay): the
    # solves would amplify solver noise into the subnormal range and
    # poison the warm starts (a NaN factor collapses the next solve with
    # "input matrix is singular"). Pin the factors and emit the decayed
    # momentum; the cautious flip zeroes the update for a zero gradient.
    return M, w_A, w_B, M


def _bilatmuon_factored_svd_core(
    m_A: torch.Tensor,
    m_B: torch.Tensor,
    g: torch.Tensor,
    poly_A: torch.Tensor,
    poly_B: torch.Tensor,
    rank: int,
    p: float,
    q: float,
    quantize_grad: bool,
    eps: float,
    ortho_dtype: torch.dtype,
    ns_substeps: int,
    rescale_ortho: bool,
    step: torch.Tensor,
    als_eps: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Factored mode, SVD: decompose the gradient, momentumize each factor
    with its own rate, mutually orthogonalize, compose."""
    if quantize_grad:
        g = _stoch_round_fp32(g)
    a, b = _svd_split(g, rank, p, q, canonicalize=True)
    m_A = m_A * poly_A + (1.0 - poly_A) * a
    m_B = m_B * poly_B + (1.0 - poly_B) * b
    m_A = torch.where(torch.isfinite(m_A), m_A, a)
    m_B = torch.where(torch.isfinite(m_B), m_B, b)
    m_A, m_B = _rebalance_pair(m_A, m_B)
    _, _, delta = _compose_tail(
        m_A, m_B, rank, eps, ortho_dtype, ns_substeps, rescale_ortho,
    )
    return m_A, m_B, delta


def _bilatmuon_full_rsvd_core(
    M: torch.Tensor,
    g: torch.Tensor,
    omega: torch.Tensor,
    poly: torch.Tensor,
    rank: int,
    p: float,
    q: float,
    quantize_grad: bool,
    eps: float,
    ortho_dtype: torch.dtype,
    ns_substeps: int,
    rescale_ortho: bool,
    step: torch.Tensor,
    oversample: int,
    power: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Full mode, randomized SVD: momentumize M, decompose it with the
    randomized range finder, mutually orthogonalize, compose."""
    if quantize_grad:
        g = _stoch_round_fp32(g)
    M = M * poly + (1.0 - poly) * g
    M = torch.where(torch.isfinite(M), M, g)
    A, B = _rsvd_split(M, rank, p, q, omega, oversample, power)
    u_hat, v_hat, delta = _compose_tail(
        A, B, rank, eps, ortho_dtype, ns_substeps, rescale_ortho,
    )
    return M, u_hat, v_hat, delta


def _bilatmuon_factored_rsvd_core(
    m_A: torch.Tensor,
    m_B: torch.Tensor,
    g: torch.Tensor,
    omega: torch.Tensor,
    poly_A: torch.Tensor,
    poly_B: torch.Tensor,
    rank: int,
    p: float,
    q: float,
    quantize_grad: bool,
    eps: float,
    ortho_dtype: torch.dtype,
    ns_substeps: int,
    rescale_ortho: bool,
    step: torch.Tensor,
    oversample: int,
    power: int,
    als_eps: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Factored mode, randomized SVD: decompose the gradient, momentumize
    each factor with its own rate, mutually orthogonalize, compose."""
    if quantize_grad:
        g = _stoch_round_fp32(g)
    a, b = _rsvd_split(g, rank, p, q, omega, oversample, power,
                       canonicalize=True)
    m_A = m_A * poly_A + (1.0 - poly_A) * a
    m_B = m_B * poly_B + (1.0 - poly_B) * b
    m_A = torch.where(torch.isfinite(m_A), m_A, a)
    m_B = torch.where(torch.isfinite(m_B), m_B, b)
    m_A, m_B = _rebalance_pair(m_A, m_B)
    _, _, delta = _compose_tail(
        m_A, m_B, rank, eps, ortho_dtype, ns_substeps, rescale_ortho,
    )
    return m_A, m_B, delta


def _bilatmuon_factored_als_core(
    m_A: torch.Tensor,
    m_B: torch.Tensor,
    g: torch.Tensor,
    poly_A: torch.Tensor,
    poly_B: torch.Tensor,
    rank: int,
    p: float,
    q: float,
    quantize_grad: bool,
    eps: float,
    ortho_dtype: torch.dtype,
    ns_substeps: int,
    rescale_ortho: bool,
    step: torch.Tensor,
    als_eps: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Factored mode, ALS: fit the gradient with an SVD-free half-sweep
    against the momentum factors, momentumize, mutually orthogonalize."""
    if quantize_grad:
        g = _stoch_round_fp32(g)
    m_A = torch.where(torch.isfinite(m_A), m_A, torch.zeros_like(m_A))
    m_B = torch.where(torch.isfinite(m_B), m_B, torch.zeros_like(m_B))
    a = _als_solve(m_B @ m_B.mT, (g @ m_B.mT).mT, als_eps).mT
    b = _als_solve(a.mT @ a, a.mT @ g, als_eps)
    a, b = _canonicalize(a, b)
    m_A = m_A * poly_A + (1.0 - poly_A) * a
    m_B = m_B * poly_B + (1.0 - poly_B) * b
    m_A = torch.where(torch.isfinite(m_A), m_A, a)
    m_B = torch.where(torch.isfinite(m_B), m_B, b)
    m_A, m_B = _rebalance_pair(m_A, m_B)
    _, _, delta = _compose_tail(
        m_A, m_B, rank, eps, ortho_dtype, ns_substeps, rescale_ortho,
    )
    return m_A, m_B, delta


def _bilatmuon_plain_core(
    m: torch.Tensor,
    g: torch.Tensor,
    poly: torch.Tensor,
    quantize_grad: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """1D/0D fallback: plain poly-beta momentum (the update is the momentum)."""
    if quantize_grad:
        g = _stoch_round_fp32(g)
    m = m * poly + (1.0 - poly) * g
    m = torch.where(torch.isfinite(m), m, g)
    return m, m


_RAW_CORES = {
    ("full", "svd"): _bilatmuon_full_svd_core,
    ("full", "rsvd"): _bilatmuon_full_rsvd_core,
    ("full", "als"): _bilatmuon_full_als_core,
    ("factored", "svd"): _bilatmuon_factored_svd_core,
    ("factored", "rsvd"): _bilatmuon_factored_rsvd_core,
    ("factored", "als"): _bilatmuon_factored_als_core,
}


def _run_core(
    core: callable,
    key: tuple,
    *args,
) -> tuple:
    """Run a (possibly compiled) core, retrying eagerly on SVD convergence
    failures. The compiled graph's gesdd cannot catch its own LAPACK
    failure; the raw eager core re-runs the SVD through _robust_svd's
    fp64/scipy fallbacks."""
    try:
        return core(*args)
    except torch.linalg.LinAlgError:
        return _RAW_CORES[key](*args)


class BilatMuon(Optimizer):
    r"""BilatMuon: mutual Newton-Schulz over two momentumized factors.

    Args:
        params: iterable of parameters to optimize.
        lr (float): learning rate (default: 1e-2). The update is the
            mutually orthogonalized composition with a flat unit spectrum
            (all nonzero singular values ~1), so its RMS stays near O(1)
            regardless of gradient scale; the step size lives entirely in
            lr (unless ``rescale_ortho`` restores the pre-orthogonalization
            norm) -- treat lr as the tuning dial.
        mode (str): "full" (default) -- a full m x n momentum buffer is the
            state and the momentum itself is decomposed fresh each step;
            "factored" -- the momentum lives in two factor buffers
            (FacMuon-style, one poly-beta rate per factor).
        betas (Tuple[float, float]): poly-beta momentum rates. Full mode
            uses betas[0] for the momentum buffer; factored mode uses one
            rate per factor (default: (0.9, 0.95)). Debiased (poly = 1 at
            step 1, so the first step is a pure factorized-gradient step).
        weight_decay (float): decoupled weight decay (default: 0.0).
            Applied cautiously (sign-flip variant) by default.
        rank (int, optional): rank of the factorization (default: None =
            full rank min(m, n)). Smaller rank = less state memory (dial).
        factorization (str): "svd" (default, exact per-step SVD), "rsvd"
            (randomized SVD — the same (p, q) split and canonicalization
            at O(m n k) instead of O(m n min(m, n)); used when the rank
            dial r < min(m, n), where it is 10-100x cheaper, and routed to
            the exact SVD at full rank), or "als" (cheap SVD-free
            half-sweep; step 1 still uses the SVD/rsvd to seed the
            factors).
        rsvd_oversample (int): oversampling for the randomized SVD range
            finder (default: 8).
        rsvd_power (int): power iterations for the randomized SVD range
            finder (default: 1).
        split (Tuple[float, float]): asymmetric singular-value split
            (p, q) with p + q = 1 (default: (0.5, 0.5)): A = U Sigma^p and
            B = Sigma^q V^T, so one factor carries more of the magnitude
            and the other more of the geometry.
        ns_substeps (int): Newton-Schulz substeps for the mutual
            orthogonalization, cycling the repo's pre-optimized cubic
            coefficient pairs (default: 2).
        rescale_ortho (bool): rescale the composed update back to the
            pre-orthogonalization product norm ||A @ B||_F (norm-preserving
            Muon: orthogonalize the direction, restore the magnitude). With
            False (default) the update is normalized to unit Frobenius
            energy and lr carries the magnitude.
        cautious_update (bool): flip the sign of every update component
            that disagrees with the raw gradient instead of masking it
            (default: True).
        cautious_wd (bool): apply weight decay with a flipped sign wherever
            the decay would fight the update (default: True).
        stochastic_bf16 (bool): store the momentum/factor state in bf16
            with stochastic rounding on write (default: True). fp32 master
            weights are kept.
        quantize_grad (bool): stochastically round the raw gradient to bf16
            precision before factorization (default: False; unbiased,
            halves the factorization cost).
        precondition (str): gradient preconditioner applied before the
            momentum/factorization (default: "none"). "sinkhorn" -- RMS
            Sinkhorn row/col balancing (alternating row/col RMS
            normalization toward 1, ``sinkhorn_iters`` iterations; falls
            back to "rms" when it cannot be applied: 1D/0D tensors or
            degenerate row/column marginals). "rms" -- divide the whole
            gradient by its RMS (scale-only).
        sinkhorn_iters (int): iterations for the "sinkhorn" preconditioner
            (default: 5).
        als_eps (float): ridge regularization for the ALS solves
            (default: 1e-6). Relative to the gram trace (scale-aware).
        eps (float): stability constant for the Newton-Schulz folding
            (default: 1e-8).
        ortho_dtype (torch.dtype): compute dtype for the mutual
            orthogonalization (default: torch.bfloat16).
        compile_step (bool): torch.compile (max-autotune) the per-mode core
            functions, cached per shape, with eager fallback (default: False).

    State per 2D+ parameter: full mode = one momentum buffer (m x n, bf16 by
    default) plus a step counter (+ two r(m+n) ALS warm-start factors when
    factorization="als"); factored mode = exactly two momentum factor tensors
    (r(m + n) elements total, bf16 by default). 1D/0D parameters keep a
    single momentum buffer.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-2,
        mode: str = "full",
        betas: Tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.0,
        rank: Optional[int] = None,
        factorization: str = "als",
        rsvd_oversample: int = 8,
        rsvd_power: int = 1,
        split: Tuple[float, float] = (0.5, 0.5),
        ns_substeps: int = 2,
        rescale_ortho: bool = True,
        cautious_update: bool = True,
        cautious_wd: bool = True,
        stochastic_bf16: bool = True,
        quantize_grad: bool = False,
        precondition: str = "rms",
        sinkhorn_iters: int = 3,
        als_eps: float = 1e-6,
        eps: float = 1e-8,
        ortho_dtype: torch.dtype = torch.bfloat16,
        compile_step: bool = True,
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid lr: {lr}")
        if mode not in ("full", "factored"):
            raise ValueError(f"Invalid mode: {mode}")
        if factorization not in ("svd", "rsvd", "als"):
            raise ValueError(f"Invalid factorization: {factorization}")
        if rsvd_oversample < 1 or rsvd_power < 0:
            raise ValueError(
                f"Invalid rsvd_oversample/rsvd_power: {rsvd_oversample}, {rsvd_power}")
        if not (0.0 <= betas[0] < 1.0 and 0.0 <= betas[1] < 1.0):
            raise ValueError(f"Invalid betas: {betas}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        if rank is not None and rank < 1:
            raise ValueError(f"Invalid rank: {rank}")
        if len(split) != 2 or not (0.0 <= split[0] <= 1.0
                                   and 0.0 <= split[1] <= 1.0
                                   and abs(split[0] + split[1] - 1.0) < 1e-6):
            raise ValueError(f"Invalid split: {split} (need p, q with p+q=1)")
        if ns_substeps < 1:
            raise ValueError(f"Invalid ns_substeps: {ns_substeps}")
        if precondition not in ("none", "sinkhorn", "rms"):
            raise ValueError(f"Invalid precondition: {precondition}")
        if sinkhorn_iters < 1:
            raise ValueError(f"Invalid sinkhorn_iters: {sinkhorn_iters}")
        defaults = dict(
            lr=lr,
            mode=mode,
            betas=betas,
            weight_decay=weight_decay,
            rank=rank,
            factorization=factorization,
            rsvd_oversample=rsvd_oversample,
            rsvd_power=rsvd_power,
            split=split,
            ns_substeps=ns_substeps,
            rescale_ortho=rescale_ortho,
            cautious_update=cautious_update,
            cautious_wd=cautious_wd,
            stochastic_bf16=stochastic_bf16,
            quantize_grad=quantize_grad,
            precondition=precondition,
            sinkhorn_iters=sinkhorn_iters,
            als_eps=als_eps,
            eps=eps,
            ortho_dtype=ortho_dtype,
            compile_step=compile_step,
        )
        super().__init__(params, defaults)
        self._compiled: Dict[tuple, callable] = {}
        self._omegas: Dict[tuple, torch.Tensor] = {}

    def _init_state(self, p: torch.Tensor, group: dict) -> None:
        state = self.state[p]
        state["step"] = 0
        dtype = torch.bfloat16 if group["stochastic_bf16"] else torch.float32
        m, n = _reshape_to_2d(p.detach()).shape
        if p.ndim <= 1:
            state["m"] = torch.zeros((1, n), dtype=dtype, device=p.device)
        else:
            rank = group["rank"]
            r = min(m, n) if rank is None else min(rank, m, n)
            r = max(r, 1)
            if group["mode"] == "full":
                state["M"] = torch.zeros((m, n), dtype=dtype, device=p.device)
                if group["factorization"] == "als":
                    state["w_A"] = torch.zeros((m, r), dtype=dtype,
                                               device=p.device)
                    state["w_B"] = torch.zeros((r, n), dtype=dtype,
                                               device=p.device)
            else:
                state["m_A"] = torch.zeros((m, r), dtype=dtype, device=p.device)
                state["m_B"] = torch.zeros((r, n), dtype=dtype, device=p.device)

    def _get_core(self, key: tuple):
        core = self._compiled.get(key)
        if core is None:
            fn_key = key[:2] if key[0] != "plain" else ("plain",)
            fn = {
                ("full", "svd"): _bilatmuon_full_svd_core,
                ("full", "rsvd"): _bilatmuon_full_rsvd_core,
                ("full", "als"): _bilatmuon_full_als_core,
                ("factored", "svd"): _bilatmuon_factored_svd_core,
                ("factored", "rsvd"): _bilatmuon_factored_rsvd_core,
                ("factored", "als"): _bilatmuon_factored_als_core,
                ("plain",): _bilatmuon_plain_core,
            }[fn_key]
            try:
                torch.set_float32_matmul_precision('high')
                core = torch.compile(fn, mode="default")
            except Exception:
                core = fn
            self._compiled[key] = core
        return core

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            wd = group["weight_decay"]
            betas = group["betas"]
            mode = group["mode"]
            rank_opt = group["rank"]
            factorization = group["factorization"]
            rsvd_oversample = group["rsvd_oversample"]
            rsvd_power = group["rsvd_power"]
            p_s, q_s = group["split"]
            ns_substeps = group["ns_substeps"]
            rescale_ortho = group["rescale_ortho"]
            cautious_update = group["cautious_update"]
            cautious_wd = group["cautious_wd"]
            stochastic_bf16 = group["stochastic_bf16"]
            quantize_grad = group["quantize_grad"]
            precondition = group["precondition"]
            sinkhorn_iters = group["sinkhorn_iters"]
            als_eps = group["als_eps"]
            eps = group["eps"]
            ortho_dtype = group["ortho_dtype"]
            compile_step = group["compile_step"]
            state_dtype = torch.bfloat16 if stochastic_bf16 else torch.float32
            static_key = (quantize_grad, eps, ortho_dtype, als_eps,
                          ns_substeps, rescale_ortho, p_s, q_s,
                          rsvd_oversample, rsvd_power)
            for p in group["params"]:
                g = p.grad
                if g is None:
                    continue
                if g.is_sparse:
                    raise RuntimeError("BilatMuon does not support sparse gradients")
                if not g.is_contiguous():
                    g = g.contiguous()
                state = self.state[p]
                if len(state) == 0:
                    self._init_state(p, group)
                state["step"] += 1
                step_t = state["step"]
                g2 = _reshape_to_2d(g)
                g2 = _precondition_grad(g2, precondition, sinkhorn_iters, eps)
                if not torch.isfinite(g2).all():
                    continue
                if p.ndim <= 1:
                    poly = torch.full((), _poly_beta(betas[0], step_t),
                                      dtype=torch.float32, device=p.device)
                    if compile_step:
                        core = self._get_core(
                            ("plain", ("plain", g2.numel()) + static_key)
                        )
                    else:
                        core = _bilatmuon_plain_core
                    m_new, delta = core(state["m"].float(), g2, poly,
                                        quantize_grad)
                    if state_dtype == torch.bfloat16:
                        copy_stochastic_(state["m"], m_new)
                    else:
                        state["m"].copy_(m_new)
                else:
                    m, n = g2.shape
                    r = min(m, n) if rank_opt is None else min(rank_opt, m, n)
                    r = max(r, 1)
                    r_full = min(m, n)
                    # rsvd wins when the rank dial r < min(m, n) (10-100x
                    # cheaper than the exact SVD); at full rank the exact
                    # SVD is used. ALS seeds step 1 from the cheaper
                    # approximate factorization.
                    use_svd = factorization == "svd"
                    use_rsvd = False
                    if factorization == "rsvd":
                        use_rsvd = r < r_full
                        use_svd = not use_rsvd
                    elif factorization == "als" and step_t == 1:
                        use_rsvd = r < r_full
                        use_svd = not use_rsvd
                    key = (mode,
                           "rsvd" if use_rsvd else ("svd" if use_svd else "als"))
                    step_t_t = torch.tensor(step_t, dtype=torch.int64,
                                            device=p.device)
                    omega = None
                    if use_rsvd:
                        k = min(r + rsvd_oversample, m, n)
                        omega = self._omegas.get((n, k))
                        if omega is None:
                            gen = torch.Generator(device=p.device).manual_seed(0)
                            omega = torch.randn(n, k, dtype=torch.float32,
                                                device=p.device, generator=gen)
                            self._omegas[(n, k)] = omega
                    if compile_step:
                        core = self._get_core(
                            key + (m, n, r) + static_key
                        )
                    else:
                        core = _RAW_CORES[key]
                    if mode == "full":
                        poly = torch.full((), _poly_beta(betas[0], step_t),
                                          dtype=torch.float32, device=p.device)
                        if use_svd:
                            M_new, w_A_new, w_B_new, delta = _run_core(
                                core, key,
                                state["M"].float(), g2, poly, r, p_s, q_s,
                                quantize_grad, eps, ortho_dtype, ns_substeps,
                                rescale_ortho, step_t_t,
                            )
                        elif use_rsvd:
                            M_new, w_A_new, w_B_new, delta = _run_core(
                                core, key,
                                state["M"].float(), g2, omega, poly, r, p_s,
                                q_s, quantize_grad, eps, ortho_dtype,
                                ns_substeps, rescale_ortho, step_t_t,
                                rsvd_oversample, rsvd_power,
                            )
                        else:
                            M_new, w_A_new, w_B_new, delta = _run_core(
                                core, key,
                                state["M"].float(), g2,
                                state["w_A"].float(), state["w_B"].float(),
                                poly, r, p_s, q_s, quantize_grad, eps,
                                ortho_dtype, ns_substeps, rescale_ortho,
                                als_eps, step_t_t,
                            )
                        if state_dtype == torch.bfloat16:
                            copy_stochastic_(state["M"], M_new)
                        else:
                            state["M"].copy_(M_new)
                        if factorization == "als":
                            if state_dtype == torch.bfloat16:
                                copy_stochastic_(state["w_A"], w_A_new)
                                copy_stochastic_(state["w_B"], w_B_new)
                            else:
                                state["w_A"].copy_(w_A_new)
                                state["w_B"].copy_(w_B_new)
                    else:
                        poly_A = torch.full((), _poly_beta(betas[0], step_t),
                                            dtype=torch.float32,
                                            device=p.device)
                        poly_B = torch.full((), _poly_beta(betas[1], step_t),
                                            dtype=torch.float32,
                                            device=p.device)
                        if use_svd:
                            m_A_new, m_B_new, delta = _run_core(
                                core, key,
                                state["m_A"].float(), state["m_B"].float(),
                                g2, poly_A, poly_B, r, p_s, q_s,
                                quantize_grad, eps, ortho_dtype, ns_substeps,
                                rescale_ortho, step_t_t, als_eps,
                            )
                        elif use_rsvd:
                            m_A_new, m_B_new, delta = _run_core(
                                core, key,
                                state["m_A"].float(), state["m_B"].float(),
                                g2, omega, poly_A, poly_B, r, p_s, q_s,
                                quantize_grad, eps, ortho_dtype, ns_substeps,
                                rescale_ortho, step_t_t, rsvd_oversample,
                                rsvd_power, als_eps,
                            )
                        else:
                            m_A_new, m_B_new, delta = _run_core(
                                core, key,
                                state["m_A"].float(), state["m_B"].float(),
                                g2, poly_A, poly_B, r, p_s, q_s,
                                quantize_grad, eps, ortho_dtype, ns_substeps,
                                rescale_ortho, step_t_t, als_eps,
                            )
                        if state_dtype == torch.bfloat16:
                            copy_stochastic_(state["m_A"], m_A_new)
                            copy_stochastic_(state["m_B"], m_B_new)
                        else:
                            state["m_A"].copy_(m_A_new)
                            state["m_B"].copy_(m_B_new)
                delta = delta.reshape_as(p)
                delta = torch.where(torch.isfinite(delta), delta,
                                    torch.zeros_like(delta))
                if cautious_update:
                    delta = g.sign() * delta.abs()
                if wd != 0.0:
                    if cautious_wd:
                        s = torch.where(p.sign() == delta.sign(),
                                        torch.ones_like(delta),
                                        -torch.ones_like(delta))
                        decay = wd * s * p
                    else:
                        decay = wd * p
                    p.add_(delta + decay, alpha=-lr)
                else:
                    p.add_(delta, alpha=-lr)
        return loss
