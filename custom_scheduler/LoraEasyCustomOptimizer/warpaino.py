# Source: https://github.com/Clybius/Personalized-Optimizers

"""WarpAINO: AINOOpt with the WarpAdam learnable low-rank gradient warp.

Based on AINOOpt, with the WarpAdam (arXiv:2409.04244) learnable distortion
matrix P = I + U @ V^T inserted *after* Sinkhorn normalization, before the
sign-momentum / innovation machinery:

    g_norm   = Sinkhorn(g)                    (RMS row/col norm for 1D)
    g_warped = g_norm + U @ (V^T @ g_norm)    (P = I + U @ V^T)

All subsequent AINOOpt machinery -- tracked sign momentum, CAME-style
factorized row & column squared innovation tracking, tracked value momentum,
Gram Newton-Schulz orthogonalization, row-wise variance normalization,
tensor-wise RMS rescale, cautious updates & cautious weight decay,
stochastic rounding for BF16, foreach and torch.compile support -- operates
on the warped gradient.

P is trained per-parameter with the WarpAdam online meta-objective (no
closure, one closure-free pass per step): after each step, U and V take one
SGD step on the one-step-ahead gradient prediction loss

    ||P @ g_norm_{t-1} - g_norm_t||^2 / ||g_norm_{t-1}||^2,

i.e. P learns the transition structure of the *normalized* gradient process
(the paper's "transfer off-diagonal information"). The norm normalization
makes the meta-dynamics scale-invariant (stable for meta_lr * ||V||^2 < 2,
independent of the gradient magnitude); because the warp input is
Sinkhorn-normalized (unit row/col RMS), the meta-dynamics are additionally
bounded per layer. meta_wd damps U and V back toward 0, keeping P near the
identity anchor.

U is zero-initialized so P = I exactly and WarpAINO is bitwise-identical to
AINOOpt until the factors learn (the first step is exactly AINOOpt); V is
random-initialized to break the stationary point of the meta-objective at
the identity anchor (with U = V = 0 the meta-gradients vanish and P can
never learn). rank=0 or meta_lr=0 disables the warp entirely (plain
AINOOpt).

Factors are shaped (m, r) with r = min(rank, m), m = the mixing dimension
(number of rows after 2D reshape for >= 2D tensors, the full size for
1D/0D tensors) -- matching WarpAdam's mixing convention.

Hyperparameters added over AINOOpt:
    rank (int): Rank r of the P factors U, V (capped at m).
        rank=0 disables the warp entirely (plain AINOOpt). (default: 16)
    meta_lr (float): Learning rate for the U, V meta-update.
        meta_lr=0 disables the warp (plain AINOOpt). (default: 1e-3)
    meta_wd (float): Damping on U and V toward 0 (identity anchor).
        (default: 1e-2)
"""

import torch
from torch.optim import Optimizer
from typing import Tuple, List, Union, Iterable

GRAM_NEWTON_SCHULZ_2STEP_COEFFS1 = [
    (1.4897216394163149, -0.5798724169434551, 0.0831346315615072),
    (2.0181598271548000, -1.5523232773433393, 0.5343894201774000),
]


def copy_stochastic_(target: torch.Tensor, source: torch.Tensor) -> None:
    """Stochastically round a float32 source into bfloat16."""
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
        return t.reshape(len(t), -1)
    if t.ndim < 2:
        return t.reshape(1, -1)
    return t


@torch.no_grad()
def gram_newton_schulz_2step(
    M: torch.Tensor,
    eps: float = 1e-7,
    ortho_dtype=torch.bfloat16,
) -> torch.Tensor:
    """2-step Gram Newton-Schulz with pre-optimized unconstrained coefficients."""
    X = M.to(ortho_dtype)
    transposed = False
    if X.size(0) > X.size(1):
        X = X.T
        transposed = True

    # AOL-Gram folding
    A = X @ X.mT
    rescaling = A.abs().sum(dim=-1).clamp_min_(eps)
    s = rescaling.rsqrt().unsqueeze(-1)
    X = X * s
    R = s * A * s.mT

    n, m = X.shape
    I = torch.eye(n, dtype=X.dtype, device=X.device)
    Q = I

    # Apply pre-optimized coefficients
    for a, b, c in GRAM_NEWTON_SCHULZ_2STEP_COEFFS1:
        R2 = R @ R
        z = a * I + b * R + c * R2
        Q = Q @ z
        R = z @ R @ z

    out = Q @ X

    if transposed:
        out = out.T

    return out.to(M.dtype)


@torch.no_grad()
def _warp_meta_update(
    state: dict,
    g_cur_2d: torch.Tensor,
    meta_lr: float,
    meta_wd: float,
) -> None:
    """One SGD step of U, V on the one-step-ahead prediction loss
    ||P @ g_{t-1} - g_t||^2 / ||g_{t-1}||^2 (normalized gradients, in 2D
    warp form). Also stores the current gradient as the new previous one."""
    U = state["U"]
    V = state["V"]
    prev = state.get("prev_g")
    if prev is not None:
        R = (prev + U @ (V.t() @ prev)) - g_cur_2d
        scale = 1.0 / prev.norm().square().clamp_min_(1e-12)
        U.add_(R @ (prev.t() @ V), alpha=-meta_lr * scale)
        V.add_(prev @ (R.t() @ U), alpha=-meta_lr * scale)
        if meta_wd > 0:
            U.mul_(1.0 - meta_lr * meta_wd)
            V.mul_(1.0 - meta_lr * meta_wd)
    state["prev_g"] = g_cur_2d.clone()


class WarpAINO(Optimizer):
    """Innovation Noise-conditioned Optimizer (WarpAINO) with CAME Factorization,
    Sign Momentum & the WarpAdam learnable low-rank gradient warp.

    Every Sinkhorn-normalized gradient is linearly warped by a learnable
    low-rank, identity-anchored distortion matrix P = I + U @ V^T before it
    enters the sign-momentum / innovation machinery:

        g_warped = g_norm + U @ (V^T @ g_norm),   U = 0, V ~ N(0, 1)

    U is zero-initialized so P = I exactly and WarpAINO is exactly AINOOpt
    until the factors learn; V is random-initialized to break the stationary
    point of the meta-objective at the identity anchor. After each step, U
    and V are updated with one SGD step (no closure required) on the
    one-step-ahead prediction loss of the normalized gradient process
    ||P @ g_{t-1} - g_t||^2 / ||g_{t-1}||^2; meta_wd damps them back toward
    0. rank=0 or meta_lr=0 disables the warp (plain AINOOpt).
    """

    def __init__(
        self,
        params,
        lr: float = 0.0001,
        betas: Tuple[float, float, float] = (0.95, 0.95, 0.999),
        weight_decay: float = 0.0,
        cautious_update: bool = True,
        cautious_wd: bool = True,
        stochastic_fp: bool = True,
        compile_step: bool = False,
        foreach: bool = False,
        sinkhorn_steps: int = 5,
        ortho_dtype: torch.dtype = torch.bfloat16,
        eps: float = 1e-16,
        rank: int = 16,
        meta_lr: float = 1e-3,
        meta_wd: float = 1e-2,
        **kwargs,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if len(betas) != 3:
            raise ValueError(f"betas must be a 3-tuple of floats, got {betas}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0 (beta1): {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1 (beta2): {betas[1]}")
        if not 0.0 <= betas[2] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 2 (beta3): {betas[2]}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if eps < 0.0:
            raise ValueError(f"Invalid eps value: {eps}")
        if not isinstance(rank, int) or rank < 0:
            raise ValueError(f"Invalid rank: {rank}")
        if meta_lr < 0.0:
            raise ValueError(f"Invalid meta_lr value: {meta_lr}")
        if meta_wd < 0.0:
            raise ValueError(f"Invalid meta_wd value: {meta_wd}")

        defaults = dict(
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
            cautious_update=cautious_update,
            cautious_wd=cautious_wd,
            stochastic_fp=stochastic_fp,
            sinkhorn_steps=sinkhorn_steps,
            ortho_dtype=ortho_dtype,
            eps=eps,
            rank=rank,
            meta_lr=meta_lr,
            meta_wd=meta_wd,
        )
        super().__init__(params, defaults)

        self._compile_step = compile_step
        self._foreach = foreach

        if self._compile_step:
            try:
                torch._dynamo.config.recompile_limit = max(
                    torch._dynamo.config.recompile_limit, 64
                )
                self._compiled_step_2d = torch.compile(
                    self._aino_step_core_2d,
                    fullgraph=True,
                    dynamic=False,
                )
                self._compiled_step_1d = torch.compile(
                    self._aino_step_core_1d,
                    fullgraph=True,
                    dynamic=False,
                )
                self._compiled_warp_step_2d = torch.compile(
                    self._warpaino_step_core_2d,
                    fullgraph=True,
                    dynamic=False,
                )
                self._compiled_warp_step_1d = torch.compile(
                    self._warpaino_step_core_1d,
                    fullgraph=True,
                    dynamic=False,
                )
            except Exception as e:
                import logging
                logging.warning(
                    f"torch.compile failed to initialize: {e}. Falling back to uncompiled step."
                )
                self._compiled_step_2d = self._aino_step_core_2d
                self._compiled_step_1d = self._aino_step_core_1d
                self._compiled_warp_step_2d = self._warpaino_step_core_2d
                self._compiled_warp_step_1d = self._warpaino_step_core_1d
        else:
            self._compiled_step_2d = self._aino_step_core_2d
            self._compiled_step_1d = self._aino_step_core_1d
            self._compiled_warp_step_2d = self._warpaino_step_core_2d
            self._compiled_warp_step_1d = self._warpaino_step_core_1d

    @staticmethod
    def _aino_step_core_2d(
        p_2d: torch.Tensor,
        g_2d: torch.Tensor,
        momentum: torch.Tensor,
        sign_momentum: torch.Tensor,
        exp_avg_sq_row: torch.Tensor,
        exp_avg_sq_col: torch.Tensor,
        row_var: torch.Tensor,
        beta1: float,
        beta2: float,
        beta3: float,
        weight_decay: float,
        eps: float,
        step_t: torch.Tensor,
        sinkhorn_steps: int,
        cautious_update: bool,
        cautious_wd: bool,
        ortho_dtype: torch.dtype,
    ) -> torch.Tensor:
        """Core 2D+ update step compilable by torch.compile (plain AINOOpt,
        used when the warp is disabled)."""
        # 1. Sinkhorn normalization for 2D+ parameters
        g_norm = g_2d
        for _ in range(sinkhorn_steps):
            row_rms = torch.sqrt(g_norm.pow(2).mean(dim=-1, keepdim=True) + eps)
            g_norm = g_norm / row_rms
            col_rms = torch.sqrt(g_norm.pow(2).mean(dim=-2, keepdim=True) + eps)
            g_norm = g_norm / col_rms

        # Compute poly-beta values for beta1, beta2, and beta3
        beta1_pow = beta1 ** step_t
        poly_beta1 = torch.where(
            step_t > 1.0,
            (beta1_pow - beta1) / (beta1_pow - 1.0),
            torch.zeros_like(beta1_pow),
        )

        beta2_pow = beta2 ** step_t
        poly_beta2 = torch.where(
            step_t > 1.0,
            (beta2_pow - beta2) / (beta2_pow - 1.0),
            torch.zeros_like(beta2_pow),
        )

        beta3_pow = beta3 ** step_t
        poly_beta3 = torch.where(
            step_t > 1.0,
            (beta3_pow - beta3) / (beta3_pow - 1.0),
            torch.zeros_like(beta3_pow),
        )

        # 2. Track gradient sign momentum using beta1
        g_sign = g_norm.sign()
        sign_momentum.lerp_(g_sign, 1.0 - beta1)

        # 3. Track CAME-style factorized row & column squared innovation using poly_beta2
        diff_sq = (g_norm - momentum).pow(2)
        exp_avg_sq_row.lerp_(diff_sq.mean(dim=-1, keepdim=True), 1.0 - poly_beta2)
        exp_avg_sq_col.lerp_(diff_sq.mean(dim=-2, keepdim=True), 1.0 - poly_beta2)

        # Construct factorized denominator
        r_factor = (exp_avg_sq_row + eps).sqrt()
        c_factor = ((exp_avg_sq_col + eps) / (exp_avg_sq_col.mean(dim=-1, keepdim=True) + eps)).sqrt()
        denom = r_factor * c_factor

        # 4. Update tracked value momentum using poly_beta1
        momentum.lerp_(g_norm, 1.0 - poly_beta1)

        # 5. Craft update: apply tracked sign to absolute value of (momentum / denom)
        update = sign_momentum * (momentum.abs() / denom)

        # 6. Orthogonalize update using 2-step Gram Newton-Schulz
        O = gram_newton_schulz_2step(update, eps=1e-7, ortho_dtype=ortho_dtype)

        # 7. Track row-wise variance using poly_beta3 & normalize using updated tracked row-wise variance
        row_sq = O.pow(2).mean(dim=-1)
        row_var.lerp_(row_sq, 1.0 - poly_beta3)
        O_norm = O / torch.sqrt(row_var.clamp_min(eps).unsqueeze(-1))

        # 8. Rescale back to update's RMS norm
        target_rms = torch.sqrt(update.pow(2).mean() + eps)
        current_rms = torch.sqrt(O_norm.pow(2).mean() + eps)
        update_final = O_norm * (target_rms / (current_rms + eps))

        # 9. Cautious updates
        if cautious_update:
            mask = (g_2d * update_final > 0).to(update_final.dtype)
            mask_mean = mask.mean().clamp_min(1e-3)
            update_final = update_final * mask / mask_mean

        # Decoupled weight decay
        if weight_decay != 0:
            if cautious_wd:
                wd_mask = (update_final.sign() == p_2d.sign()).to(update_final.dtype)
                update_final = update_final + weight_decay * p_2d * wd_mask
            else:
                update_final = update_final + weight_decay * p_2d

        return update_final

    @staticmethod
    def _aino_step_core_1d(
        p_data: torch.Tensor,
        g_data: torch.Tensor,
        momentum: torch.Tensor,
        sign_momentum: torch.Tensor,
        exp_avg_sq: torch.Tensor,
        row_var: torch.Tensor,
        beta1: float,
        beta2: float,
        beta3: float,
        weight_decay: float,
        eps: float,
        step_t: torch.Tensor,
        cautious_update: bool,
        cautious_wd: bool,
    ) -> torch.Tensor:
        """Core 1D/0D update step compilable by torch.compile (plain AINOOpt,
        used when the warp is disabled)."""
        # 1. RMS normalize gradient for 1D/0D parameters
        grad_rms = torch.sqrt(g_data.pow(2).mean() + eps)
        g_norm = g_data / grad_rms

        # Compute poly-beta values for beta1, beta2 and beta3
        beta1_pow = beta1 ** step_t
        poly_beta1 = torch.where(
            step_t > 1.0,
            (beta1_pow - beta1) / (beta1_pow - 1.0),
            torch.zeros_like(beta1_pow),
        )

        beta2_pow = beta2 ** step_t
        poly_beta2 = torch.where(
            step_t > 1.0,
            (beta2_pow - beta2) / (beta2_pow - 1.0),
            torch.zeros_like(beta2_pow),
        )

        beta3_pow = beta3 ** step_t
        poly_beta3 = torch.where(
            step_t > 1.0,
            (beta3_pow - beta3) / (beta3_pow - 1.0),
            torch.zeros_like(beta3_pow),
        )

        # 2. Track gradient sign momentum using beta1
        g_sign = g_norm.sign()
        sign_momentum.lerp_(g_sign, 1.0 - beta1)

        # 3. Track squared innovation relative to un-updated momentum using poly_beta2
        diff_sq = (g_norm - momentum).pow(2)
        exp_avg_sq.lerp_(diff_sq, 1.0 - poly_beta2)
        denom = exp_avg_sq.sqrt().clamp_min_(eps)

        # 4. Update tracked value momentum using poly_beta1
        momentum.lerp_(g_norm, 1.0 - poly_beta1)

        # 5. Craft update: apply tracked sign to absolute value of (momentum / denom)
        update = sign_momentum * (momentum.abs() / denom)

        # 6. Skip orthogonalization for 1D/0D parameters
        O = update

        # 7. Track variance using poly_beta3 & normalize
        var_sq = O.pow(2).mean()
        row_var.lerp_(var_sq, 1.0 - poly_beta3)
        O_norm = O / torch.sqrt(row_var.clamp_min(eps))

        # 8. Rescale back to update's RMS norm
        target_rms = torch.sqrt(update.pow(2).mean() + eps)
        current_rms = torch.sqrt(O_norm.pow(2).mean() + eps)
        update_final = O_norm * (target_rms / (current_rms + eps))

        # 9. Cautious updates
        if cautious_update:
            mask = (g_data * update_final > 0).to(update_final.dtype)
            mask_mean = mask.mean().clamp_min(1e-3)
            update_final = update_final * mask / mask_mean

        # Decoupled weight decay
        if weight_decay != 0:
            if cautious_wd:
                wd_mask = (update_final.sign() == p_data.sign()).to(update_final.dtype)
                update_final = update_final + weight_decay * p_data * wd_mask
            else:
                update_final = update_final + weight_decay * p_data

        return update_final

    @staticmethod
    def _warpaino_step_core_2d(
        p_2d: torch.Tensor,
        g_2d: torch.Tensor,
        U: torch.Tensor,
        V: torch.Tensor,
        momentum: torch.Tensor,
        sign_momentum: torch.Tensor,
        exp_avg_sq_row: torch.Tensor,
        exp_avg_sq_col: torch.Tensor,
        row_var: torch.Tensor,
        beta1: float,
        beta2: float,
        beta3: float,
        weight_decay: float,
        eps: float,
        step_t: torch.Tensor,
        sinkhorn_steps: int,
        cautious_update: bool,
        cautious_wd: bool,
        ortho_dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Core 2D+ update step compilable by torch.compile, with the WarpAdam
        warp P = I + U @ V^T applied to the Sinkhorn-normalized gradient.
        Returns (update_final, g_norm)."""
        # 1. Sinkhorn normalization for 2D+ parameters
        g_norm = g_2d
        for _ in range(sinkhorn_steps):
            row_rms = torch.sqrt(g_norm.pow(2).mean(dim=-1, keepdim=True) + eps)
            g_norm = g_norm / row_rms
            col_rms = torch.sqrt(g_norm.pow(2).mean(dim=-2, keepdim=True) + eps)
            g_norm = g_norm / col_rms

        # 1b. Warp: g -> P @ g with P = I + U @ V^T (mixes the row dimension)
        g_warped = g_norm + U @ (V.t() @ g_norm)

        # Compute poly-beta values for beta1, beta2, and beta3
        beta1_pow = beta1 ** step_t
        poly_beta1 = torch.where(
            step_t > 1.0,
            (beta1_pow - beta1) / (beta1_pow - 1.0),
            torch.zeros_like(beta1_pow),
        )

        beta2_pow = beta2 ** step_t
        poly_beta2 = torch.where(
            step_t > 1.0,
            (beta2_pow - beta2) / (beta2_pow - 1.0),
            torch.zeros_like(beta2_pow),
        )

        beta3_pow = beta3 ** step_t
        poly_beta3 = torch.where(
            step_t > 1.0,
            (beta3_pow - beta3) / (beta3_pow - 1.0),
            torch.zeros_like(beta3_pow),
        )

        # 2. Track gradient sign momentum using beta1 (on the warped gradient)
        g_sign = g_warped.sign()
        sign_momentum.lerp_(g_sign, 1.0 - beta1)

        # 3. Track CAME-style factorized row & column squared innovation using poly_beta2
        diff_sq = (g_warped - momentum).pow(2)
        exp_avg_sq_row.lerp_(diff_sq.mean(dim=-1, keepdim=True), 1.0 - poly_beta2)
        exp_avg_sq_col.lerp_(diff_sq.mean(dim=-2, keepdim=True), 1.0 - poly_beta2)

        # Construct factorized denominator
        r_factor = (exp_avg_sq_row + eps).sqrt()
        c_factor = ((exp_avg_sq_col + eps) / (exp_avg_sq_col.mean(dim=-1, keepdim=True) + eps)).sqrt()
        denom = r_factor * c_factor

        # 4. Update tracked value momentum using poly_beta1 (on the warped gradient)
        momentum.lerp_(g_warped, 1.0 - poly_beta1)

        # 5. Craft update: apply tracked sign to absolute value of (momentum / denom)
        update = sign_momentum * (momentum.abs() / denom)

        # 6. Orthogonalize update using 2-step Gram Newton-Schulz
        O = gram_newton_schulz_2step(update, eps=eps, ortho_dtype=ortho_dtype)

        # 7. Track row-wise variance using poly_beta3 & normalize using updated tracked row-wise variance
        row_sq = O.pow(2).mean(dim=-1)
        row_var.lerp_(row_sq, 1.0 - poly_beta3)
        O_norm = O / torch.sqrt(row_var.clamp_min(eps).unsqueeze(-1))

        # 8. Rescale back to update's RMS norm
        target_rms = torch.sqrt(update.pow(2).mean() + eps)
        current_rms = torch.sqrt(O_norm.pow(2).mean() + eps)
        update_final = O_norm * (target_rms / (current_rms + eps))

        # 9. Cautious updates (masked against the warped gradient)
        if cautious_update:
            mask = (g_warped * update_final > 0).to(update_final.dtype)
            mask_mean = mask.mean().clamp_min(1e-3)
            update_final = update_final * mask / mask_mean

        # Decoupled weight decay
        if weight_decay != 0:
            if cautious_wd:
                wd_mask = (update_final.sign() == p_2d.sign()).to(update_final.dtype)
                update_final = update_final + weight_decay * p_2d * wd_mask
            else:
                update_final = update_final + weight_decay * p_2d

        return update_final, g_norm

    @staticmethod
    def _warpaino_step_core_1d(
        p_data: torch.Tensor,
        g_data: torch.Tensor,
        U: torch.Tensor,
        V: torch.Tensor,
        momentum: torch.Tensor,
        sign_momentum: torch.Tensor,
        exp_avg_sq: torch.Tensor,
        row_var: torch.Tensor,
        beta1: float,
        beta2: float,
        beta3: float,
        weight_decay: float,
        eps: float,
        step_t: torch.Tensor,
        cautious_update: bool,
        cautious_wd: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Core 1D/0D update step compilable by torch.compile, with the WarpAdam
        warp P = I + U @ V^T applied across the full vector (mixing dim =
        numel). Returns (update_final, g_norm)."""
        # 1. RMS normalize gradient for 1D/0D parameters
        grad_rms = torch.sqrt(g_data.pow(2).mean() + eps)
        g_norm = g_data / grad_rms

        # 1b. Warp across the full vector: U, V in R^{m x r}, m = numel
        g2d = g_norm.reshape(-1, 1)
        g_warped = (g2d + U @ (V.t() @ g2d)).reshape(g_norm.shape)

        # Compute poly-beta values for beta1, beta2 and beta3
        beta1_pow = beta1 ** step_t
        poly_beta1 = torch.where(
            step_t > 1.0,
            (beta1_pow - beta1) / (beta1_pow - 1.0),
            torch.zeros_like(beta1_pow),
        )

        beta2_pow = beta2 ** step_t
        poly_beta2 = torch.where(
            step_t > 1.0,
            (beta2_pow - beta2) / (beta2_pow - 1.0),
            torch.zeros_like(beta2_pow),
        )

        beta3_pow = beta3 ** step_t
        poly_beta3 = torch.where(
            step_t > 1.0,
            (beta3_pow - beta3) / (beta3_pow - 1.0),
            torch.zeros_like(beta3_pow),
        )

        # 2. Track gradient sign momentum using beta1 (on the warped gradient)
        g_sign = g_warped.sign()
        sign_momentum.lerp_(g_sign, 1.0 - beta1)

        # 3. Track squared innovation relative to un-updated momentum using poly_beta2
        diff_sq = (g_warped - momentum).pow(2)
        exp_avg_sq.lerp_(diff_sq, 1.0 - poly_beta2)
        denom = exp_avg_sq.sqrt().clamp_min_(eps)

        # 4. Update tracked value momentum using poly_beta1 (on the warped gradient)
        momentum.lerp_(g_warped, 1.0 - poly_beta1)

        # 5. Craft update: apply tracked sign to absolute value of (momentum / denom)
        update = sign_momentum * (momentum.abs() / denom)

        # 6. Skip orthogonalization for 1D/0D parameters
        O = update

        # 7. Track variance using poly_beta3 & normalize
        var_sq = O.pow(2).mean()
        row_var.lerp_(var_sq, 1.0 - poly_beta3)
        O_norm = O / torch.sqrt(row_var.clamp_min(eps))

        # 8. Rescale back to update's RMS norm
        target_rms = torch.sqrt(update.pow(2).mean() + eps)
        current_rms = torch.sqrt(O_norm.pow(2).mean() + eps)
        update_final = O_norm * (target_rms / (current_rms + eps))

        # 9. Cautious updates (masked against the warped gradient)
        if cautious_update:
            mask = (g_warped * update_final > 0).to(update_final.dtype)
            mask_mean = mask.mean().clamp_min(1e-3)
            update_final = update_final * mask / mask_mean

        # Decoupled weight decay
        if weight_decay != 0:
            if cautious_wd:
                wd_mask = (update_final.sign() == p_data.sign()).to(update_final.dtype)
                update_final = update_final + weight_decay * p_data * wd_mask
            else:
                update_final = update_final + weight_decay * p_data

        return update_final, g_norm

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2, beta3 = group["betas"]
            weight_decay = group["weight_decay"]
            cautious_update = group["cautious_update"]
            cautious_wd = group["cautious_wd"]
            stochastic_fp = group["stochastic_fp"]
            sinkhorn_steps = group["sinkhorn_steps"]
            ortho_dtype = group["ortho_dtype"]
            eps = group["eps"]

            # Initialize states if needed
            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError(
                        "WarpAINO does not support sparse gradients"
                    )
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["step_t"] = torch.tensor(0.0, dtype=torch.float32, device=p.device)
                    if p.ndim >= 2:
                        w_2d = _reshape_to_2d(p.data)
                        m, n = w_2d.shape
                        state["momentum"] = torch.zeros_like(w_2d, dtype=torch.float32)
                        state["sign_momentum"] = torch.zeros_like(w_2d, dtype=torch.float32)
                        state["exp_avg_sq_row"] = torch.zeros((m, 1), device=p.device, dtype=torch.float32)
                        state["exp_avg_sq_col"] = torch.zeros((1, n), device=p.device, dtype=torch.float32)
                        state["row_var"] = torch.zeros(m, device=p.device, dtype=torch.float32)
                        warp_m = m
                    else:
                        state["momentum"] = torch.zeros_like(p.data, dtype=torch.float32)
                        state["sign_momentum"] = torch.zeros_like(p.data, dtype=torch.float32)
                        state["exp_avg_sq"] = torch.zeros_like(p.data, dtype=torch.float32)
                        state["row_var"] = torch.zeros((), device=p.device, dtype=torch.float32)
                        warp_m = p.numel()
                    if group["rank"] > 0 and group["meta_lr"] > 0:
                        r = min(group["rank"], warp_m)
                        state["U"] = torch.zeros(
                            warp_m, r, dtype=torch.float32, device=p.device
                        )
                        state["V"] = torch.zeros(
                            warp_m, r, dtype=torch.float32, device=p.device
                        ).normal_(0.0, 1.0)

            if self._foreach:
                self._step_foreach(group)
            else:
                self._step_native(group)

        return loss

    def _step_native(self, group):
        lr = group["lr"]
        beta1, beta2, beta3 = group["betas"]
        weight_decay = group["weight_decay"]
        cautious_update = group["cautious_update"]
        cautious_wd = group["cautious_wd"]
        stochastic_fp = group["stochastic_fp"]
        sinkhorn_steps = group["sinkhorn_steps"]
        ortho_dtype = group["ortho_dtype"]
        eps = group["eps"]
        meta_lr = group["meta_lr"]
        meta_wd = group["meta_wd"]

        for p in group["params"]:
            if p.grad is None:
                continue

            state = self.state[p]
            state["step"] += 1
            if "step_t" not in state:
                state["step_t"] = torch.tensor(float(state["step"]), dtype=torch.float32, device=p.device)
            else:
                state["step_t"].fill_(float(state["step"]))
            step_t = state["step_t"]

            grad = p.grad.data
            if grad.dtype in (torch.bfloat16, torch.float16):
                grad = grad.float()

            U = state.get("U")

            if p.ndim >= 2:
                p_2d = _reshape_to_2d(p.data)
                g_2d = _reshape_to_2d(grad)

                p_2d_fp32 = p_2d.float() if p.dtype is torch.bfloat16 else p_2d.clone()

                if U is not None:
                    update_final, g_norm = self._compiled_warp_step_2d(
                        p_2d_fp32,
                        g_2d,
                        U,
                        state["V"],
                        state["momentum"],
                        state["sign_momentum"],
                        state["exp_avg_sq_row"],
                        state["exp_avg_sq_col"],
                        state["row_var"],
                        beta1,
                        beta2,
                        beta3,
                        weight_decay,
                        eps,
                        step_t,
                        sinkhorn_steps,
                        cautious_update,
                        cautious_wd,
                        ortho_dtype,
                    )
                else:
                    update_final = self._compiled_step_2d(
                        p_2d_fp32,
                        g_2d,
                        state["momentum"],
                        state["sign_momentum"],
                        state["exp_avg_sq_row"],
                        state["exp_avg_sq_col"],
                        state["row_var"],
                        beta1,
                        beta2,
                        beta3,
                        weight_decay,
                        eps,
                        step_t,
                        sinkhorn_steps,
                        cautious_update,
                        cautious_wd,
                        ortho_dtype,
                    )

                p_2d_fp32.add_(update_final, alpha=-lr)

                if p.dtype is torch.bfloat16 and stochastic_fp:
                    copy_stochastic_(p.data, p_2d_fp32.view_as(p.data))
                else:
                    p.data.copy_(p_2d_fp32.view_as(p.data))

                if U is not None:
                    _warp_meta_update(state, g_norm, meta_lr, meta_wd)

            else:
                p_fp32 = p.data.float() if p.dtype is torch.bfloat16 else p.data.clone()

                if U is not None:
                    update_final, g_norm = self._compiled_warp_step_1d(
                        p_fp32,
                        grad,
                        U,
                        state["V"],
                        state["momentum"],
                        state["sign_momentum"],
                        state["exp_avg_sq"],
                        state["row_var"],
                        beta1,
                        beta2,
                        beta3,
                        weight_decay,
                        eps,
                        step_t,
                        cautious_update,
                        cautious_wd,
                    )
                else:
                    update_final = self._compiled_step_1d(
                        p_fp32,
                        grad,
                        state["momentum"],
                        state["sign_momentum"],
                        state["exp_avg_sq"],
                        state["row_var"],
                        beta1,
                        beta2,
                        beta3,
                        weight_decay,
                        eps,
                        step_t,
                        cautious_update,
                        cautious_wd,
                    )

                p_fp32.add_(update_final, alpha=-lr)

                if p.dtype is torch.bfloat16 and stochastic_fp:
                    copy_stochastic_(p.data, p_fp32)
                else:
                    p.data.copy_(p_fp32)

                if U is not None:
                    _warp_meta_update(state, g_norm.reshape(-1, 1), meta_lr, meta_wd)

    def _step_foreach(self, group):
        lr = group["lr"]
        beta1, beta2, beta3 = group["betas"]
        weight_decay = group["weight_decay"]
        cautious_update = group["cautious_update"]
        cautious_wd = group["cautious_wd"]
        stochastic_fp = group["stochastic_fp"]
        sinkhorn_steps = group["sinkhorn_steps"]
        ortho_dtype = group["ortho_dtype"]
        eps = group["eps"]
        meta_lr = group["meta_lr"]
        meta_wd = group["meta_wd"]

        params_2d = []
        params_1d = []

        for p in group["params"]:
            if p.grad is None:
                continue
            if p.ndim >= 2:
                params_2d.append(p)
            else:
                params_1d.append(p)

        # 1. Process 1D fallback parameters
        for p in params_1d:
            state = self.state[p]
            state["step"] += 1
            if "step_t" not in state:
                state["step_t"] = torch.tensor(float(state["step"]), dtype=torch.float32, device=p.device)
            else:
                state["step_t"].fill_(float(state["step"]))
            step_t = state["step_t"]

            grad = p.grad.data.float() if p.grad.data.dtype in (torch.bfloat16, torch.float16) else p.grad.data

            p_fp32 = p.data.float() if p.dtype is torch.bfloat16 else p.data.clone()
            U = state.get("U")
            if U is not None:
                update_final, g_norm = self._compiled_warp_step_1d(
                    p_fp32,
                    grad,
                    U,
                    state["V"],
                    state["momentum"],
                    state["sign_momentum"],
                    state["exp_avg_sq"],
                    state["row_var"],
                    beta1,
                    beta2,
                    beta3,
                    weight_decay,
                    eps,
                    step_t,
                    cautious_update,
                    cautious_wd,
                )
            else:
                update_final = self._compiled_step_1d(
                    p_fp32,
                    grad,
                    state["momentum"],
                    state["sign_momentum"],
                    state["exp_avg_sq"],
                    state["row_var"],
                    beta1,
                    beta2,
                    beta3,
                    weight_decay,
                    eps,
                    step_t,
                    cautious_update,
                    cautious_wd,
                )
            p_fp32.add_(update_final, alpha=-lr)

            if p.dtype is torch.bfloat16 and stochastic_fp:
                copy_stochastic_(p.data, p_fp32)
            else:
                p.data.copy_(p_fp32)

            if U is not None:
                _warp_meta_update(state, g_norm.reshape(-1, 1), meta_lr, meta_wd)

        # 2. Process 2D+ parameters
        if not params_2d:
            return

        updates_final_list = []
        p_fp32_list = []
        p_original_list = []
        meta_list = []

        for p in params_2d:
            state = self.state[p]
            state["step"] += 1
            if "step_t" not in state:
                state["step_t"] = torch.tensor(float(state["step"]), dtype=torch.float32, device=p.device)
            else:
                state["step_t"].fill_(float(state["step"]))
            step_t = state["step_t"]

            grad = p.grad.data.float() if p.grad.data.dtype in (torch.bfloat16, torch.float16) else p.grad.data

            p_2d = _reshape_to_2d(p.data)
            g_2d = _reshape_to_2d(grad)

            p_2d_fp32 = p_2d.float() if p.dtype is torch.bfloat16 else p_2d.clone()

            U = state.get("U")
            if U is not None:
                update_final, g_norm = self._compiled_warp_step_2d(
                    p_2d_fp32,
                    g_2d,
                    U,
                    state["V"],
                    state["momentum"],
                    state["sign_momentum"],
                    state["exp_avg_sq_row"],
                    state["exp_avg_sq_col"],
                    state["row_var"],
                    beta1,
                    beta2,
                    beta3,
                    weight_decay,
                    eps,
                    step_t,
                    sinkhorn_steps,
                    cautious_update,
                    cautious_wd,
                    ortho_dtype,
                )
                meta_list.append((state, g_norm))
            else:
                update_final = self._compiled_step_2d(
                    p_2d_fp32,
                    g_2d,
                    state["momentum"],
                    state["sign_momentum"],
                    state["exp_avg_sq_row"],
                    state["exp_avg_sq_col"],
                    state["row_var"],
                    beta1,
                    beta2,
                    beta3,
                    weight_decay,
                    eps,
                    step_t,
                    sinkhorn_steps,
                    cautious_update,
                    cautious_wd,
                    ortho_dtype,
                )

            updates_final_list.append(update_final.view_as(p_2d_fp32))
            p_fp32_list.append(p_2d_fp32)
            p_original_list.append(p)

        torch._foreach_add_(p_fp32_list, updates_final_list, alpha=-lr)

        for p, p_fp32 in zip(p_original_list, p_fp32_list):
            p_reshaped = p_fp32.view_as(p.data)
            if p.dtype is torch.bfloat16 and stochastic_fp:
                copy_stochastic_(p.data, p_reshaped)
            else:
                p.data.copy_(p_reshaped)

        for state, g_norm in meta_list:
            _warp_meta_update(state, g_norm, meta_lr, meta_wd)
