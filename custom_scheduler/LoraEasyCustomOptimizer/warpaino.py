# Source: https://github.com/Clybius/Personalized-Optimizers

"""WarpAINO: AINOOpt with learnable dense or spectral update warps.

Based on AINOOpt, with the WarpAdam (arXiv:2409.04244) learnable distortion
matrix P = I + D inserted after the crafted update, immediately before
Gram Newton-Schulz orthogonalization:

    g_norm        = Sinkhorn(g)               (RMS row/col norm for 1D)
    update        = AINO(update machinery)
    update_warped = update + D @ update

The tracked sign momentum, CAME-style factorized row & column squared
innovation tracking, and tracked value momentum operate on the unwarped
normalized gradient. Gram Newton-Schulz orthogonalization and the remaining
update processing operate on the warped crafted update.

P is trained per-parameter with the WarpAdam online meta-objective (no
closure, one closure-free pass per step): after each step, D takes one SGD
step on the one-step-ahead prediction loss

    ||P @ update_{t-1} - update_t||^2 / ||update_{t-1}||^2,

i.e. P learns the transition structure of the crafted update process (the
paper's "transfer off-diagonal information") -- the same signal it is
applied to. The norm normalization makes the meta-dynamics scale-invariant
(independent of the update magnitude); meta_wd damps D back toward 0,
keeping P near the identity anchor.

D is zero-initialized so P = I exactly and WarpAINO is bitwise-identical to
AINOOpt until the dense warp learns (the first step is exactly AINOOpt).
meta_lr=0 disables the warp entirely (plain AINOOpt).

The dense residual is shaped (m, m), where m is the mixing dimension
(number of rows after 2D reshape for >= 2D tensors, the full size for
1D/0D tensors). ``warp_mode="spectral"`` instead uses a full-rank,
symmetric-circulant warp represented by bounded log-frequency scales and
applied with FFTs. Spectral 2D+ warps can independently mix the row and
column dimensions without storing dense matrices. Spectral log-scales are
kept in FP32; ``warp_dtype`` controls dense residual storage.

Hyperparameters added over AINOOpt:
    meta_lr (float): Learning rate for the dense residual meta-update.
        meta_lr=0 disables the warp (plain AINOOpt). (default: 5e-2)
    meta_wd (float): Damping on D toward 0 (identity anchor).
        (default: 1e-2)
    warp_mode (str): ``"dense"`` for the original dense residual or
        ``"spectral"`` for the FFT-based full-rank warp. (default: "dense")
    spectral_bilateral (bool): In spectral mode, independently warp both
        dimensions of 2D+ updates. Vector and scalar parameters always use
        the left/vector warp only. (default: True)
    spectral_log_bound (float): Absolute bound on spectral log-scales. Gains
        are constrained to ``[exp(-bound), exp(bound)]``. (default: 1.0)
    warp_dtype (torch.dtype): Storage dtype for D. FP16, BF16, and FP32 are
        supported; BF16 is the default.
    relative_wd (bool): If True, weight decay is RMS-relative: the p_data
        term is scaled to ``weight_decay`` times the RMS of the (cautious-
        masked) crafted update, i.e. ``wd = weight_decay * (rms_update /
        rms_param) * p`` so ``RMS(wd) = weight_decay * RMS(update)``.
        When False (default), uses absolute ``weight_decay * p``.
    relative_wd_delta (float): Pseudo-Huber smoothing threshold for relative
        weight decay denominator, i.e. ``rms_param_smooth = sqrt(rms_param^2 + delta^2)``.
        Prevents singular scaling and high-frequency sign chatter near zero by
        smoothly transitioning relative decay back to standard linear decay
        as ``rms_param << delta``. (default: 1e-3)
    relative_wd_max_contraction (float): Maximum fractional parameter shrinkage
        permitted in a single discrete step under relative weight decay. Caps the
        effective decay multiplier so that ``scale <= max_contraction / (lr * weight_decay)``,
        guaranteeing the discrete update factor ``(1 - lr * weight_decay * scale)``
        remains strictly non-negative and preventing sign-flipping overshoots.
        (default: 0.99)
"""

import torch
from torch.optim import Optimizer
from typing import Tuple, List, Union, Iterable, Optional

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


def _spectral_mirror(x: torch.Tensor) -> torch.Tensor:
    """Reflect a full FFT spectrum onto its negative-frequency indices."""
    return torch.roll(x.flip(0), shifts=1, dims=0)


def _spectral_log_gain(
    log_scale: torch.Tensor,
    log_bound: float,
) -> torch.Tensor:
    """Return real conjugate-symmetric positive gains for a FFT warp."""
    log_scale = 0.5 * (log_scale + _spectral_mirror(log_scale))
    return log_scale.clamp(-log_bound, log_bound).exp()


def _spectral_apply(
    update_2d: torch.Tensor,
    spectral_log_left: torch.Tensor,
    spectral_log_right: Optional[torch.Tensor],
    spectral_log_bound: float,
) -> torch.Tensor:
    """Apply a full-rank symmetric-circulant warp using orthonormal FFTs."""
    left_gain = _spectral_log_gain(spectral_log_left, spectral_log_bound)
    if spectral_log_right is None:
        spectrum = torch.fft.fft(update_2d, dim=0, norm="ortho")
        delta = torch.fft.ifft(
            spectrum * (left_gain[:, None] - 1.0), dim=0, norm="ortho"
        ).real
    else:
        right_gain = _spectral_log_gain(spectral_log_right, spectral_log_bound)
        spectrum = torch.fft.fft2(update_2d, dim=(0, 1), norm="ortho")
        gain_delta = left_gain[:, None] * right_gain[None, :] - 1.0
        delta = torch.fft.ifft2(
            spectrum * gain_delta, dim=(0, 1), norm="ortho"
        ).real
    return update_2d + delta


def _relative_wd_max_scale(
    lr: Union[float, torch.Tensor],
    weight_decay: Union[float, torch.Tensor],
    relative_wd_max_contraction: float,
) -> Union[float, torch.Tensor]:
    """Precompute the relative-WD overshoot cap outside compiled graphs.

    Returns ``relative_wd_max_contraction / (lr * weight_decay)`` when both
    ``lr`` and ``weight_decay`` are positive, otherwise ``+inf`` (no cap).
    When ``lr`` is a CUDA tensor, the calculation is performed on-device
    without blocking CPU-GPU transfers or pipeline stalls.
    """
    if isinstance(lr, torch.Tensor):
        if lr.device.type == "cpu":
            lr_val = lr.item()
            wd_val = float(weight_decay)
            lr_wd = lr_val * wd_val
            if lr_wd > 0:
                return float(relative_wd_max_contraction) / lr_wd
            return float("inf")
        else:
            lr_wd = lr * weight_decay
            return torch.where(
                lr_wd > 0,
                torch.as_tensor(
                    relative_wd_max_contraction,
                    device=lr.device,
                    dtype=torch.float32,
                )
                / lr_wd.clamp_min(1e-12),
                torch.as_tensor(float("inf"), device=lr.device, dtype=torch.float32),
            )
    else:
        lr_wd = float(lr) * float(weight_decay)
        if lr_wd > 0:
            return float(relative_wd_max_contraction) / lr_wd
        return float("inf")


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
    update_cur_2d: torch.Tensor,
    meta_lr: float,
    meta_wd: float,
) -> None:
    """Update the dense residual on the normalized prediction objective.

    The stored residual is D in P = I + D. It is kept in low precision to
    reduce optimizer-state memory, but the update is accumulated in FP32
    before being quantized back to the configured storage dtype.
    """
    warp = state["warp"]
    prev = state.get("prev_update")
    if prev is not None:
        warp_fp32 = warp.float()
        R = (prev + warp_fp32 @ prev) - update_cur_2d
        scale = 1.0 / prev.norm().square().clamp_min_(1e-12)
        R_scaled = R * (-meta_lr * scale)
        warp_fp32.addmm_(R_scaled, prev.t())
        if meta_wd > 0:
            warp_fp32.mul_(1.0 - meta_lr * meta_wd)
        warp.copy_(warp_fp32.to(warp.dtype))
    state["prev_update"] = update_cur_2d.clone()


@torch.no_grad()
def _spectral_meta_update(
    state: dict,
    update_cur_2d: torch.Tensor,
    meta_lr: float,
    meta_wd: float,
    spectral_log_bound: float,
) -> None:
    """Update FFT log-scales on the normalized prediction objective."""
    prev = state.get("prev_update")
    if prev is not None:
        log_left = state["spectral_log_left"]
        left_gain = _spectral_log_gain(log_left, spectral_log_bound)
        scale = 1.0 / prev.norm().square().clamp_min_(1e-12)

        log_right = state.get("spectral_log_right")
        if log_right is None or log_right.numel() == 0:
            prev_spectrum = torch.fft.fft(prev, dim=0, norm="ortho")
            current_spectrum = torch.fft.fft(update_cur_2d, dim=0, norm="ortho")
            prediction = left_gain[:, None] * prev_spectrum
            residual = prediction - current_spectrum
            grad_left = left_gain * torch.real(
                torch.conj(residual) * prev_spectrum
            ).sum(dim=1)
            grad_left.mul_(scale)
        else:
            right_gain = _spectral_log_gain(log_right, spectral_log_bound)
            prev_spectrum = torch.fft.fft2(prev, dim=(0, 1), norm="ortho")
            current_spectrum = torch.fft.fft2(
                update_cur_2d, dim=(0, 1), norm="ortho"
            )
            base_left = prev_spectrum * right_gain[None, :]
            prediction = left_gain[:, None] * base_left
            residual = prediction - current_spectrum
            grad_left = left_gain * torch.real(
                torch.conj(residual) * base_left
            ).sum(dim=1)
            base_right = prev_spectrum * left_gain[:, None]
            grad_right = right_gain * torch.real(
                torch.conj(residual) * base_right
            ).sum(dim=0)
            grad_left.mul_(scale)
            grad_right.mul_(scale)

        log_left.add_(grad_left, alpha=-meta_lr)
        if meta_wd > 0:
            log_left.mul_(1.0 - meta_lr * meta_wd)
        log_left.clamp_(-spectral_log_bound, spectral_log_bound)
        log_left.copy_(0.5 * (log_left + _spectral_mirror(log_left)))

        if log_right is not None and log_right.numel() > 0:
            log_right.add_(grad_right, alpha=-meta_lr)
            if meta_wd > 0:
                log_right.mul_(1.0 - meta_lr * meta_wd)
            log_right.clamp_(-spectral_log_bound, spectral_log_bound)
            log_right.copy_(0.5 * (log_right + _spectral_mirror(log_right)))

    state["prev_update"] = update_cur_2d.clone()


class WarpAINO(Optimizer):
    """AINO optimizer with a dense or spectral WarpAdam-style crafted-update warp.

    The Sinkhorn-normalized gradient first enters the ordinary AINO sign,
    innovation, and value-momentum machinery. The resulting crafted update is
    then linearly warped by an identity-anchored operator immediately before
    orthogonalization. The default dense mode uses:

        update_warped = update + D @ update,   D = 0

    Spectral mode uses bounded positive log-frequency scales and orthonormal
    FFTs. In 2D+ bilateral mode it applies independent symmetric-circulant
    operators on the row and column dimensions.

    D is zero-initialized so P = I exactly and WarpAINO is exactly AINOOpt
    until the dense residual learns. After each step, D is updated with one
    SGD step (no closure required) on the
    one-step-ahead prediction loss of the crafted update process
    ||P @ u_{t-1} - u_t||^2 / ||u_{t-1}||^2; meta_wd damps D back toward
    0. meta_lr=0 disables the warp (plain AINOOpt).

    The dense residual is stored in ``warp_dtype`` and promoted to FP32 for
    matrix multiplication and meta-updates.
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
        meta_lr: float = 5e-2,
        meta_wd: float = 1e-2,
        warp_dtype: torch.dtype = torch.bfloat16,
        warp_mode: str = "spectral",
        spectral_bilateral: bool = True,
        spectral_log_bound: float = 1.0,
        relative_wd: bool = False,
        relative_wd_delta: float = 1e-3,
        relative_wd_max_contraction: float = 0.99,
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
        if meta_lr < 0.0:
            raise ValueError(f"Invalid meta_lr value: {meta_lr}")
        if meta_wd < 0.0:
            raise ValueError(f"Invalid meta_wd value: {meta_wd}")
        if warp_mode not in ("dense", "spectral"):
            raise ValueError(
                f"Invalid warp_mode: {warp_mode}; expected 'dense' or 'spectral'"
            )
        if spectral_log_bound < 0.0:
            raise ValueError(
                f"Invalid spectral_log_bound value: {spectral_log_bound}"
            )
        if warp_dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise ValueError(
                "warp_dtype must be torch.float16, torch.bfloat16, or torch.float32, "
                f"got {warp_dtype}"
            )
        if not isinstance(relative_wd, bool):
            raise ValueError(f"relative_wd must be bool, got {relative_wd}")

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
            meta_lr=meta_lr,
            meta_wd=meta_wd,
            warp_dtype=warp_dtype,
            warp_mode=warp_mode,
            spectral_bilateral=spectral_bilateral,
            spectral_log_bound=spectral_log_bound,
            relative_wd=relative_wd,
            relative_wd_delta=relative_wd_delta,
            relative_wd_max_contraction=relative_wd_max_contraction,
        )
        super().__init__(params, defaults)

        self._compile_step = compile_step
        self._foreach = foreach

        if self._compile_step:
            import logging

            torch._dynamo.config.recompile_limit = max(
                torch._dynamo.config.recompile_limit, 64
            )
            try:
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
                    self._WarpAINO_step_core_2d,
                    fullgraph=True,
                    dynamic=False,
                )
                self._compiled_warp_step_1d = torch.compile(
                    self._WarpAINO_step_core_1d,
                    fullgraph=True,
                    dynamic=False,
                )
            except Exception as e:
                logging.warning(
                    f"torch.compile failed for AINO/dense cores: {e}. "
                    "Falling back to uncompiled cores."
                )
                self._compiled_step_2d = self._aino_step_core_2d
                self._compiled_step_1d = self._aino_step_core_1d
                self._compiled_warp_step_2d = self._WarpAINO_step_core_2d
                self._compiled_warp_step_1d = self._WarpAINO_step_core_1d
            if warp_mode == "spectral":
                try:
                    self._compiled_spectral_step_2d = torch.compile(
                        self._spectral_step_core_2d,
                        fullgraph=True,
                        dynamic=False,
                    )
                    self._compiled_spectral_step_1d = torch.compile(
                        self._spectral_step_core_1d,
                        fullgraph=True,
                        dynamic=False,
                    )
                except Exception as e:
                    logging.warning(
                        f"torch.compile failed for spectral cores: {e}. "
                        "Falling back to uncompiled spectral cores."
                    )
                    self._compiled_spectral_step_2d = self._spectral_step_core_2d
                    self._compiled_spectral_step_1d = self._spectral_step_core_1d
            else:
                self._compiled_spectral_step_2d = self._spectral_step_core_2d
                self._compiled_spectral_step_1d = self._spectral_step_core_1d
        else:
            self._compiled_step_2d = self._aino_step_core_2d
            self._compiled_step_1d = self._aino_step_core_1d
            self._compiled_warp_step_2d = self._WarpAINO_step_core_2d
            self._compiled_warp_step_1d = self._WarpAINO_step_core_1d
            self._compiled_spectral_step_2d = self._spectral_step_core_2d
            self._compiled_spectral_step_1d = self._spectral_step_core_1d

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
        max_scale: float,
        eps: float,
        step_t: torch.Tensor,
        sinkhorn_steps: int,
        cautious_update: bool,
        cautious_wd: bool,
        ortho_dtype: torch.dtype,
        relative_wd: bool = False,
        relative_wd_delta: float = 1e-3,
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

        # Decoupled weight decay (absolute or RMS-relative)
        if weight_decay != 0:
            if relative_wd:
                # 1. Compute RMS values
                update_rms = torch.sqrt(update_final.pow(2).mean() + eps)
                param_rms_smooth = torch.sqrt(
                    p_2d.pow(2).mean() + relative_wd_delta**2
                )
                # 2. Scale with Huber smoothing
                scale = update_rms / param_rms_smooth
                # 3. Guard against discrete time-step overshooting
                scale = torch.clamp_max(scale, max_scale)
                # 4. Apply weight decay
                if cautious_wd:
                    wd_mask = (update_final * p_2d >= 0).to(update_final.dtype)
                    update_final = update_final + weight_decay * scale * p_2d * wd_mask
                else:
                    update_final = update_final + weight_decay * scale * p_2d
            elif cautious_wd:
                wd_mask = (update_final * p_2d >= 0).to(update_final.dtype)
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
        max_scale: float,
        eps: float,
        step_t: torch.Tensor,
        cautious_update: bool,
        cautious_wd: bool,
        relative_wd: bool = False,
        relative_wd_delta: float = 1e-3,
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

        # Decoupled weight decay (absolute or RMS-relative)
        if weight_decay != 0:
            if relative_wd:
                # 1. Compute RMS values
                update_rms = torch.sqrt(update_final.pow(2).mean() + eps)
                param_rms_smooth = torch.sqrt(
                    p_data.pow(2).mean() + relative_wd_delta**2
                )
                # 2. Scale with Huber smoothing
                scale = update_rms / param_rms_smooth
                # 3. Guard against discrete time-step overshooting
                scale = torch.clamp_max(scale, max_scale)
                # 4. Apply weight decay
                if cautious_wd:
                    wd_mask = (update_final * p_data >= 0).to(update_final.dtype)
                    update_final = update_final + weight_decay * scale * p_data * wd_mask
                else:
                    update_final = update_final + weight_decay * scale * p_data
            elif cautious_wd:
                wd_mask = (update_final * p_data >= 0).to(update_final.dtype)
                update_final = update_final + weight_decay * p_data * wd_mask
            else:
                update_final = update_final + weight_decay * p_data

        return update_final

    @staticmethod
    def _WarpAINO_step_core_2d(
        p_2d: torch.Tensor,
        g_2d: torch.Tensor,
        warp: torch.Tensor,
        momentum: torch.Tensor,
        sign_momentum: torch.Tensor,
        exp_avg_sq_row: torch.Tensor,
        exp_avg_sq_col: torch.Tensor,
        row_var: torch.Tensor,
        beta1: float,
        beta2: float,
        beta3: float,
        weight_decay: float,
        max_scale: float,
        eps: float,
        step_t: torch.Tensor,
        sinkhorn_steps: int,
        cautious_update: bool,
        cautious_wd: bool,
        ortho_dtype: torch.dtype,
        relative_wd: bool = False,
        relative_wd_delta: float = 1e-3,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Core 2D+ step with P applied to the crafted update before NS."""
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

        # 6. Warp the crafted update before 2-step Gram Newton-Schulz
        update_warped = update + warp.float() @ update
        O = gram_newton_schulz_2step(update_warped, eps=1e-7, ortho_dtype=ortho_dtype)

        # 7. Track row-wise variance using poly_beta3 & normalize using updated tracked row-wise variance
        row_sq = O.pow(2).mean(dim=-1)
        row_var.lerp_(row_sq, 1.0 - poly_beta3)
        O_norm = O / torch.sqrt(row_var.clamp_min(eps).unsqueeze(-1))

        # 8. Rescale back to update's RMS norm
        target_rms = torch.sqrt(update_warped.pow(2).mean() + eps)
        current_rms = torch.sqrt(O_norm.pow(2).mean() + eps)
        update_final = O_norm * (target_rms / (current_rms + eps))

        # 9. Cautious updates (masked against the original gradient)
        if cautious_update:
            mask = (g_2d * update_final > 0).to(update_final.dtype)
            mask_mean = mask.mean().clamp_min(1e-3)
            update_final = update_final * mask / mask_mean

        # Decoupled weight decay (absolute or RMS-relative)
        if weight_decay != 0:
            if relative_wd:
                # 1. Compute RMS values
                update_rms = torch.sqrt(update_final.pow(2).mean() + eps)
                param_rms_smooth = torch.sqrt(
                    p_2d.pow(2).mean() + relative_wd_delta**2
                )
                # 2. Scale with Huber smoothing
                scale = update_rms / param_rms_smooth
                # 3. Guard against discrete time-step overshooting
                scale = torch.clamp_max(scale, max_scale)
                # 4. Apply weight decay
                if cautious_wd:
                    wd_mask = (update_final * p_2d >= 0).to(update_final.dtype)
                    update_final = update_final + weight_decay * scale * p_2d * wd_mask
                else:
                    update_final = update_final + weight_decay * scale * p_2d
            elif cautious_wd:
                wd_mask = (update_final * p_2d >= 0).to(update_final.dtype)
                update_final = update_final + weight_decay * p_2d * wd_mask
            else:
                update_final = update_final + weight_decay * p_2d

        return update_final, update

    @staticmethod
    def _WarpAINO_step_core_1d(
        p_data: torch.Tensor,
        g_data: torch.Tensor,
        warp: torch.Tensor,
        momentum: torch.Tensor,
        sign_momentum: torch.Tensor,
        exp_avg_sq: torch.Tensor,
        row_var: torch.Tensor,
        beta1: float,
        beta2: float,
        beta3: float,
        weight_decay: float,
        max_scale: float,
        eps: float,
        step_t: torch.Tensor,
        cautious_update: bool,
        cautious_wd: bool,
        relative_wd: bool = False,
        relative_wd_delta: float = 1e-3,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Core 1D/0D step with P applied to the crafted update before NS."""
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

        # 6. Warp the crafted update before the (skipped) orthogonalization
        update_2d = update.reshape(-1, 1)
        O = (update_2d + warp.float() @ update_2d).reshape(update.shape)

        # 7. Track variance using poly_beta3 & normalize
        var_sq = O.pow(2).mean()
        row_var.lerp_(var_sq, 1.0 - poly_beta3)
        O_norm = O / torch.sqrt(row_var.clamp_min(eps))

        # 8. Rescale back to update's RMS norm
        target_rms = torch.sqrt(O.pow(2).mean() + eps)
        current_rms = torch.sqrt(O_norm.pow(2).mean() + eps)
        update_final = O_norm * (target_rms / (current_rms + eps))

        # 9. Cautious updates (masked against the original gradient)
        if cautious_update:
            mask = (g_data * update_final > 0).to(update_final.dtype)
            mask_mean = mask.mean().clamp_min(1e-3)
            update_final = update_final * mask / mask_mean

        # Decoupled weight decay (absolute or RMS-relative)
        if weight_decay != 0:
            if relative_wd:
                # 1. Compute RMS values
                update_rms = torch.sqrt(update_final.pow(2).mean() + eps)
                param_rms_smooth = torch.sqrt(
                    p_data.pow(2).mean() + relative_wd_delta**2
                )
                # 2. Scale with Huber smoothing
                scale = update_rms / param_rms_smooth
                # 3. Guard against discrete time-step overshooting
                scale = torch.clamp_max(scale, max_scale)
                # 4. Apply weight decay
                if cautious_wd:
                    wd_mask = (update_final * p_data >= 0).to(update_final.dtype)
                    update_final = update_final + weight_decay * scale * p_data * wd_mask
                else:
                    update_final = update_final + weight_decay * scale * p_data
            elif cautious_wd:
                wd_mask = (update_final * p_data >= 0).to(update_final.dtype)
                update_final = update_final + weight_decay * p_data * wd_mask
            else:
                update_final = update_final + weight_decay * p_data

        return update_final, update

    @staticmethod
    def _spectral_step_core_2d(
        p_2d: torch.Tensor,
        g_2d: torch.Tensor,
        spectral_log_left: torch.Tensor,
        spectral_log_right: torch.Tensor,
        momentum: torch.Tensor,
        sign_momentum: torch.Tensor,
        exp_avg_sq_row: torch.Tensor,
        exp_avg_sq_col: torch.Tensor,
        row_var: torch.Tensor,
        beta1: float,
        beta2: float,
        beta3: float,
        weight_decay: float,
        max_scale: float,
        eps: float,
        step_t: torch.Tensor,
        sinkhorn_steps: int,
        cautious_update: bool,
        cautious_wd: bool,
        ortho_dtype: torch.dtype,
        spectral_bilateral: bool,
        spectral_log_bound: float,
        relative_wd: bool = False,
        relative_wd_delta: float = 1e-3,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Core 2D+ step with a full-rank FFT spectral warp."""
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

        # 3. Track factorized squared innovation using poly_beta2
        diff_sq = (g_norm - momentum).pow(2)
        exp_avg_sq_row.lerp_(diff_sq.mean(dim=-1, keepdim=True), 1.0 - poly_beta2)
        exp_avg_sq_col.lerp_(diff_sq.mean(dim=-2, keepdim=True), 1.0 - poly_beta2)

        r_factor = (exp_avg_sq_row + eps).sqrt()
        c_factor = (
            (exp_avg_sq_col + eps)
            / (exp_avg_sq_col.mean(dim=-1, keepdim=True) + eps)
        ).sqrt()
        denom = r_factor * c_factor

        # 4. Update tracked value momentum using poly_beta1
        momentum.lerp_(g_norm, 1.0 - poly_beta1)

        # 5. Craft update
        update = sign_momentum * (momentum.abs() / denom)

        # 6. Apply the full-rank spectral warp before Gram Newton-Schulz.
        if spectral_bilateral and spectral_log_right.numel() > 0:
            warped = _spectral_apply(
                update,
                spectral_log_left,
                spectral_log_right,
                spectral_log_bound,
            )
        else:
            warped = _spectral_apply(
                update, spectral_log_left, None, spectral_log_bound
            )
        update_warped = torch.where(step_t <= 1.0, update, warped)
        O = gram_newton_schulz_2step(
            update_warped, eps=1e-7, ortho_dtype=ortho_dtype
        )

        # 7. Track row-wise variance
        row_sq = O.pow(2).mean(dim=-1)
        row_var.lerp_(row_sq, 1.0 - poly_beta3)
        O_norm = O / torch.sqrt(row_var.clamp_min(eps).unsqueeze(-1))

        # 8. Rescale back to the warped update's RMS norm
        target_rms = torch.sqrt(update_warped.pow(2).mean() + eps)
        current_rms = torch.sqrt(O_norm.pow(2).mean() + eps)
        update_final = O_norm * (target_rms / (current_rms + eps))

        # 9. Cautious updates against the original gradient
        if cautious_update:
            mask = (g_2d * update_final > 0).to(update_final.dtype)
            mask_mean = mask.mean().clamp_min(1e-3)
            update_final = update_final * mask / mask_mean

        if weight_decay != 0:
            if relative_wd:
                # 1. Compute RMS values
                update_rms = torch.sqrt(update_final.pow(2).mean() + eps)
                param_rms_smooth = torch.sqrt(
                    p_2d.pow(2).mean() + relative_wd_delta**2
                )
                # 2. Scale with Huber smoothing
                scale = update_rms / param_rms_smooth
                # 3. Guard against discrete time-step overshooting
                scale = torch.clamp_max(scale, max_scale)
                # 4. Apply weight decay
                if cautious_wd:
                    wd_mask = (update_final * p_2d >= 0).to(update_final.dtype)
                    update_final = update_final + weight_decay * scale * p_2d * wd_mask
                else:
                    update_final = update_final + weight_decay * scale * p_2d
            elif cautious_wd:
                wd_mask = (update_final * p_2d >= 0).to(update_final.dtype)
                update_final = update_final + weight_decay * p_2d * wd_mask
            else:
                update_final = update_final + weight_decay * p_2d

        return update_final, update

    @staticmethod
    def _spectral_step_core_1d(
        p_data: torch.Tensor,
        g_data: torch.Tensor,
        spectral_log_left: torch.Tensor,
        momentum: torch.Tensor,
        sign_momentum: torch.Tensor,
        exp_avg_sq: torch.Tensor,
        row_var: torch.Tensor,
        beta1: float,
        beta2: float,
        beta3: float,
        weight_decay: float,
        max_scale: float,
        eps: float,
        step_t: torch.Tensor,
        cautious_update: bool,
        cautious_wd: bool,
        spectral_log_bound: float,
        relative_wd: bool = False,
        relative_wd_delta: float = 1e-3,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Core 1D/0D step with a full-rank FFT spectral warp."""
        # 1. RMS normalize gradient for 1D/0D parameters
        grad_rms = torch.sqrt(g_data.pow(2).mean() + eps)
        g_norm = g_data / grad_rms

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

        # 3. Track squared innovation relative to un-updated momentum
        diff_sq = (g_norm - momentum).pow(2)
        exp_avg_sq.lerp_(diff_sq, 1.0 - poly_beta2)
        denom = exp_avg_sq.sqrt().clamp_min_(eps)

        # 4. Update tracked value momentum using poly_beta1
        momentum.lerp_(g_norm, 1.0 - poly_beta1)

        # 5. Craft update
        update = sign_momentum * (momentum.abs() / denom)

        # 6. Apply the spectral warp to the vector dimension
        update_2d = update.reshape(-1, 1)
        warped = _spectral_apply(
            update_2d, spectral_log_left, None, spectral_log_bound
        )
        O = torch.where(step_t <= 1.0, update_2d, warped).reshape(update.shape)

        # 7. Track variance
        var_sq = O.pow(2).mean()
        row_var.lerp_(var_sq, 1.0 - poly_beta3)
        O_norm = O / torch.sqrt(row_var.clamp_min(eps))

        # 8. Rescale back to the warped update's RMS norm
        target_rms = torch.sqrt(O.pow(2).mean() + eps)
        current_rms = torch.sqrt(O_norm.pow(2).mean() + eps)
        update_final = O_norm * (target_rms / (current_rms + eps))

        # 9. Cautious updates against the original gradient
        if cautious_update:
            mask = (g_data * update_final > 0).to(update_final.dtype)
            mask_mean = mask.mean().clamp_min(1e-3)
            update_final = update_final * mask / mask_mean

        if weight_decay != 0:
            if relative_wd:
                # 1. Compute RMS values (reuse target_rms if update_final was not modulated by cautious_update)
                update_rms = (
                    torch.sqrt(update_final.pow(2).mean() + eps)
                    if cautious_update
                    else target_rms
                )
                param_rms_smooth = torch.sqrt(
                    p_data.pow(2).mean() + relative_wd_delta**2
                )
                # 2. Scale with Huber smoothing
                scale = update_rms / param_rms_smooth
                # 3. Guard against discrete time-step overshooting
                scale = torch.clamp_max(scale, max_scale)
                # 4. Apply weight decay
                eff_wd = weight_decay * scale
                if cautious_wd:
                    wd_mask = (update_final * p_data >= 0).to(update_final.dtype)
                    update_final = update_final + eff_wd * (p_data * wd_mask)
                else:
                    update_final = update_final + eff_wd * p_data
            elif cautious_wd:
                wd_mask = (update_final * p_data >= 0).to(update_final.dtype)
                update_final = update_final + weight_decay * (p_data * wd_mask)
            else:
                update_final = update_final + weight_decay * p_data

        return update_final, update

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

                    # Skip warp distortion for 1D vectors, biases, norm layers, scalars, and DoRA scales
                    is_1d_or_scalar = (
                        p.ndim < 2
                        or p.numel() == 1
                        or getattr(p, "is_scalar", False)
                        or getattr(p, "is_bias", False)
                        or getattr(p, "is_norm", False)
                        or getattr(p, "_is_dora_scale", False)
                    )
                    if group["meta_lr"] > 0 and not is_1d_or_scalar:
                        if group["warp_mode"] == "dense":
                            state["warp"] = torch.zeros(
                                warp_m,
                                warp_m,
                                dtype=group["warp_dtype"],
                                device=p.device,
                            )
                        else:
                            state["spectral_log_left"] = torch.zeros(
                                warp_m, dtype=torch.float32, device=p.device
                            )
                            # Only enable bilateral (right) spectral warp for hidden layers
                            if p.ndim >= 2 and group["spectral_bilateral"] and getattr(p, "is_hidden", True):
                                state["spectral_log_right"] = torch.zeros(
                                    w_2d.shape[1],
                                    dtype=torch.float32,
                                    device=p.device,
                                )

            if self._foreach:
                self._step_foreach(group)
            else:
                self._step_native(group)

        return loss

    def _step_native(self, group):
        lr = group["lr"]
        lr_float = float(lr) if isinstance(lr, torch.Tensor) else float(lr)
        beta1, beta2, beta3 = group["betas"]
        weight_decay = float(group["weight_decay"])
        cautious_update = group["cautious_update"]
        cautious_wd = group["cautious_wd"]
        stochastic_fp = group["stochastic_fp"]
        sinkhorn_steps = group["sinkhorn_steps"]
        ortho_dtype = group["ortho_dtype"]
        eps = group["eps"]
        meta_lr = group["meta_lr"]
        meta_wd = group["meta_wd"]
        spectral_bilateral = group["spectral_bilateral"]
        spectral_log_bound = group["spectral_log_bound"]
        relative_wd = group.get("relative_wd", False)
        relative_wd_delta = group.get("relative_wd_delta", 1e-3)
        relative_wd_max_contraction = group.get("relative_wd_max_contraction", 0.99)

        for p in group["params"]:
            if p.grad is None:
                continue

            wd_ratio = getattr(p, "weight_decay_ratio", None)
            if wd_ratio is None:
                if getattr(p, "is_bias", False) or getattr(p, "is_norm", False) or getattr(p, "is_scalar", False) or getattr(p, "_is_dora_scale", False):
                    wd_ratio = 0.0
                else:
                    wd_ratio = 1.0
            p_weight_decay = weight_decay * float(wd_ratio)
            p_max_scale = _relative_wd_max_scale(
                lr, p_weight_decay, relative_wd_max_contraction
            )

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

            warp = state.get("warp")
            spectral_log_left = state.get("spectral_log_left")

            if p.ndim >= 2:
                p_2d = _reshape_to_2d(p.data)
                g_2d = _reshape_to_2d(grad)

                p_2d_fp32 = p_2d.float() if p.dtype is torch.bfloat16 else p_2d
                # Adaptive Sinkhorn: 2 iterations for high aspect ratio matrices (e.g. LoRA)
                p_sinkhorn_steps = 2 if (p_2d.shape[0] / max(p_2d.shape[1], 1) > 32 or p_2d.shape[1] / max(p_2d.shape[0], 1) > 32) else sinkhorn_steps

                if spectral_log_left is not None:
                    spectral_log_right = state.get("spectral_log_right")
                    if spectral_log_right is None:
                        spectral_log_right = torch.empty(
                            0, dtype=torch.float32, device=p.device
                        )
                    update_final, crafted_update = self._compiled_spectral_step_2d(
                        p_2d_fp32,
                        g_2d,
                        spectral_log_left,
                        spectral_log_right,
                        state["momentum"],
                        state["sign_momentum"],
                        state["exp_avg_sq_row"],
                        state["exp_avg_sq_col"],
                        state["row_var"],
                        beta1,
                        beta2,
                        beta3,
                        p_weight_decay,
                        p_max_scale,
                        eps,
                        step_t,
                        p_sinkhorn_steps,
                        cautious_update,
                        cautious_wd,
                        ortho_dtype,
                        spectral_bilateral,
                        spectral_log_bound,
                        relative_wd,
                        relative_wd_delta,
                    )
                elif warp is not None:
                    update_final, crafted_update = self._compiled_warp_step_2d(
                        p_2d_fp32,
                        g_2d,
                        warp,
                        state["momentum"],
                        state["sign_momentum"],
                        state["exp_avg_sq_row"],
                        state["exp_avg_sq_col"],
                        state["row_var"],
                        beta1,
                        beta2,
                        beta3,
                        p_weight_decay,
                        p_max_scale,
                        eps,
                        step_t,
                        p_sinkhorn_steps,
                        cautious_update,
                        cautious_wd,
                        ortho_dtype,
                        relative_wd,
                        relative_wd_delta,
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
                        p_weight_decay,
                        p_max_scale,
                        eps,
                        step_t,
                        p_sinkhorn_steps,
                        cautious_update,
                        cautious_wd,
                        ortho_dtype,
                        relative_wd,
                        relative_wd_delta,
                    )

                p_2d_fp32.add_(update_final, alpha=-lr_float)

                if p.dtype is torch.bfloat16:
                    if stochastic_fp:
                        copy_stochastic_(p.data, p_2d_fp32.view_as(p.data))
                    else:
                        p.data.copy_(p_2d_fp32.view_as(p.data))

                if spectral_log_left is not None:
                    _spectral_meta_update(
                        state,
                        crafted_update,
                        meta_lr,
                        meta_wd,
                        spectral_log_bound,
                    )
                elif warp is not None:
                    _warp_meta_update(state, crafted_update, meta_lr, meta_wd)

            else:
                p_fp32 = p.data.float() if p.dtype is torch.bfloat16 else p.data

                if spectral_log_left is not None:
                    update_final, crafted_update = self._compiled_spectral_step_1d(
                        p_fp32,
                        grad,
                        spectral_log_left,
                        state["momentum"],
                        state["sign_momentum"],
                        state["exp_avg_sq"],
                        state["row_var"],
                        beta1,
                        beta2,
                        beta3,
                        p_weight_decay,
                        p_max_scale,
                        eps,
                        step_t,
                        cautious_update,
                        cautious_wd,
                        spectral_log_bound,
                        relative_wd,
                        relative_wd_delta,
                    )
                elif warp is not None:
                    update_final, crafted_update = self._compiled_warp_step_1d(
                        p_fp32,
                        grad,
                        warp,
                        state["momentum"],
                        state["sign_momentum"],
                        state["exp_avg_sq"],
                        state["row_var"],
                        beta1,
                        beta2,
                        beta3,
                        p_weight_decay,
                        p_max_scale,
                        eps,
                        step_t,
                        cautious_update,
                        cautious_wd,
                        relative_wd,
                        relative_wd_delta,
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
                        p_weight_decay,
                        p_max_scale,
                        eps,
                        step_t,
                        cautious_update,
                        cautious_wd,
                        relative_wd,
                        relative_wd_delta,
                    )

                p_fp32.add_(update_final, alpha=-lr_float)

                if p.dtype is torch.bfloat16:
                    if stochastic_fp:
                        copy_stochastic_(p.data, p_fp32)
                    else:
                        p.data.copy_(p_fp32)

                if spectral_log_left is not None:
                    _spectral_meta_update(
                        state,
                        crafted_update.reshape(-1, 1),
                        meta_lr,
                        meta_wd,
                        spectral_log_bound,
                    )
                elif warp is not None:
                    _warp_meta_update(
                        state, crafted_update.reshape(-1, 1), meta_lr, meta_wd
                    )

    def _step_foreach(self, group):
        lr = group["lr"]
        lr_float = float(lr) if isinstance(lr, torch.Tensor) else float(lr)
        beta1, beta2, beta3 = group["betas"]
        weight_decay = float(group["weight_decay"])
        cautious_update = group["cautious_update"]
        cautious_wd = group["cautious_wd"]
        stochastic_fp = group["stochastic_fp"]
        sinkhorn_steps = group["sinkhorn_steps"]
        ortho_dtype = group["ortho_dtype"]
        eps = group["eps"]
        meta_lr = group["meta_lr"]
        meta_wd = group["meta_wd"]
        spectral_bilateral = group["spectral_bilateral"]
        spectral_log_bound = group["spectral_log_bound"]
        relative_wd = group.get("relative_wd", False)
        relative_wd_delta = group.get("relative_wd_delta", 1e-3)
        relative_wd_max_contraction = group.get("relative_wd_max_contraction", 0.99)

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
            wd_ratio = getattr(p, "weight_decay_ratio", None)
            if wd_ratio is None:
                if getattr(p, "is_bias", False) or getattr(p, "is_norm", False) or getattr(p, "is_scalar", False) or getattr(p, "_is_dora_scale", False):
                    wd_ratio = 0.0
                else:
                    wd_ratio = 1.0
            p_weight_decay = weight_decay * float(wd_ratio)
            p_max_scale = _relative_wd_max_scale(
                lr, p_weight_decay, relative_wd_max_contraction
            )

            state = self.state[p]
            state["step"] += 1
            if "step_t" not in state:
                state["step_t"] = torch.tensor(float(state["step"]), dtype=torch.float32, device=p.device)
            else:
                state["step_t"].fill_(float(state["step"]))
            step_t = state["step_t"]

            grad = p.grad.data.float() if p.grad.data.dtype in (torch.bfloat16, torch.float16) else p.grad.data

            p_fp32 = p.data.float() if p.dtype is torch.bfloat16 else p.data
            warp = state.get("warp")
            spectral_log_left = state.get("spectral_log_left")
            if spectral_log_left is not None:
                update_final, crafted_update = self._compiled_spectral_step_1d(
                    p_fp32,
                    grad,
                    spectral_log_left,
                    state["momentum"],
                    state["sign_momentum"],
                    state["exp_avg_sq"],
                    state["row_var"],
                    beta1,
                    beta2,
                    beta3,
                    p_weight_decay,
                    p_max_scale,
                    eps,
                    step_t,
                    cautious_update,
                    cautious_wd,
                    spectral_log_bound,
                    relative_wd,
                    relative_wd_delta,
                )
            elif warp is not None:
                update_final, crafted_update = self._compiled_warp_step_1d(
                    p_fp32,
                    grad,
                    warp,
                    state["momentum"],
                    state["sign_momentum"],
                    state["exp_avg_sq"],
                    state["row_var"],
                    beta1,
                    beta2,
                    beta3,
                    p_weight_decay,
                    p_max_scale,
                    eps,
                    step_t,
                    cautious_update,
                    cautious_wd,
                    relative_wd,
                    relative_wd_delta,
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
                    p_weight_decay,
                    p_max_scale,
                    eps,
                    step_t,
                    cautious_update,
                    cautious_wd,
                    relative_wd,
                    relative_wd_delta,
                )
            p_fp32.add_(update_final, alpha=-lr_float)

            if p.dtype is torch.bfloat16:
                if stochastic_fp:
                    copy_stochastic_(p.data, p_fp32)
                else:
                    p.data.copy_(p_fp32)

            if spectral_log_left is not None:
                _spectral_meta_update(
                    state,
                    crafted_update.reshape(-1, 1),
                    meta_lr,
                    meta_wd,
                    spectral_log_bound,
                )
            elif warp is not None:
                _warp_meta_update(
                    state, crafted_update.reshape(-1, 1), meta_lr, meta_wd
                )

        # 2. Process 2D+ parameters
        if not params_2d:
            return

        updates_final_list = []
        p_fp32_list = []
        p_original_list = []
        meta_list = []

        for p in params_2d:
            wd_ratio = getattr(p, "weight_decay_ratio", None)
            if wd_ratio is None:
                if getattr(p, "is_bias", False) or getattr(p, "is_norm", False) or getattr(p, "is_scalar", False) or getattr(p, "_is_dora_scale", False):
                    wd_ratio = 0.0
                else:
                    wd_ratio = 1.0
            p_weight_decay = weight_decay * float(wd_ratio)
            p_max_scale = _relative_wd_max_scale(
                lr, p_weight_decay, relative_wd_max_contraction
            )

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

            p_2d_fp32 = p_2d.float() if p.dtype is torch.bfloat16 else p_2d
            # Adaptive Sinkhorn: 2 iterations for high aspect ratio matrices (e.g. LoRA)
            p_sinkhorn_steps = 2 if (p_2d.shape[0] / max(p_2d.shape[1], 1) > 32 or p_2d.shape[1] / max(p_2d.shape[0], 1) > 32) else sinkhorn_steps

            warp = state.get("warp")
            spectral_log_left = state.get("spectral_log_left")
            if spectral_log_left is not None:
                spectral_log_right = state.get("spectral_log_right")
                if spectral_log_right is None:
                    spectral_log_right = torch.empty(
                        0, dtype=torch.float32, device=p.device
                    )
                update_final, crafted_update = self._compiled_spectral_step_2d(
                    p_2d_fp32,
                    g_2d,
                    spectral_log_left,
                    spectral_log_right,
                    state["momentum"],
                    state["sign_momentum"],
                    state["exp_avg_sq_row"],
                    state["exp_avg_sq_col"],
                    state["row_var"],
                    beta1,
                    beta2,
                    beta3,
                    p_weight_decay,
                    p_max_scale,
                    eps,
                    step_t,
                    p_sinkhorn_steps,
                    cautious_update,
                    cautious_wd,
                    ortho_dtype,
                    spectral_bilateral,
                    spectral_log_bound,
                    relative_wd,
                    relative_wd_delta,
                    )
                meta_list.append((state, crafted_update))
            elif warp is not None:
                update_final, crafted_update = self._compiled_warp_step_2d(
                    p_2d_fp32,
                    g_2d,
                    warp,
                    state["momentum"],
                    state["sign_momentum"],
                    state["exp_avg_sq_row"],
                    state["exp_avg_sq_col"],
                    state["row_var"],
                    beta1,
                    beta2,
                    beta3,
                    p_weight_decay,
                    p_max_scale,
                    eps,
                    step_t,
                    p_sinkhorn_steps,
                    cautious_update,
                    cautious_wd,
                    ortho_dtype,
                    relative_wd,
                    relative_wd_delta,
                    )
                meta_list.append((state, crafted_update))
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
                    p_weight_decay,
                    p_max_scale,
                    eps,
                    step_t,
                    p_sinkhorn_steps,
                    cautious_update,
                    cautious_wd,
                    ortho_dtype,
                    relative_wd,
                    relative_wd_delta,
                    )

            updates_final_list.append(update_final.view_as(p_2d_fp32))
            p_fp32_list.append(p_2d_fp32)
            p_original_list.append(p)

        torch._foreach_add_(p_fp32_list, updates_final_list, alpha=-lr_float)

        for p, p_fp32 in zip(p_original_list, p_fp32_list):
            if p.dtype is torch.bfloat16:
                p_reshaped = p_fp32.view_as(p.data)
                if stochastic_fp:
                    copy_stochastic_(p.data, p_reshaped)
                else:
                    p.data.copy_(p_reshaped)

        for state, crafted_update in meta_list:
            if "spectral_log_left" in state:
                _spectral_meta_update(
                    state,
                    crafted_update,
                    meta_lr,
                    meta_wd,
                    spectral_log_bound,
                )
            else:
                _warp_meta_update(state, crafted_update, meta_lr, meta_wd)
