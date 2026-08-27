"""Faithful Simplified-AdEMAMix combined with AINO geometry.

This optimizer keeps the AINO gradient normalization and geometric post-
processing, but replaces AINO's sign/innovation update construction with the
theory-style Simplified-AdEMAMix update:

    m_t = beta1 * m_{t-1} + g_norm_t
    v_t = beta2 * v_{t-1} + (1 - beta2) * g_norm_t**2
    u_t = (m_t + alpha * g_norm_t) / (sqrt(v_t) + eps)

The resulting signed update is then optionally passed through WarpAINO's
dense or spectral warp and AINO's Newton-Schulz/RMS post-processing. Unlike
the conservative ``aino_mix_alpha`` option on WarpAINO, this class does not
retain AINO sign momentum or innovation variance tracking.

The full gradient-square state is intentional: it preserves the
Simplified-AdEMAMix algorithm rather than substituting AINO's factorized
innovation estimate. This costs one additional FP32 tensor per parameter.
"""

import logging
import math
from numbers import Real
from typing import Optional, Tuple, Union

import torch
from torch.optim import Optimizer

try:
    from .warpaino import (
        WarpAINO,
        _apply_lr_update_,
        _finalize_aino_update_1d,
        _finalize_aino_update_2d,
        _foreach_apply_lr_,
        _get_fp32_work,
        _relative_wd_max_scale,
        _reshape_to_2d,
        _sinkhorn_normalize,
        _spectral_apply,
        _warp_meta_update,
        _writeback_fp32_work_,
    )
except ImportError:
    # Support the standalone import pattern used by tests when optional
    # dependencies imported by the package __init__ are unavailable.
    from warpaino_standalone import (
        WarpAINO,
        _apply_lr_update_,
        _finalize_aino_update_1d,
        _finalize_aino_update_2d,
        _foreach_apply_lr_,
        _get_fp32_work,
        _relative_wd_max_scale,
        _reshape_to_2d,
        _sinkhorn_normalize,
        _spectral_apply,
        _warp_meta_update,
        _writeback_fp32_work_,
    )


def _scheduled_beta1(
    step_t: torch.Tensor,
    beta_end: float,
    beta_start: float,
    warmup: Optional[int],
) -> torch.Tensor:
    """Return the paper's log-half-life beta1 warm-up on the target device."""
    beta_end_t = torch.as_tensor(beta_end, dtype=torch.float32, device=step_t.device)
    if warmup is None or warmup <= 0:
        return beta_end_t

    # These are constants with respect to the optimizer step and are evaluated
    # on the host only once per call; the interpolation itself stays on-device.
    half_life_start = math.log(0.5) / math.log(beta_start + 1e-8) - 1.0
    half_life_end = math.log(0.5) / math.log(beta_end + 1e-8) - 1.0
    progress = step_t / float(warmup)
    interpolated = (1.0 - progress) * half_life_start + progress * half_life_end
    warmed = torch.pow(
        torch.as_tensor(0.5, dtype=torch.float32, device=step_t.device),
        1.0 / (interpolated + 1.0),
    )
    return torch.where(step_t < float(warmup), warmed, beta_end_t)


def _prepare_simplified_aino_update_2d(
    g_2d: torch.Tensor,
    momentum: torch.Tensor,
    exp_avg_sq: torch.Tensor,
    beta1_t: torch.Tensor,
    beta2: float,
    alpha: float,
    eps: float,
    step_t: torch.Tensor,
    sinkhorn_steps: int,
    num_sum: torch.Tensor,
    bias_correction1: bool,
    bias_correction2: bool,
) -> torch.Tensor:
    """Prepare the faithful signed Simplified-AdEMAMix 2D update."""
    g_norm = _sinkhorn_normalize(g_2d, sinkhorn_steps, eps)

    # Theory-style momentum: the current gradient has coefficient 1.
    momentum.mul_(beta1_t).add_(g_norm)
    exp_avg_sq.mul_(beta2).addcmul_(g_norm, g_norm, value=1.0 - beta2)

    numerator = momentum + alpha * g_norm
    if bias_correction1:
        numerator = numerator / num_sum.clamp_min_(1e-12)

    denominator = exp_avg_sq
    if bias_correction2:
        correction = (1.0 - beta2**step_t).clamp_min_(1e-12)
        denominator = denominator / correction
    denominator = denominator.sqrt().add_(eps)
    return numerator / denominator


def _prepare_simplified_aino_update_1d(
    g_data: torch.Tensor,
    momentum: torch.Tensor,
    exp_avg_sq: torch.Tensor,
    beta1_t: torch.Tensor,
    beta2: float,
    alpha: float,
    eps: float,
    step_t: torch.Tensor,
    num_sum: torch.Tensor,
    bias_correction1: bool,
    bias_correction2: bool,
) -> torch.Tensor:
    """Prepare the faithful signed Simplified-AdEMAMix 1D update."""
    grad_rms = torch.sqrt(g_data.pow(2).mean() + eps)
    g_norm = g_data / grad_rms

    momentum.mul_(beta1_t).add_(g_norm)
    exp_avg_sq.mul_(beta2).addcmul_(g_norm, g_norm, value=1.0 - beta2)

    numerator = momentum + alpha * g_norm
    if bias_correction1:
        numerator = numerator / num_sum.clamp_min_(1e-12)

    denominator = exp_avg_sq
    if bias_correction2:
        correction = (1.0 - beta2**step_t).clamp_min_(1e-12)
        denominator = denominator / correction
    denominator = denominator.sqrt().add_(eps)
    return numerator / denominator


class SimplifiedAdEMAMixAINO(WarpAINO):
    """Faithful Simplified-AdEMAMix update with AINO geometry and warping.

    ``betas`` has the layout ``(beta1, beta2, beta3)``. ``beta1`` is used in
    the theory-style momentum recurrence, ``beta2`` is the EMA coefficient of
    a full gradient-square state, and ``beta3`` remains AINO's row-variance
    coefficient used after Newton-Schulz processing.

    AINO's Sinkhorn/RMS gradient normalization, Newton-Schulz processing,
    cautious update, weight decay, and optional dense/spectral WarpAINO
    operator are retained. AINO sign momentum and innovation variance are not
    used, because they would make the update no longer faithful to
    Simplified-AdEMAMix.
    """

    def __init__(
        self,
        params,
        lr: Union[float, torch.Tensor] = 1e-4,
        betas: Tuple[float, float, float] = (0.99, 0.95, 0.999),
        alpha: float = 0.0,
        beta1_warmup: Optional[int] = None,
        min_beta1: float = 0.9,
        bias_correction1: bool = False,
        bias_correction2: bool = True,
        weight_decay: float = 0.0,
        cautious_update: bool = True,
        cautious_wd: bool = True,
        stochastic_fp: bool = True,
        kahan_sum: bool = False,
        meta_ema: bool = False,
        meta_ema_beta: float = 0.85,
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
        if (
            isinstance(alpha, bool)
            or not isinstance(alpha, Real)
            or not math.isfinite(float(alpha))
            or alpha < 0.0
        ):
            raise ValueError(f"alpha must be a finite non-negative float, got {alpha}")
        if (
            isinstance(min_beta1, bool)
            or not isinstance(min_beta1, Real)
            or not math.isfinite(float(min_beta1))
            or not 0.0 <= min_beta1 < 1.0
        ):
            raise ValueError(
                f"min_beta1 must be a finite value in [0, 1), got {min_beta1}"
            )
        if beta1_warmup is not None and (
            isinstance(beta1_warmup, bool)
            or not isinstance(beta1_warmup, int)
            or beta1_warmup < 0
        ):
            raise ValueError(
                "beta1_warmup must be None or a non-negative integer, "
                f"got {beta1_warmup}"
            )
        if not isinstance(bias_correction1, bool):
            raise ValueError(
                f"bias_correction1 must be bool, got {bias_correction1}"
            )
        if not isinstance(bias_correction2, bool):
            raise ValueError(
                f"bias_correction2 must be bool, got {bias_correction2}"
            )

        # Parent initialization supplies the validated WarpAINO geometry,
        # weight-decay, low-precision, and warp defaults. Its AINO core is not
        # compiled here because this class uses separate hybrid cores.
        super().__init__(
            params,
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
            cautious_update=cautious_update,
            cautious_wd=cautious_wd,
            stochastic_fp=stochastic_fp,
            kahan_sum=kahan_sum,
            meta_ema=meta_ema,
            meta_ema_beta=meta_ema_beta,
            compile_step=False,
            foreach=foreach,
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
            aino_mix_alpha=0.0,
        )

        extra_defaults = dict(
            alpha=float(alpha),
            beta1_warmup=beta1_warmup,
            min_beta1=float(min_beta1),
            bias_correction1=bias_correction1,
            bias_correction2=bias_correction2,
        )
        self.defaults.update(extra_defaults)
        for group in self.param_groups:
            group.update(extra_defaults)

        self._compile_step = compile_step
        self._foreach = foreach
        if compile_step:
            try:
                self._compiled_hybrid_2d = torch.compile(
                    self._hybrid_step_core_2d,
                    fullgraph=True,
                    dynamic=False,
                )
                self._compiled_hybrid_warp_2d = torch.compile(
                    self._hybrid_warp_step_core_2d,
                    fullgraph=True,
                    dynamic=False,
                )
                self._compiled_hybrid_spectral_2d = torch.compile(
                    self._hybrid_spectral_step_core_2d,
                    fullgraph=True,
                    dynamic=False,
                )
                self._compiled_hybrid_1d = torch.compile(
                    self._hybrid_step_core_1d,
                    fullgraph=True,
                    dynamic=False,
                )
            except Exception as exc:
                logging.warning(
                    "torch.compile failed for SimplifiedAdEMAMixAINO cores: %s. "
                    "Falling back to uncompiled cores.",
                    exc,
                )
                self._set_uncompiled_cores()
        else:
            self._set_uncompiled_cores()

    def _set_uncompiled_cores(self) -> None:
        self._compiled_hybrid_2d = self._hybrid_step_core_2d
        self._compiled_hybrid_warp_2d = self._hybrid_warp_step_core_2d
        self._compiled_hybrid_spectral_2d = self._hybrid_spectral_step_core_2d
        self._compiled_hybrid_1d = self._hybrid_step_core_1d

    @staticmethod
    def _hybrid_step_core_2d(
        p_2d: torch.Tensor,
        g_2d: torch.Tensor,
        momentum: torch.Tensor,
        exp_avg_sq: torch.Tensor,
        row_var: torch.Tensor,
        num_sum: torch.Tensor,
        beta1_t: torch.Tensor,
        beta2: float,
        beta3: float,
        alpha: float,
        weight_decay: float,
        max_scale: Union[float, torch.Tensor],
        eps: float,
        step_t: torch.Tensor,
        sinkhorn_steps: int,
        cautious_update: bool,
        cautious_wd: bool,
        ortho_dtype: torch.dtype,
        relative_wd: bool,
        relative_wd_delta: float,
        bias_correction1: bool,
        bias_correction2: bool,
    ) -> torch.Tensor:
        update = _prepare_simplified_aino_update_2d(
            g_2d,
            momentum,
            exp_avg_sq,
            beta1_t,
            beta2,
            alpha,
            eps,
            step_t,
            sinkhorn_steps,
            num_sum,
            bias_correction1,
            bias_correction2,
        )
        return _finalize_aino_update_2d(
            p_2d,
            g_2d,
            update,
            row_var,
            beta3,
            weight_decay,
            max_scale,
            eps,
            step_t,
            cautious_update,
            cautious_wd,
            ortho_dtype,
            relative_wd,
            relative_wd_delta,
            False,
            10.0,
        )

    @staticmethod
    def _hybrid_warp_step_core_2d(
        p_2d: torch.Tensor,
        g_2d: torch.Tensor,
        warp: torch.Tensor,
        momentum: torch.Tensor,
        exp_avg_sq: torch.Tensor,
        row_var: torch.Tensor,
        num_sum: torch.Tensor,
        beta1_t: torch.Tensor,
        beta2: float,
        beta3: float,
        alpha: float,
        weight_decay: float,
        max_scale: Union[float, torch.Tensor],
        eps: float,
        step_t: torch.Tensor,
        sinkhorn_steps: int,
        cautious_update: bool,
        cautious_wd: bool,
        ortho_dtype: torch.dtype,
        relative_wd: bool,
        relative_wd_delta: float,
        bias_correction1: bool,
        bias_correction2: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        update = _prepare_simplified_aino_update_2d(
            g_2d,
            momentum,
            exp_avg_sq,
            beta1_t,
            beta2,
            alpha,
            eps,
            step_t,
            sinkhorn_steps,
            num_sum,
            bias_correction1,
            bias_correction2,
        )
        update_warped = torch.addmm(update, warp.float(), update)
        update_final = _finalize_aino_update_2d(
            p_2d,
            g_2d,
            update_warped,
            row_var,
            beta3,
            weight_decay,
            max_scale,
            eps,
            step_t,
            cautious_update,
            cautious_wd,
            ortho_dtype,
            relative_wd,
            relative_wd_delta,
            False,
            10.0,
        )
        return update_final, update

    @staticmethod
    def _hybrid_spectral_step_core_2d(
        p_2d: torch.Tensor,
        g_2d: torch.Tensor,
        spectral_log_left: torch.Tensor,
        spectral_log_right: torch.Tensor,
        momentum: torch.Tensor,
        exp_avg_sq: torch.Tensor,
        row_var: torch.Tensor,
        num_sum: torch.Tensor,
        beta1_t: torch.Tensor,
        beta2: float,
        beta3: float,
        alpha: float,
        weight_decay: float,
        max_scale: Union[float, torch.Tensor],
        eps: float,
        step_t: torch.Tensor,
        sinkhorn_steps: int,
        cautious_update: bool,
        cautious_wd: bool,
        ortho_dtype: torch.dtype,
        spectral_bilateral: bool,
        spectral_log_bound: float,
        relative_wd: bool,
        relative_wd_delta: float,
        bias_correction1: bool,
        bias_correction2: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        update = _prepare_simplified_aino_update_2d(
            g_2d,
            momentum,
            exp_avg_sq,
            beta1_t,
            beta2,
            alpha,
            eps,
            step_t,
            sinkhorn_steps,
            num_sum,
            bias_correction1,
            bias_correction2,
        )
        if spectral_bilateral and spectral_log_right.numel() > 0:
            update_warped = _spectral_apply(
                update,
                spectral_log_left,
                spectral_log_right,
                spectral_log_bound,
            )
        else:
            update_warped = _spectral_apply(
                update, spectral_log_left, None, spectral_log_bound
            )
        update_warped = torch.where(step_t <= 1.0, update, update_warped)
        update_final = _finalize_aino_update_2d(
            p_2d,
            g_2d,
            update_warped,
            row_var,
            beta3,
            weight_decay,
            max_scale,
            eps,
            step_t,
            cautious_update,
            cautious_wd,
            ortho_dtype,
            relative_wd,
            relative_wd_delta,
            False,
            10.0,
        )
        return update_final, update

    @staticmethod
    def _hybrid_step_core_1d(
        p_data: torch.Tensor,
        g_data: torch.Tensor,
        momentum: torch.Tensor,
        exp_avg_sq: torch.Tensor,
        row_var: torch.Tensor,
        num_sum: torch.Tensor,
        beta1_t: torch.Tensor,
        beta2: float,
        beta3: float,
        alpha: float,
        weight_decay: float,
        max_scale: Union[float, torch.Tensor],
        eps: float,
        step_t: torch.Tensor,
        cautious_update: bool,
        cautious_wd: bool,
        relative_wd: bool,
        relative_wd_delta: float,
        bias_correction1: bool,
        bias_correction2: bool,
    ) -> torch.Tensor:
        update = _prepare_simplified_aino_update_1d(
            g_data,
            momentum,
            exp_avg_sq,
            beta1_t,
            beta2,
            alpha,
            eps,
            step_t,
            num_sum,
            bias_correction1,
            bias_correction2,
        )
        return _finalize_aino_update_1d(
            p_data,
            g_data,
            update,
            row_var,
            beta3,
            weight_decay,
            max_scale,
            eps,
            step_t,
            cautious_update,
            cautious_wd,
            relative_wd,
            relative_wd_delta,
            False,
            10.0,
        )

    def _initialize_state(self, p: torch.Tensor, group: dict) -> None:
        state = self.state[p]
        if len(state) != 0:
            return

        state["step"] = 0
        state["step_t"] = torch.zeros((), dtype=torch.float32, device=p.device)
        state["num_sum"] = torch.zeros((), dtype=torch.float32, device=p.device)

        if p.ndim >= 2:
            w_2d = _reshape_to_2d(p.data)
            m, _ = w_2d.shape
            state["momentum"] = torch.zeros_like(w_2d, dtype=torch.float32)
            state["exp_avg_sq"] = torch.zeros_like(w_2d, dtype=torch.float32)
            state["row_var"] = torch.zeros(m, dtype=torch.float32, device=p.device)
            warp_m = m
        else:
            state["momentum"] = torch.zeros_like(p.data, dtype=torch.float32)
            state["exp_avg_sq"] = torch.zeros_like(p.data, dtype=torch.float32)
            state["row_var"] = torch.zeros((), dtype=torch.float32, device=p.device)
            warp_m = None

        if group["kahan_sum"] and p.dtype in (torch.float16, torch.bfloat16):
            state["param_compensation"] = torch.zeros_like(
                p.data, dtype=torch.float32
            )

        if (
            group["meta_lr"] > 0
            and p.ndim >= 2
            and p.numel() > 1
            and not (
                getattr(p, "is_scalar", False)
                or getattr(p, "is_bias", False)
                or getattr(p, "is_norm", False)
                or getattr(p, "_is_dora_scale", False)
            )
        ):
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
                if group["spectral_bilateral"] and getattr(p, "is_hidden", True):
                    state["spectral_log_right"] = torch.zeros(
                        _reshape_to_2d(p.data).shape[1],
                        dtype=torch.float32,
                        device=p.device,
                    )

    @staticmethod
    def _parameter_weight_decay(p: torch.Tensor, group: dict) -> float:
        wd_ratio = getattr(p, "weight_decay_ratio", None)
        if wd_ratio is None:
            if (
                getattr(p, "is_bias", False)
                or getattr(p, "is_norm", False)
                or getattr(p, "is_scalar", False)
                or getattr(p, "_is_dora_scale", False)
            ):
                wd_ratio = 0.0
            else:
                wd_ratio = 1.0
        return float(group["weight_decay"]) * float(wd_ratio)

    def _run_core_2d(
        self,
        p_2d: torch.Tensor,
        g_2d: torch.Tensor,
        state: dict,
        group: dict,
        p_weight_decay: float,
        p_max_scale: Union[float, torch.Tensor],
        step_t: torch.Tensor,
        sinkhorn_steps: int,
        beta1_t: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        _, beta2, beta3 = group["betas"]
        common = (
            p_2d,
            g_2d,
            state["momentum"],
            state["exp_avg_sq"],
            state["row_var"],
            state["num_sum"],
            beta1_t,
            beta2,
            beta3,
            group["alpha"],
            p_weight_decay,
            p_max_scale,
            group["eps"],
            step_t,
            sinkhorn_steps,
            group["cautious_update"],
            group["cautious_wd"],
            group["ortho_dtype"],
            group.get("relative_wd", False),
            group.get("relative_wd_delta", 1e-3),
            group["bias_correction1"],
            group["bias_correction2"],
        )
        spectral_left = state.get("spectral_log_left")
        if spectral_left is not None:
            spectral_right = state.get("spectral_log_right")
            if spectral_right is None:
                spectral_right = torch.empty(0, dtype=torch.float32, device=p_2d.device)
            return self._compiled_hybrid_spectral_2d(
                p_2d,
                g_2d,
                spectral_left,
                spectral_right,
                state["momentum"],
                state["exp_avg_sq"],
                state["row_var"],
                state["num_sum"],
                beta1_t,
                beta2,
                beta3,
                group["alpha"],
                p_weight_decay,
                p_max_scale,
                group["eps"],
                step_t,
                sinkhorn_steps,
                group["cautious_update"],
                group["cautious_wd"],
                group["ortho_dtype"],
                group["spectral_bilateral"],
                group["spectral_log_bound"],
                group.get("relative_wd", False),
                group.get("relative_wd_delta", 1e-3),
                group["bias_correction1"],
                group["bias_correction2"],
            )
        warp = state.get("warp")
        if warp is not None:
            return self._compiled_hybrid_warp_2d(*common[:2], warp, *common[2:])
        return self._compiled_hybrid_2d(*common), None

    def _run_core_1d(
        self,
        p_data: torch.Tensor,
        g_data: torch.Tensor,
        state: dict,
        group: dict,
        p_weight_decay: float,
        p_max_scale: Union[float, torch.Tensor],
        step_t: torch.Tensor,
        beta1_t: torch.Tensor,
    ) -> torch.Tensor:
        _, beta2, beta3 = group["betas"]
        return self._compiled_hybrid_1d(
            p_data,
            g_data,
            state["momentum"],
            state["exp_avg_sq"],
            state["row_var"],
            state["num_sum"],
            beta1_t,
            beta2,
            beta3,
            group["alpha"],
            p_weight_decay,
            p_max_scale,
            group["eps"],
            step_t,
            group["cautious_update"],
            group["cautious_wd"],
            group.get("relative_wd", False),
            group.get("relative_wd_delta", 1e-3),
            group["bias_correction1"],
            group["bias_correction2"],
        )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            pending = []
            beta1_end, _, _ = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError(
                        "SimplifiedAdEMAMixAINO does not support sparse gradients"
                    )
                self._initialize_state(p, group)
                state = self.state[p]
                state["step"] += 1
                state["step_t"].fill_(float(state["step"]))
                step_t = state["step_t"]

                beta1_t = _scheduled_beta1(
                    step_t,
                    beta_end=beta1_end,
                    beta_start=group["min_beta1"],
                    warmup=group["beta1_warmup"],
                )
                state["num_sum"].mul_(beta1_t).add_(1.0)

                p_weight_decay = self._parameter_weight_decay(p, group)
                p_max_scale = _relative_wd_max_scale(
                    group["lr"],
                    p_weight_decay,
                    group.get("relative_wd_max_contraction", 0.99),
                )
                grad = p.grad.data
                if grad.dtype in (torch.float16, torch.bfloat16):
                    grad = grad.float()

                if p.ndim >= 2:
                    p_2d = _reshape_to_2d(p.data)
                    g_2d = _reshape_to_2d(grad)
                    p_2d_fp32 = _reshape_to_2d(
                        _get_fp32_work(state, p, group["kahan_sum"])
                    )
                    sinkhorn_steps = group["sinkhorn_steps"]
                    if (
                        p_2d.shape[0] / max(p_2d.shape[1], 1) > 32
                        or p_2d.shape[1] / max(p_2d.shape[0], 1) > 32
                    ):
                        sinkhorn_steps = 2
                    update_final, crafted_update = self._run_core_2d(
                        p_2d_fp32,
                        g_2d,
                        state,
                        group,
                        p_weight_decay,
                        p_max_scale,
                        step_t,
                        sinkhorn_steps,
                        beta1_t,
                    )
                    if self._foreach:
                        pending.append(
                            (p, p_2d_fp32, update_final.view_as(p_2d_fp32), state, crafted_update)
                        )
                    else:
                        _apply_lr_update_(p_2d_fp32, update_final, group["lr"])
                        if p.dtype in (torch.float16, torch.bfloat16):
                            _writeback_fp32_work_(
                                p,
                                p_2d_fp32.view_as(p.data),
                                state,
                                group["stochastic_fp"],
                                group["kahan_sum"],
                            )
                        self._update_warp_state(state, crafted_update, group)
                else:
                    p_fp32 = _get_fp32_work(state, p, group["kahan_sum"])
                    update_final = self._run_core_1d(
                        p_fp32,
                        grad,
                        state,
                        group,
                        p_weight_decay,
                        p_max_scale,
                        step_t,
                        beta1_t,
                    )
                    _apply_lr_update_(p_fp32, update_final, group["lr"])
                    if p.dtype in (torch.float16, torch.bfloat16):
                        _writeback_fp32_work_(
                            p,
                            p_fp32,
                            state,
                            group["stochastic_fp"],
                            group["kahan_sum"],
                        )

            if pending:
                parameters = [item[1] for item in pending]
                updates = [item[2] for item in pending]
                _foreach_apply_lr_(parameters, updates, group["lr"])
                for p, p_fp32, _, state, crafted_update in pending:
                    if p.dtype in (torch.float16, torch.bfloat16):
                        _writeback_fp32_work_(
                            p,
                            p_fp32.view_as(p.data),
                            state,
                            group["stochastic_fp"],
                            group["kahan_sum"],
                        )
                    self._update_warp_state(state, crafted_update, group)

        return loss

    def _update_warp_state(
        self, state: dict, crafted_update: Optional[torch.Tensor], group: dict
    ) -> None:
        if crafted_update is None:
            return
        if "spectral_log_left" in state:
            self._run_spectral_meta_update(
                state,
                crafted_update,
                group["meta_lr"],
                group["meta_wd"],
                group["spectral_log_bound"],
                group["meta_ema"],
                group["meta_ema_beta"],
            )
        elif "warp" in state:
            _warp_meta_update(
                state,
                crafted_update,
                group["meta_lr"],
                group["meta_wd"],
                group["meta_ema"],
                group["meta_ema_beta"],
                stochastic_fp=group["stochastic_fp"],
            )


__all__ = ["SimplifiedAdEMAMixAINO"]
