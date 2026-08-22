"""Regression tests for the WarpAINO core deduplication refactor.

The six step cores (``_aino_step_core_{2d,1d}``, ``_WarpAINO_step_core_{2d,1d}``,
``_spectral_step_core_{2d,1d}``) previously each contained an inline copy of:

* the poly-beta horizon correction,
* the Sinkhorn row/col RMS normalization loop,
* the cautious masking block, and
* the decoupled (absolute or RMS-relative) weight-decay block.

These were extracted into ``_poly_beta``, ``_sinkhorn_normalize``,
``_cautious_mask``, and ``_apply_decoupled_wd``. These tests verify that the
helpers reproduce the original inline code exactly, and that the refactored
cores are numerically identical to verbatim pre-refactor reference
implementations, on CUDA.
"""

import os
import sys

import pytest
import torch

# Ensure the custom_scheduler package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    from LoraEasyCustomOptimizer.warpaino import (
        WarpAINO,
        _apply_lr_update_,
        _apply_decoupled_wd,
        _cautious_mask,
        _foreach_apply_lr_,
        _grokfast_filter,
        _poly_beta,
        _select_sign_momentum,
        _sinkhorn_normalize,
        _spectral_apply,
        _spectral_log_gain,
        _update_meta_history,
        gram_newton_schulz_2step,
    )
except ImportError:
    # The package __init__ pulls in optional heavy dependencies
    # (pytorch_optimizer, adv_optm, ...) that may be absent; warpaino.py
    # itself only requires torch, so load it directly as a fallback.
    import importlib.util

    _module_path = os.path.join(
        os.path.dirname(__file__), "..", "LoraEasyCustomOptimizer", "warpaino.py"
    )
    _spec = importlib.util.spec_from_file_location("warpaino_standalone", _module_path)
    _module = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_module)
    WarpAINO = _module.WarpAINO
    _apply_lr_update_ = _module._apply_lr_update_
    _apply_decoupled_wd = _module._apply_decoupled_wd
    _cautious_mask = _module._cautious_mask
    _foreach_apply_lr_ = _module._foreach_apply_lr_
    _grokfast_filter = _module._grokfast_filter
    _poly_beta = _module._poly_beta
    _select_sign_momentum = _module._select_sign_momentum
    _sinkhorn_normalize = _module._sinkhorn_normalize
    _spectral_apply = _module._spectral_apply
    _spectral_log_gain = _module._spectral_log_gain
    _update_meta_history = _module._update_meta_history
    gram_newton_schulz_2step = _module.gram_newton_schulz_2step

DEVICE = "cuda"

BETAS = (0.95, 0.95, 0.999)
EPS = 1e-16


def _requires_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")


# ---------------------------------------------------------------------------
# Helper unit tests vs. the original inline formulas
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("beta", [0.9, 0.95, 0.999])
@pytest.mark.parametrize("t", [1.0, 2.0, 5.0, 100.0, 1000.0])
def test_poly_beta_matches_original_formula(beta, t):
    """_poly_beta reproduces the inline (beta^t - beta) / (beta^t - 1), 0 at t <= 1."""
    _requires_cuda()
    step_t = torch.tensor(t, device=DEVICE)
    beta_pow = beta ** step_t
    expected = torch.where(
        step_t > 1.0,
        (beta_pow - beta) / (beta_pow - 1.0),
        torch.zeros_like(beta_pow),
    )
    torch.testing.assert_close(_poly_beta(beta, step_t), expected)


def test_poly_beta_zero_at_first_step():
    """The poly-beta correction is exactly 0 for step_t <= 1."""
    _requires_cuda()
    for t in (0.0, 1.0):
        step_t = torch.tensor(t, device=DEVICE)
        assert _poly_beta(0.95, step_t).item() == 0.0


@pytest.mark.parametrize("steps", [0, 1, 2, 5])
def test_sinkhorn_normalize_matches_original_loop(steps):
    """_sinkhorn_normalize reproduces the inline alternating row/col RMS loop."""
    _requires_cuda()
    torch.manual_seed(0)
    g = torch.randn(8, 16, device=DEVICE)

    expected = g
    for _ in range(steps):
        row_rms = torch.sqrt(expected.pow(2).mean(dim=-1, keepdim=True) + EPS)
        expected = expected / row_rms
        col_rms = torch.sqrt(expected.pow(2).mean(dim=-2, keepdim=True) + EPS)
        expected = expected / col_rms

    torch.testing.assert_close(_sinkhorn_normalize(g, steps, EPS), expected)


def test_sinkhorn_normalize_does_not_mutate_input():
    """The helper must not modify the caller's gradient tensor."""
    _requires_cuda()
    g = torch.randn(4, 8, device=DEVICE)
    g_orig = g.clone()
    _sinkhorn_normalize(g, 3, EPS)
    torch.testing.assert_close(g, g_orig)


def test_cautious_mask_matches_original_block():
    """_cautious_mask reproduces the inline mask/renormalize block."""
    _requires_cuda()
    torch.manual_seed(1)
    update = torch.randn(32, device=DEVICE)
    grad = torch.randn(32, device=DEVICE)

    mask = (grad * update > 0).to(update.dtype)
    mask_mean = mask.mean().clamp_min(1e-3)
    expected = update * mask / mask_mean

    torch.testing.assert_close(_cautious_mask(update, grad), expected)


def test_cautious_mask_all_disagree_degenerate():
    """With full disagreement the mask mean floor (1e-3) keeps output finite."""
    _requires_cuda()
    update = torch.ones(16, device=DEVICE)
    grad = -torch.ones(16, device=DEVICE)
    out = _cautious_mask(update, grad)
    assert torch.isfinite(out).all()
    assert (out == 0).all()


@pytest.mark.parametrize("cautious_wd", [False, True])
@pytest.mark.parametrize("cautious_update", [False, True])
def test_apply_decoupled_wd_relative_matches_original(cautious_update, cautious_wd):
    """Relative weight decay matches the original inline block for all flag combos."""
    _requires_cuda()
    torch.manual_seed(2)
    p = torch.randn(16, device=DEVICE)
    update_final = torch.randn(16, device=DEVICE)
    weight_decay, delta, max_scale = 0.1, 1e-3, 50.0

    # Original inline block (pre-refactor), verbatim semantics
    ref = update_final
    update_rms = torch.sqrt(ref.pow(2).mean() + EPS)
    param_rms_smooth = torch.sqrt(p.pow(2).mean() + delta**2)
    scale = update_rms / param_rms_smooth
    scale = torch.clamp_max(scale, max_scale)
    if cautious_wd:
        wd_mask = (ref * p >= 0).to(ref.dtype)
        ref = ref + weight_decay * scale * p * wd_mask
    else:
        ref = ref + weight_decay * scale * p

    out = _apply_decoupled_wd(
        update_final, p, weight_decay, cautious_wd,
        relative_wd=True, relative_wd_delta=delta, max_scale=max_scale, eps=EPS,
        cautious_update=cautious_update, pre_mask_rms=None,
    )
    torch.testing.assert_close(out, ref)


@pytest.mark.parametrize("cautious_wd", [False, True])
def test_apply_decoupled_wd_absolute_matches_original(cautious_wd):
    """Absolute weight decay matches the original inline block."""
    _requires_cuda()
    torch.manual_seed(3)
    p = torch.randn(16, device=DEVICE)
    update_final = torch.randn(16, device=DEVICE)
    weight_decay = 0.2

    ref = update_final
    if cautious_wd:
        wd_mask = (ref * p >= 0).to(ref.dtype)
        ref = ref + weight_decay * p * wd_mask
    else:
        ref = ref + weight_decay * p

    out = _apply_decoupled_wd(
        update_final, p, weight_decay, cautious_wd,
        relative_wd=False, relative_wd_delta=1e-3, max_scale=float("inf"), eps=EPS,
    )
    torch.testing.assert_close(out, ref)


def test_apply_decoupled_wd_zero_wd_is_identity():
    """weight_decay=0 must return the update unchanged (no RMS computation)."""
    _requires_cuda()
    update = torch.randn(8, device=DEVICE)
    p = torch.randn(8, device=DEVICE)
    out = _apply_decoupled_wd(
        update, p, 0.0, cautious_wd=True,
        relative_wd=True, relative_wd_delta=1e-3, max_scale=1.0, eps=EPS,
    )
    assert out is update


@pytest.mark.parametrize("foreach", [False, True])
def test_learning_rate_helpers_match_tensor_reference(foreach):
    """CUDA tensor learning rates are applied without converting to Python.

    This covers the native and foreach paths used by ``step()``. The expected
    result is formed with device arithmetic, so the test also catches an
    accidental host scalar conversion in future changes.
    """
    _requires_cuda()
    torch.manual_seed(5)
    lr = torch.tensor(0.25, device=DEVICE)
    parameters = [torch.randn(8, device=DEVICE), torch.randn(4, device=DEVICE)]
    updates = [torch.randn_like(parameter) for parameter in parameters]
    expected = [parameter - lr * update for parameter, update in zip(parameters, updates)]

    if foreach:
        _foreach_apply_lr_(parameters, updates, lr)
    else:
        for parameter, update in zip(parameters, updates):
            _apply_lr_update_(parameter, update, lr)

    for parameter, expected_parameter in zip(parameters, expected):
        torch.testing.assert_close(parameter, expected_parameter)


@pytest.mark.parametrize("foreach", [False, True])
def test_learning_rate_helpers_preserve_python_float_path(foreach):
    """Python-float learning rates retain the standard optimizer semantics."""
    _requires_cuda()
    torch.manual_seed(6)
    lr = 0.125
    parameters = [torch.randn(8, device=DEVICE), torch.randn(4, device=DEVICE)]
    updates = [torch.randn_like(parameter) for parameter in parameters]
    expected = [parameter - lr * update for parameter, update in zip(parameters, updates)]

    if foreach:
        _foreach_apply_lr_(parameters, updates, lr)
    else:
        for parameter, update in zip(parameters, updates):
            _apply_lr_update_(parameter, update, lr)

    for parameter, expected_parameter in zip(parameters, expected):
        torch.testing.assert_close(parameter, expected_parameter)


def test_apply_decoupled_wd_pre_mask_rms_reuse_is_exact():
    """When cautious masking is off, passing pre_mask_rms must give the same
    result as recomputing the update RMS (the unmasked update was already
    rescaled to exactly that RMS)."""
    _requires_cuda()
    torch.manual_seed(4)
    p = torch.randn(16, device=DEVICE)
    update = torch.randn(16, device=DEVICE)
    target_rms = torch.sqrt(update.pow(2).mean() + EPS)

    recomputed = _apply_decoupled_wd(
        update, p, 0.1, cautious_wd=False,
        relative_wd=True, relative_wd_delta=1e-3, max_scale=1e6, eps=EPS,
        cautious_update=False, pre_mask_rms=None,
    )
    reused = _apply_decoupled_wd(
        update, p, 0.1, cautious_wd=False,
        relative_wd=True, relative_wd_delta=1e-3, max_scale=1e6, eps=EPS,
        cautious_update=False, pre_mask_rms=target_rms,
    )
    torch.testing.assert_close(reused, recomputed)


# ---------------------------------------------------------------------------
# Core equivalence tests vs. verbatim pre-refactor reference implementations
# ---------------------------------------------------------------------------


def _reference_2d_core(
    p_2d, g_2d, momentum, sign_momentum, exp_avg_sq_row, exp_avg_sq_col, row_var,
    beta1, beta2, beta3, weight_decay, max_scale, eps, step_t, sinkhorn_steps,
    cautious_update, cautious_wd, ortho_dtype, relative_wd, relative_wd_delta,
    warp=None, spectral_log_left=None, spectral_log_right=None,
    spectral_bilateral=True, spectral_log_bound=1.0,
):
    """Verbatim pre-refactor 2D core (warp=None and spectral_log_left=None give
    the plain AINOOpt path; warp gives the dense path; spectral gives the FFT path)."""
    g_norm = g_2d
    for _ in range(sinkhorn_steps):
        row_rms = torch.sqrt(g_norm.pow(2).mean(dim=-1, keepdim=True) + eps)
        g_norm = g_norm / row_rms
        col_rms = torch.sqrt(g_norm.pow(2).mean(dim=-2, keepdim=True) + eps)
        g_norm = g_norm / col_rms

    beta1_pow = beta1 ** step_t
    poly_beta1 = torch.where(
        step_t > 1.0, (beta1_pow - beta1) / (beta1_pow - 1.0), torch.zeros_like(beta1_pow)
    )
    beta2_pow = beta2 ** step_t
    poly_beta2 = torch.where(
        step_t > 1.0, (beta2_pow - beta2) / (beta2_pow - 1.0), torch.zeros_like(beta2_pow)
    )
    beta3_pow = beta3 ** step_t
    poly_beta3 = torch.where(
        step_t > 1.0, (beta3_pow - beta3) / (beta3_pow - 1.0), torch.zeros_like(beta3_pow)
    )

    g_sign = g_norm.sign()
    sign_momentum.lerp_(g_sign, 1.0 - beta1)

    diff_sq = (g_norm - momentum).pow(2)
    exp_avg_sq_row.lerp_(diff_sq.mean(dim=-1, keepdim=True), 1.0 - poly_beta2)
    exp_avg_sq_col.lerp_(diff_sq.mean(dim=-2, keepdim=True), 1.0 - poly_beta2)

    r_factor = (exp_avg_sq_row + eps).sqrt()
    c_factor = (
        (exp_avg_sq_col + eps) / (exp_avg_sq_col.mean(dim=-1, keepdim=True) + eps)
    ).sqrt()
    denom = r_factor * c_factor

    momentum.lerp_(g_norm, 1.0 - poly_beta1)

    update = sign_momentum * (momentum.abs() / denom)

    if spectral_log_left is not None:
        if spectral_bilateral and spectral_log_right.numel() > 0:
            warped = _spectral_apply(
                update, spectral_log_left, spectral_log_right, spectral_log_bound
            )
        else:
            warped = _spectral_apply(update, spectral_log_left, None, spectral_log_bound)
        update_warped = torch.where(step_t <= 1.0, update, warped)
    elif warp is not None:
        update_warped = update + warp.float() @ update
    else:
        update_warped = update

    O = gram_newton_schulz_2step(update_warped, eps=1e-7, ortho_dtype=ortho_dtype)

    row_sq = O.pow(2).mean(dim=-1)
    row_var.lerp_(row_sq, 1.0 - poly_beta3)
    O_norm = O / torch.sqrt(row_var.clamp_min(eps).unsqueeze(-1))

    target_rms = torch.sqrt(update_warped.pow(2).mean() + eps)
    current_rms = torch.sqrt(O_norm.pow(2).mean() + eps)
    update_final = O_norm * (target_rms / (current_rms + eps))

    if cautious_update:
        mask = (g_2d * update_final > 0).to(update_final.dtype)
        mask_mean = mask.mean().clamp_min(1e-3)
        update_final = update_final * mask / mask_mean

    if weight_decay != 0:
        if relative_wd:
            update_rms = torch.sqrt(update_final.pow(2).mean() + eps)
            param_rms_smooth = torch.sqrt(p_2d.pow(2).mean() + relative_wd_delta**2)
            scale = update_rms / param_rms_smooth
            scale = torch.clamp_max(scale, max_scale)
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


def _make_2d_state(m, n, seed):
    torch.manual_seed(seed)
    return dict(
        p=torch.randn(m, n, device=DEVICE),
        g=torch.randn(m, n, device=DEVICE),
        momentum=torch.randn(m, n, device=DEVICE) * 0.1,
        sign_momentum=torch.randn(m, n, device=DEVICE).sign() * 0.5,
        exp_avg_sq_row=torch.rand(m, 1, device=DEVICE) * 0.1,
        exp_avg_sq_col=torch.rand(1, n, device=DEVICE) * 0.1,
        row_var=torch.rand(m, device=DEVICE) * 0.1,
    )


def _clone_state(s):
    return {k: v.clone() for k, v in s.items()}


@pytest.mark.parametrize("cautious_update", [False, True])
@pytest.mark.parametrize("cautious_wd", [False, True])
@pytest.mark.parametrize("relative_wd", [False, True])
@pytest.mark.parametrize("step", [1.0, 7.0])
def test_aino_core_2d_matches_reference(
    cautious_update, cautious_wd, relative_wd, step
):
    """The refactored plain-AINOOpt 2D core is bitwise-identical to the
    pre-refactor implementation across all flag combinations."""
    _requires_cuda()
    base = _make_2d_state(8, 12, seed=10)
    step_t = torch.tensor(step, device=DEVICE)
    kwargs = dict(
        beta1=BETAS[0], beta2=BETAS[1], beta3=BETAS[2],
        weight_decay=0.1, max_scale=25.0, eps=EPS, step_t=step_t,
        sinkhorn_steps=3, cautious_update=cautious_update, cautious_wd=cautious_wd,
        ortho_dtype=torch.bfloat16, relative_wd=relative_wd,
        relative_wd_delta=1e-3,
    )

    s_ref = _clone_state(base)
    ref_out, ref_update = _reference_2d_core(
        s_ref["p"], s_ref["g"], s_ref["momentum"], s_ref["sign_momentum"],
        s_ref["exp_avg_sq_row"], s_ref["exp_avg_sq_col"], s_ref["row_var"], **kwargs,
    )

    s_new = _clone_state(base)
    new_out = WarpAINO._aino_step_core_2d(
        s_new["p"], s_new["g"], s_new["momentum"], s_new["sign_momentum"],
        s_new["exp_avg_sq_row"], s_new["exp_avg_sq_col"], s_new["row_var"], **kwargs,
    )

    torch.testing.assert_close(new_out, ref_out)
    # All mutated optimizer state must match too
    for k in ("momentum", "sign_momentum", "exp_avg_sq_row", "exp_avg_sq_col", "row_var"):
        torch.testing.assert_close(s_new[k], s_ref[k], msg=f"state mismatch: {k}")


@pytest.mark.parametrize("cautious_update", [False, True])
@pytest.mark.parametrize("relative_wd", [False, True])
def test_warp_core_2d_matches_reference(cautious_update, relative_wd):
    """The refactored dense-warp 2D core matches the pre-refactor implementation."""
    _requires_cuda()
    base = _make_2d_state(8, 12, seed=11)
    torch.manual_seed(12)
    warp = (torch.randn(8, 8, device=DEVICE) * 0.05).to(torch.bfloat16)
    step_t = torch.tensor(5.0, device=DEVICE)
    kwargs = dict(
        beta1=BETAS[0], beta2=BETAS[1], beta3=BETAS[2],
        weight_decay=0.1, max_scale=25.0, eps=EPS, step_t=step_t,
        sinkhorn_steps=3, cautious_update=cautious_update, cautious_wd=True,
        ortho_dtype=torch.bfloat16, relative_wd=relative_wd,
        relative_wd_delta=1e-3,
    )

    s_ref = _clone_state(base)
    ref_out, ref_update = _reference_2d_core(
        s_ref["p"], s_ref["g"], s_ref["momentum"], s_ref["sign_momentum"],
        s_ref["exp_avg_sq_row"], s_ref["exp_avg_sq_col"], s_ref["row_var"],
        warp=warp, **kwargs,
    )

    s_new = _clone_state(base)
    new_out, new_update = WarpAINO._WarpAINO_step_core_2d(
        s_new["p"], s_new["g"], warp, s_new["momentum"], s_new["sign_momentum"],
        s_new["exp_avg_sq_row"], s_new["exp_avg_sq_col"], s_new["row_var"], **kwargs,
    )

    torch.testing.assert_close(new_out, ref_out)
    torch.testing.assert_close(new_update, ref_update)
    for k in ("momentum", "sign_momentum", "exp_avg_sq_row", "exp_avg_sq_col", "row_var"):
        torch.testing.assert_close(s_new[k], s_ref[k], msg=f"state mismatch: {k}")


@pytest.mark.parametrize("bilateral", [False, True])
@pytest.mark.parametrize("step", [1.0, 7.0])
def test_spectral_core_2d_matches_reference(bilateral, step):
    """The refactored spectral 2D core matches the pre-refactor implementation,
    including the first-step identity guard."""
    _requires_cuda()
    base = _make_2d_state(8, 12, seed=13)
    torch.manual_seed(14)
    log_left = (torch.randn(8, device=DEVICE) * 0.3)
    log_right = (torch.randn(12, device=DEVICE) * 0.3) if bilateral else torch.empty(0, device=DEVICE)
    step_t = torch.tensor(step, device=DEVICE)
    kwargs = dict(
        beta1=BETAS[0], beta2=BETAS[1], beta3=BETAS[2],
        weight_decay=0.1, max_scale=25.0, eps=EPS, step_t=step_t,
        sinkhorn_steps=3, cautious_update=True, cautious_wd=True,
        ortho_dtype=torch.bfloat16, relative_wd=True, relative_wd_delta=1e-3,
    )

    s_ref = _clone_state(base)
    ref_out, ref_update = _reference_2d_core(
        s_ref["p"], s_ref["g"], s_ref["momentum"], s_ref["sign_momentum"],
        s_ref["exp_avg_sq_row"], s_ref["exp_avg_sq_col"], s_ref["row_var"],
        spectral_log_left=log_left, spectral_log_right=log_right,
        spectral_bilateral=True, spectral_log_bound=1.0, **kwargs,
    )

    s_new = _clone_state(base)
    new_out, new_update = WarpAINO._spectral_step_core_2d(
        s_new["p"], s_new["g"], log_left, log_right, s_new["momentum"],
        s_new["sign_momentum"], s_new["exp_avg_sq_row"], s_new["exp_avg_sq_col"],
        s_new["row_var"], spectral_bilateral=True, spectral_log_bound=1.0, **kwargs,
    )

    torch.testing.assert_close(new_out, ref_out)
    torch.testing.assert_close(new_update, ref_update)
    for k in ("momentum", "sign_momentum", "exp_avg_sq_row", "exp_avg_sq_col", "row_var"):
        torch.testing.assert_close(s_new[k], s_ref[k], msg=f"state mismatch: {k}")


def _reference_1d_core(
    p_data, g_data, momentum, sign_momentum, exp_avg_sq, row_var,
    beta1, beta2, beta3, weight_decay, max_scale, eps, step_t,
    cautious_update, cautious_wd, relative_wd, relative_wd_delta,
    warp=None, spectral_log_left=None, spectral_log_bound=1.0,
):
    """Verbatim pre-refactor 1D core."""
    grad_rms = torch.sqrt(g_data.pow(2).mean() + eps)
    g_norm = g_data / grad_rms

    beta1_pow = beta1 ** step_t
    poly_beta1 = torch.where(
        step_t > 1.0, (beta1_pow - beta1) / (beta1_pow - 1.0), torch.zeros_like(beta1_pow)
    )
    beta2_pow = beta2 ** step_t
    poly_beta2 = torch.where(
        step_t > 1.0, (beta2_pow - beta2) / (beta2_pow - 1.0), torch.zeros_like(beta2_pow)
    )
    beta3_pow = beta3 ** step_t
    poly_beta3 = torch.where(
        step_t > 1.0, (beta3_pow - beta3) / (beta3_pow - 1.0), torch.zeros_like(beta3_pow)
    )

    g_sign = g_norm.sign()
    sign_momentum.lerp_(g_sign, 1.0 - beta1)

    diff_sq = (g_norm - momentum).pow(2)
    exp_avg_sq.lerp_(diff_sq, 1.0 - poly_beta2)
    denom = exp_avg_sq.sqrt().clamp_min_(eps)

    momentum.lerp_(g_norm, 1.0 - poly_beta1)

    update = sign_momentum * (momentum.abs() / denom)

    if spectral_log_left is not None:
        update_2d = update.reshape(-1, 1)
        warped = _spectral_apply(update_2d, spectral_log_left, None, spectral_log_bound)
        O = torch.where(step_t <= 1.0, update_2d, warped).reshape(update.shape)
    elif warp is not None:
        update_2d = update.reshape(-1, 1)
        O = (update_2d + warp.float() @ update_2d).reshape(update.shape)
    else:
        O = update

    var_sq = O.pow(2).mean()
    row_var.lerp_(var_sq, 1.0 - poly_beta3)
    O_norm = O / torch.sqrt(row_var.clamp_min(eps))

    target_rms = torch.sqrt(O.pow(2).mean() + eps)
    current_rms = torch.sqrt(O_norm.pow(2).mean() + eps)
    update_final = O_norm * (target_rms / (current_rms + eps))

    if cautious_update:
        mask = (g_data * update_final > 0).to(update_final.dtype)
        mask_mean = mask.mean().clamp_min(1e-3)
        update_final = update_final * mask / mask_mean

    if weight_decay != 0:
        if relative_wd:
            update_rms = torch.sqrt(update_final.pow(2).mean() + eps)
            param_rms_smooth = torch.sqrt(p_data.pow(2).mean() + relative_wd_delta**2)
            scale = update_rms / param_rms_smooth
            scale = torch.clamp_max(scale, max_scale)
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


def _make_1d_state(n, seed):
    torch.manual_seed(seed)
    return dict(
        p=torch.randn(n, device=DEVICE),
        g=torch.randn(n, device=DEVICE),
        momentum=torch.randn(n, device=DEVICE) * 0.1,
        sign_momentum=torch.randn(n, device=DEVICE).sign() * 0.5,
        exp_avg_sq=torch.rand(n, device=DEVICE) * 0.1,
        row_var=torch.rand((), device=DEVICE) * 0.1,
    )


@pytest.mark.parametrize("mode", ["plain", "dense", "spectral"])
@pytest.mark.parametrize("relative_wd", [False, True])
def test_1d_cores_match_reference(mode, relative_wd):
    """All three refactored 1D cores match the pre-refactor implementation."""
    _requires_cuda()
    base = _make_1d_state(16, seed=20)
    torch.manual_seed(21)
    warp = (torch.randn(16, 16, device=DEVICE) * 0.05).to(torch.bfloat16)
    log_left = torch.randn(16, device=DEVICE) * 0.3
    step_t = torch.tensor(4.0, device=DEVICE)
    kwargs = dict(
        beta1=BETAS[0], beta2=BETAS[1], beta3=BETAS[2],
        weight_decay=0.1, max_scale=25.0, eps=EPS, step_t=step_t,
        cautious_update=True, cautious_wd=True,
        relative_wd=relative_wd, relative_wd_delta=1e-3,
    )

    s_ref = _clone_state(base)
    extra = {}
    if mode == "dense":
        extra["warp"] = warp
    elif mode == "spectral":
        extra["spectral_log_left"] = log_left
        extra["spectral_log_bound"] = 1.0
    ref_out, ref_update = _reference_1d_core(
        s_ref["p"], s_ref["g"], s_ref["momentum"], s_ref["sign_momentum"],
        s_ref["exp_avg_sq"], s_ref["row_var"], **extra, **kwargs,
    )

    s_new = _clone_state(base)
    if mode == "plain":
        # The plain core returns a single tensor (no crafted update is needed
        # without a warp meta-objective).
        new_out = WarpAINO._aino_step_core_1d(
            s_new["p"], s_new["g"], s_new["momentum"], s_new["sign_momentum"],
            s_new["exp_avg_sq"], s_new["row_var"], **kwargs,
        )
        new_update = ref_update
    elif mode == "dense":
        new_out, new_update = WarpAINO._WarpAINO_step_core_1d(
            s_new["p"], s_new["g"], warp, s_new["momentum"], s_new["sign_momentum"],
            s_new["exp_avg_sq"], s_new["row_var"], **kwargs,
        )
    else:
        new_out, new_update = WarpAINO._spectral_step_core_1d(
            s_new["p"], s_new["g"], log_left, s_new["momentum"], s_new["sign_momentum"],
            s_new["exp_avg_sq"], s_new["row_var"], spectral_log_bound=1.0, **kwargs,
        )

    torch.testing.assert_close(new_out, ref_out)
    torch.testing.assert_close(new_update, ref_update)
    for k in ("momentum", "sign_momentum", "exp_avg_sq", "row_var"):
        torch.testing.assert_close(s_new[k], s_ref[k], msg=f"state mismatch: {k}")


# ---------------------------------------------------------------------------
# Full-optimizer smoke tests (all dispatch paths through the shared helpers)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("foreach", [False, True])
@pytest.mark.parametrize("warp_mode", ["dense", "spectral"])
@pytest.mark.parametrize("meta_lr", [0.0, 1e-2])
def test_full_optimizer_smoke_cuda(foreach, warp_mode, meta_lr):
    """WarpAINO end-to-end steps (native + foreach, warp on/off) stay finite
    and update parameters, exercising every refactored core through step()."""
    _requires_cuda()
    torch.manual_seed(30)

    lin = torch.nn.Linear(8, 4, device=DEVICE)
    bias_vec = torch.nn.Parameter(torch.randn(6, device=DEVICE))
    params = list(lin.parameters()) + [bias_vec]

    opt = WarpAINO(
        params,
        lr=1e-3,
        weight_decay=1e-2,
        relative_wd=True,
        meta_lr=meta_lr,
        warp_mode=warp_mode,
        foreach=foreach,
    )

    before = [p.detach().clone() for p in params]
    for _ in range(3):
        x = torch.randn(4, 8, device=DEVICE)
        loss = lin(x).square().mean() + bias_vec.square().mean()
        loss.backward()
        opt.step()
        opt.zero_grad()

    for b, p in zip(before, params):
        assert torch.isfinite(p.data).all()
        assert not torch.equal(b, p.data)


def test_full_optimizer_compile_step_cuda():
    """The torch.compile'd cores still trace through the extracted helpers."""
    _requires_cuda()
    torch.manual_seed(31)

    lin = torch.nn.Linear(8, 4, device=DEVICE)
    opt = WarpAINO(
        list(lin.parameters()),
        lr=1e-3,
        weight_decay=1e-2,
        relative_wd=True,
        compile_step=True,
        warp_mode="spectral",
    )

    for _ in range(2):
        x = torch.randn(4, 8, device=DEVICE)
        lin(x).square().mean().backward()
        opt.step()
        opt.zero_grad()

    assert torch.isfinite(lin.weight).all()
    assert torch.isfinite(lin.bias).all()


def test_full_optimizer_compile_step_tensor_lr_cuda():
    """Compiled cores must not branch on a tensor-valued learning rate
    (covers the scenario from test_warpaino_relative_wd.py, which cannot be
    collected in environments without pytorch_optimizer)."""
    _requires_cuda()
    torch.manual_seed(32)

    lin = torch.nn.Linear(8, 4, device=DEVICE)
    opt = WarpAINO(
        list(lin.parameters()),
        lr=1e-3,
        weight_decay=1e-2,
        relative_wd=True,
        compile_step=True,
        warp_mode="spectral",
    )

    # Simulate accelerate/scheduler passing lr as a 0-dim CUDA tensor.
    for group in opt.param_groups:
        group["lr"] = torch.tensor(group["lr"], device=DEVICE)

    for _ in range(2):
        x = torch.randn(4, 8, device=DEVICE)
        lin(x).square().mean().backward()
        opt.step()
        opt.zero_grad()

    assert torch.isfinite(lin.weight).all()
    assert torch.isfinite(lin.bias).all()


@pytest.mark.parametrize("foreach", [False, True])
@pytest.mark.parametrize("shape", [(16,), (4, 4)])
def test_fp16_parameters_use_rounding_compensation_cuda(foreach, shape):
    """Small FP16 updates accumulate through rounding compensation.

    A one-step update at this learning rate is below the FP16 spacing near
    one, so the public parameter remains unchanged while its FP32 Kahan-style
    compensation records the lost rounding error. This exercises both
    dimensionality paths and both optimizer dispatch modes without storing a
    duplicate FP32 parameter.
    """
    _requires_cuda()
    torch.manual_seed(33)
    parameter = torch.nn.Parameter(torch.ones(shape, device=DEVICE, dtype=torch.float16))
    initial = parameter.detach().float().clone()
    optimizer = WarpAINO(
        [parameter],
        lr=1e-5,
        weight_decay=0.0,
        meta_lr=0.0,
        kahan_sum=True,
        foreach=foreach,
    )

    parameter.grad = torch.ones_like(parameter)
    optimizer.step()

    state = optimizer.state[parameter]
    assert "master_param" not in state
    compensation = state["param_compensation"]
    assert compensation.dtype is torch.float32
    assert compensation.device == parameter.device
    torch.testing.assert_close(parameter.float(), initial)
    assert torch.any(compensation != 0)


@pytest.mark.parametrize("foreach", [False, True])
def test_fp16_rounding_compensation_is_disabled_by_default(foreach):
    """The default configuration does not allocate compensation state."""
    _requires_cuda()
    parameter = torch.nn.Parameter(
        torch.ones(16, device=DEVICE, dtype=torch.float16)
    )
    optimizer = WarpAINO(
        [parameter],
        lr=1e-5,
        weight_decay=0.0,
        meta_lr=0.0,
        foreach=foreach,
    )

    parameter.grad = torch.ones_like(parameter)
    optimizer.step()

    state = optimizer.state[parameter]
    assert state.get("param_compensation") is None


def test_meta_ema_reuses_previous_update_buffer():
    """The EMA target updates the existing previous-update buffer in-place."""
    _requires_cuda()
    previous = torch.randn(8, 4, device=DEVICE)
    current = torch.randn(8, 4, device=DEVICE)
    state = {"prev_update": previous.clone()}
    previous_reference = state["prev_update"]
    beta = 0.75
    expected = beta * previous_reference.clone() + (1.0 - beta) * current

    _update_meta_history(state, current, meta_ema=True, meta_ema_beta=beta)

    assert state["prev_update"] is previous_reference
    torch.testing.assert_close(state["prev_update"], expected)


def test_meta_ema_disabled_preserves_latest_update_semantics():
    """Disabled EMA stores the current update directly, as in legacy behavior."""
    _requires_cuda()
    current = torch.randn(8, 4, device=DEVICE)
    state = {"prev_update": torch.randn(8, 4, device=DEVICE)}
    _update_meta_history(state, current, meta_ema=False, meta_ema_beta=0.98)
    torch.testing.assert_close(state["prev_update"], current)
    assert state["prev_update"] is not current


def test_nesterov_sign_momentum_changes_lookahead_direction():
    """Nesterov sign mode can look ahead of a still-negative sign EMA."""
    _requires_cuda()
    sign_momentum = torch.full((16,), -0.05, device=DEVICE)
    g_sign = torch.ones(16, device=DEVICE)
    ordinary = _select_sign_momentum(sign_momentum, g_sign, 0.9, False)
    nesterov = _select_sign_momentum(sign_momentum, g_sign, 0.9, True)

    assert (ordinary < 0).all()
    assert (nesterov > 0).all()


def test_grokfast_filter_matches_first_step_formula():
    """GrokFast updates the EMA before adding its amplified slow component."""
    _requires_cuda()
    gradient = torch.ones(16, device=DEVICE)
    state = {"step": 1}
    group = {
        "grokfast": True,
        "grokfast_after_step": 0,
        "grokfast_alpha": 0.75,
        "grokfast_lamb": 2.0,
    }

    filtered = _grokfast_filter(state, gradient, group)

    expected_ema = torch.full_like(gradient, 0.25)
    expected = gradient + 2.0 * expected_ema
    torch.testing.assert_close(state["grokfast_ema"], expected_ema)
    torch.testing.assert_close(filtered, expected)


def test_grokfast_after_step_keeps_filter_inactive():
    """The after-step gate avoids allocating or updating GrokFast state early."""
    _requires_cuda()
    gradient = torch.randn(16, device=DEVICE)
    state = {"step": 2}
    group = {
        "grokfast": True,
        "grokfast_after_step": 2,
        "grokfast_alpha": 0.98,
        "grokfast_lamb": 2.0,
    }

    filtered = _grokfast_filter(state, gradient, group)

    assert filtered is gradient
    assert "grokfast_ema" not in state


def test_rms_clip_limits_rescale_ratio():
    """RMS clipping limits pathological post-orthogonalization amplification."""
    _requires_cuda()
    p = torch.randn(8, 8, device=DEVICE)
    g = torch.randn(8, 8, device=DEVICE)
    step_t = torch.tensor(2.0, device=DEVICE)

    def run(rms_clip):
        momentum = torch.zeros_like(p)
        sign_momentum = torch.zeros_like(p)
        row = torch.zeros(8, 1, device=DEVICE)
        col = torch.zeros(1, 8, device=DEVICE)
        row_var = torch.full((8,), 1e6, device=DEVICE)
        return WarpAINO._aino_step_core_2d(
            p, g, momentum, sign_momentum, row, col, row_var,
            beta1=0.95, beta2=0.95, beta3=0.999,
            weight_decay=0.0, max_scale=float("inf"), eps=1e-16,
            step_t=step_t, sinkhorn_steps=2, cautious_update=False,
            cautious_wd=False, ortho_dtype=torch.bfloat16,
            rms_clip=rms_clip, rms_clip_max=0.5,
        )

    unclipped = run(False)
    clipped = run(True)
    assert torch.isfinite(clipped).all()
    assert clipped.norm() < unclipped.norm()


@pytest.mark.parametrize("foreach", [False, True])
def test_all_optional_features_smoke_cuda(foreach):
    """All requested techniques compose through the normal optimizer path."""
    _requires_cuda()
    torch.manual_seed(35)
    lin = torch.nn.Linear(8, 4, device=DEVICE)
    vector = torch.nn.Parameter(torch.randn(6, device=DEVICE))
    optimizer = WarpAINO(
        list(lin.parameters()) + [vector],
        lr=1e-3,
        meta_lr=1e-2,
        meta_ema=True,
        meta_ema_beta=0.98,
        nesterov_sign=True,
        rms_clip=True,
        rms_clip_max=10.0,
        grokfast=True,
        grokfast_alpha=0.98,
        grokfast_lamb=2.0,
        foreach=foreach,
        warp_mode="spectral",
    )

    for _ in range(3):
        loss = lin(torch.randn(4, 8, device=DEVICE)).square().mean()
        loss = loss + vector.square().mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    for parameter in list(lin.parameters()) + [vector]:
        assert torch.isfinite(parameter).all()
    assert "grokfast_ema" in optimizer.state[lin.weight]
    assert "prev_update" in optimizer.state[lin.weight]


def test_all_optional_features_compile_step_cuda():
    """Optional core features trace successfully with compiled stepping."""
    _requires_cuda()
    torch.manual_seed(36)
    lin = torch.nn.Linear(8, 4, device=DEVICE)
    optimizer = WarpAINO(
        list(lin.parameters()),
        lr=1e-3,
        meta_lr=1e-2,
        meta_ema=True,
        nesterov_sign=True,
        rms_clip=True,
        rms_clip_max=10.0,
        grokfast=True,
        compile_step=True,
        warp_mode="spectral",
    )

    for _ in range(2):
        lin(torch.randn(4, 8, device=DEVICE)).square().mean().backward()
        optimizer.step()
        optimizer.zero_grad()

    assert torch.isfinite(lin.weight).all()


@pytest.mark.parametrize("bilateral", [False, True])
@pytest.mark.parametrize("shape", [(7, 6), (8, 5)])
def test_spectral_rfft_matches_full_fft_reference(bilateral, shape):
    """The half-spectrum spectral operator matches the former full FFT math."""
    _requires_cuda()
    torch.manual_seed(34)
    update = torch.randn(shape, device=DEVICE)
    log_left = torch.randn(shape[0], device=DEVICE) * 0.3
    log_right = torch.randn(shape[1], device=DEVICE) * 0.3
    bound = 0.8

    left_gain = _spectral_log_gain(log_left, bound)
    if bilateral:
        right_gain = _spectral_log_gain(log_right, bound)
        spectrum = torch.fft.fft2(update, dim=(0, 1), norm="ortho")
        delta = torch.fft.ifft2(
            spectrum * (left_gain[:, None] * right_gain[None, :] - 1.0),
            dim=(0, 1),
            norm="ortho",
        ).real
    else:
        spectrum = torch.fft.fft(update, dim=0, norm="ortho")
        delta = torch.fft.ifft(
            spectrum * (left_gain[:, None] - 1.0), dim=0, norm="ortho"
        ).real
    expected = update + delta

    actual = _spectral_apply(
        update,
        log_left,
        log_right if bilateral else None,
        bound,
    )
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"lr": -1.0},
        {"betas": (0.9, 0.9)},
        {"betas": (0.9, float("nan"), 0.9)},
        {"weight_decay": -1.0},
        {"eps": float("nan")},
        {"meta_lr": -1.0},
        {"meta_wd": float("inf")},
        {"kahan_sum": 1},
        {"spectral_bilateral": 1},
        {"sinkhorn_steps": -1},
        {"sinkhorn_steps": 1.5},
        {"spectral_log_bound": -1.0},
        {"relative_wd_delta": -1.0},
        {"relative_wd_max_contraction": 0.0},
        {"relative_wd_max_contraction": 1.1},
        {"meta_ema_beta": 1.0},
        {"nesterov_sign": 1},
        {"rms_clip_max": 0.0},
        {"grokfast": 1},
        {"grokfast_alpha": 1.0},
        {"grokfast_after_step": -1},
    ],
)
def test_constructor_rejects_invalid_options_cuda(kwargs):
    """Invalid scalar, tensor, boolean, and range options fail early."""
    _requires_cuda()
    parameter = torch.nn.Parameter(torch.zeros(4, device=DEVICE))
    with pytest.raises(ValueError):
        WarpAINO([parameter], **kwargs)


@pytest.mark.parametrize("foreach", [False, True])
def test_constructor_accepts_float_and_cuda_tensor_lr(foreach):
    """Both supported learning-rate representations work when passed to init."""
    _requires_cuda()
    parameter = torch.nn.Parameter(torch.ones(16, device=DEVICE))
    lr = torch.tensor(1e-4, device=DEVICE)
    optimizer = WarpAINO(
        [parameter], lr=lr, meta_lr=0.0, foreach=foreach
    )
    parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    assert torch.isfinite(parameter).all()
    assert optimizer.param_groups[0]["lr"].device == lr.device


def test_constructor_rejects_invalid_cuda_tensor_lr():
    """Learning-rate tensors must be scalar, floating-point, finite, and non-negative."""
    _requires_cuda()
    parameter = torch.nn.Parameter(torch.zeros(4, device=DEVICE))
    with pytest.raises(ValueError):
        WarpAINO([parameter], lr=torch.tensor([1e-3], device=DEVICE))
    with pytest.raises(ValueError):
        WarpAINO([parameter], lr=torch.tensor(-1e-3, device=DEVICE))
