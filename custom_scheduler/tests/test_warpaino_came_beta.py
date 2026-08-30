"""Tests for the dedicated ``came_beta`` parameter of WarpAINO.

``came_confidence`` previously reused ``beta3`` (the third entry of
``betas``) for its factorized residual EMA. A separate ``came_beta``
hyperparameter (default 0.9999) now controls that EMA independently.

These tests verify, on CUDA:

* the default value and constructor validation of ``came_beta``,
* that ``came_beta`` actually drives the CAME residual EMA state,
* that it is threaded through all three 2D cores (plain, dense warp,
  spectral warp) and the foreach path,
* that it is inert when ``came_confidence`` is disabled,
* that ``came_beta=beta3`` reproduces the residual EMA of a manual
  reference implementation (the pre-change behavior).
"""

import math
import os
import sys

import pytest
import torch

# Ensure the custom_scheduler package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    from LoraEasyCustomOptimizer.warpaino import (
        WarpAINO,
        _poly_beta,
        _sinkhorn_normalize,
    )
except ImportError:
    # The package __init__ pulls in optional heavy dependencies that may
    # be absent; warpaino.py itself only requires torch, so load it
    # directly as a fallback.
    import importlib.util

    _module_path = os.path.join(
        os.path.dirname(__file__), "..", "LoraEasyCustomOptimizer", "warpaino.py"
    )
    _spec = importlib.util.spec_from_file_location("warpaino_standalone", _module_path)
    _module = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_module)
    WarpAINO = _module.WarpAINO
    _poly_beta = _module._poly_beta
    _sinkhorn_normalize = _module._sinkhorn_normalize

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required for these tests"
)

DEVICE = torch.device("cuda")


def _make_opt(params, **kwargs):
    kwargs.setdefault("came_confidence", True)
    kwargs.setdefault("meta_lr", 0.0)
    return WarpAINO(params, **kwargs)


def _run_steps(opt, params, grads, steps):
    """Run ``steps`` optimizer steps; ``grads`` is a per-step gradient list
    applied to every parameter."""
    for t in range(steps):
        for p in params:
            p.grad = grads[t].clone()
        opt.step()


def _make_grads(shape, steps, seed=1234):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    return [
        torch.randn(*shape, device=DEVICE, generator=generator)
        for _ in range(steps)
    ]


def test_default_came_beta_is_0_9999():
    p = torch.nn.Parameter(torch.randn(8, 4, device=DEVICE))
    opt = WarpAINO([p], meta_lr=0.0)
    assert opt.param_groups[0]["came_beta"] == 0.9999


@pytest.mark.parametrize("bad", [1.0, -0.1, float("nan"), float("inf")])
def test_invalid_came_beta_raises(bad):
    p = torch.nn.Parameter(torch.randn(8, 4, device=DEVICE))
    with pytest.raises(ValueError):
        WarpAINO([p], came_beta=bad)


def test_came_beta_accepts_valid_values():
    p = torch.nn.Parameter(torch.randn(8, 4, device=DEVICE))
    opt = WarpAINO([p], came_beta=0.5, meta_lr=0.0)
    assert opt.param_groups[0]["came_beta"] == 0.5


def test_came_beta_changes_residual_state_and_updates():
    """Different came_beta values must produce different residual EMAs."""
    steps = 4
    init = torch.randn(8, 4, device=DEVICE, generator=torch.Generator(device="cuda").manual_seed(7))
    grads = _make_grads((8, 4), steps)

    states = {}
    finals = {}
    for came_beta in (0.5, 0.9999):
        p = torch.nn.Parameter(init.clone())
        opt = _make_opt([p], came_beta=came_beta)
        _run_steps(opt, [p], grads, steps)
        assert "exp_avg_res_row" in opt.state[p]
        states[came_beta] = opt.state[p]["exp_avg_res_row"].clone()
        finals[came_beta] = p.data.clone()

    assert not torch.allclose(states[0.5], states[0.9999])
    assert not torch.allclose(finals[0.5], finals[0.9999])


def test_came_beta_deterministic():
    """The same came_beta must reproduce identical results."""
    steps = 3
    init = torch.randn(8, 4, device=DEVICE, generator=torch.Generator(device="cuda").manual_seed(11))
    grads = _make_grads((8, 4), steps, seed=99)

    finals = []
    for _ in range(2):
        p = torch.nn.Parameter(init.clone())
        opt = _make_opt([p], came_beta=0.75)
        _run_steps(opt, [p], grads, steps)
        finals.append(p.data.clone())

    assert torch.allclose(finals[0], finals[1])


def test_came_beta_ignored_without_came_confidence():
    """came_beta must have no effect when came_confidence is disabled."""
    steps = 3
    init = torch.randn(8, 4, device=DEVICE, generator=torch.Generator(device="cuda").manual_seed(21))
    grads = _make_grads((8, 4), steps, seed=55)

    finals = []
    for came_beta in (0.5, 0.9999):
        p = torch.nn.Parameter(init.clone())
        opt = WarpAINO([p], came_confidence=False, came_beta=came_beta, meta_lr=0.0)
        _run_steps(opt, [p], grads, steps)
        assert "exp_avg_res_row" not in opt.state[p]
        finals.append(p.data.clone())

    assert torch.allclose(finals[0], finals[1])


def test_came_beta_threaded_through_dense_warp_core():
    steps = 3
    init = torch.randn(8, 4, device=DEVICE, generator=torch.Generator(device="cuda").manual_seed(31))
    grads = _make_grads((8, 4), steps, seed=77)

    finals = []
    for came_beta in (0.5, 0.9999):
        p = torch.nn.Parameter(init.clone())
        opt = WarpAINO(
            [p],
            came_confidence=True,
            came_beta=came_beta,
            warp_mode="dense",
            meta_lr=5e-2,
        )
        _run_steps(opt, [p], grads, steps)
        assert "warp" in opt.state[p]
        finals.append(p.data.clone())

    assert not torch.allclose(finals[0], finals[1])


def test_came_beta_threaded_through_spectral_warp_core():
    steps = 3
    init = torch.randn(8, 4, device=DEVICE, generator=torch.Generator(device="cuda").manual_seed(41))
    grads = _make_grads((8, 4), steps, seed=88)

    finals = []
    for came_beta in (0.5, 0.9999):
        p = torch.nn.Parameter(init.clone())
        opt = WarpAINO(
            [p],
            came_confidence=True,
            came_beta=came_beta,
            warp_mode="spectral",
            meta_lr=5e-2,
        )
        _run_steps(opt, [p], grads, steps)
        assert "spectral_log_left" in opt.state[p]
        finals.append(p.data.clone())

    assert not torch.allclose(finals[0], finals[1])


def test_came_beta_threaded_through_foreach_path():
    steps = 3
    init = torch.randn(8, 4, device=DEVICE, generator=torch.Generator(device="cuda").manual_seed(51))
    grads = _make_grads((8, 4), steps, seed=66)

    finals = []
    for came_beta in (0.5, 0.9999):
        p = torch.nn.Parameter(init.clone())
        opt = _make_opt([p], came_beta=came_beta, foreach=True)
        _run_steps(opt, [p], grads, steps)
        finals.append(p.data.clone())

    assert not torch.allclose(finals[0], finals[1])


def test_came_beta_matches_manual_residual_reference():
    """came_beta=beta3 must reproduce the pre-change residual EMA exactly.

    The reference below re-implements the 2D tracking recurrence with the
    residual EMA driven by ``_poly_beta(came_beta, step_t)``, which is what
    the optimizer did before the dedicated parameter existed (with
    ``came_beta`` hardcoded to ``beta3``).
    """
    steps = 4
    beta1, beta2, beta3 = 0.95, 0.95, 0.999
    came_beta = beta3  # pre-change behavior
    eps = 1e-16
    sinkhorn_steps = 5

    init = torch.randn(8, 4, device=DEVICE, generator=torch.Generator(device="cuda").manual_seed(61))
    grads = _make_grads((8, 4), steps, seed=44)

    p = torch.nn.Parameter(init.clone())
    opt = _make_opt([p], betas=(beta1, beta2, beta3), came_beta=came_beta)
    _run_steps(opt, [p], grads, steps)

    # Manual reference recurrence
    momentum = torch.zeros(8, 4, device=DEVICE)
    sign_momentum = torch.zeros(8, 4, device=DEVICE)
    exp_avg_sq_row = torch.zeros(8, 1, device=DEVICE)
    exp_avg_sq_col = torch.zeros(1, 4, device=DEVICE)
    exp_avg_res_row = torch.zeros(8, 1, device=DEVICE)

    for t, g in enumerate(grads, start=1):
        step_t = torch.tensor(float(t), device=DEVICE)
        g_norm = _sinkhorn_normalize(g, sinkhorn_steps, eps)
        poly_beta1 = _poly_beta(beta1, step_t)
        poly_beta2 = _poly_beta(beta2, step_t)
        poly_came_beta = _poly_beta(came_beta, step_t)

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

        preconditioned = g_norm / denom
        residual = (preconditioned - momentum).pow(2).add_(eps)
        exp_avg_res_row.lerp_(residual.mean(dim=-1, keepdim=True), 1.0 - poly_came_beta)

    assert torch.allclose(
        opt.state[p]["exp_avg_res_row"], exp_avg_res_row, rtol=1e-5, atol=1e-7
    )
    # Sanity: the poly correction is active at these steps (not a trivial EMA).
    # float32 precision in ``_poly_beta`` requires a relaxed tolerance.
    assert math.isclose(
        float(_poly_beta(came_beta, torch.tensor(2.0))),
        came_beta / (came_beta + 1.0),
        rel_tol=1e-4,
    )
