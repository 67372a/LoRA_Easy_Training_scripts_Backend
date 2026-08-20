"""Tests for WarpAINO RMS-relative (``relative_wd``) weight decay.

Validates the two hyperparameters added for relative weight decay:

* ``relative_wd_delta`` -- Pseudo-Huber smoothing of the parameter RMS in the
  weight-decay denominator (``param_rms_smooth = sqrt(rms_param^2 + delta^2)``).
* ``relative_wd_max_contraction`` -- discrete-time overshoot guard capping the
  effective decay multiplier to ``max_contraction / (lr * weight_decay)``.

The overshoot cap is precomputed outside the ``torch.compile``'d step cores so
the compiled graph never contains data-dependent control flow on the learning
rate (which can be a tensor when driven by schedulers/accelerate).
"""

import os
import sys

import pytest
import torch

# Ensure the custom_scheduler package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from LoraEasyCustomOptimizer.warpaino import WarpAINO, _relative_wd_max_scale


DEVICE = "cuda"


def _requires_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")


def _run_1d_core(
    p,
    g,
    weight_decay,
    max_scale,
    eps,
    delta,
    cautious_wd=False,
    cautious_update=False,
):
    """Run the plain AINO 1D core with fresh state and ``relative_wd=True``."""
    momentum = torch.zeros_like(p)
    sign_momentum = torch.zeros_like(p)
    exp_avg_sq = torch.zeros_like(p)
    row_var = torch.zeros((), device=p.device, dtype=torch.float32)
    step_t = torch.tensor(1.0, device=p.device)
    return WarpAINO._aino_step_core_1d(
        p,
        g,
        momentum,
        sign_momentum,
        exp_avg_sq,
        row_var,
        beta1=0.95,
        beta2=0.95,
        beta3=0.999,
        weight_decay=weight_decay,
        max_scale=max_scale,
        eps=eps,
        step_t=step_t,
        cautious_update=cautious_update,
        cautious_wd=cautious_wd,
        relative_wd=True,
        relative_wd_delta=delta,
    )


def _reference_wd(update_no_wd, p, weight_decay, lr, eps, delta, max_contraction):
    """Reference implementation of the relative weight decay term."""
    update_rms = torch.sqrt(update_no_wd.pow(2).mean() + eps)
    param_rms_smooth = torch.sqrt(p.pow(2).mean() + delta**2)
    scale = update_rms / param_rms_smooth
    if weight_decay > 0 and lr > 0:
        scale = torch.clamp_max(scale, max_contraction / (lr * weight_decay))
    return weight_decay * scale * p


def test_relative_wd_matches_reference():
    """The relative weight-decay term matches the Huber-smoothed formula."""
    _requires_cuda()
    torch.manual_seed(0)
    p = torch.randn(8, device=DEVICE)
    g = torch.randn(8, device=DEVICE)
    wd = 0.3
    lr = 0.1
    eps = 1e-8
    delta = 1e-2
    max_contraction = 0.99

    max_scale = _relative_wd_max_scale(lr, wd, max_contraction)
    update_no_wd = _run_1d_core(p, g, 0.0, max_scale, eps, delta)
    update_wd = _run_1d_core(p, g, wd, max_scale, eps, delta)

    expected_wd = _reference_wd(
        update_no_wd, p, wd, lr, eps, delta, max_contraction
    )
    torch.testing.assert_close(update_wd, update_no_wd + expected_wd)


def test_relative_wd_max_contraction_caps_scale():
    """``relative_wd_max_contraction`` caps the effective decay scale."""
    _requires_cuda()
    torch.manual_seed(1)
    # Small parameter norm -> large uncapped scale, forcing the cap to bind.
    p = torch.randn(16, device=DEVICE) * 0.01
    g = torch.randn(16, device=DEVICE)
    lr = 10.0
    wd = 10.0
    eps = 1e-8
    delta = 1e-4
    max_contraction = 0.1

    max_scale = _relative_wd_max_scale(lr, wd, max_contraction)
    update_no_wd = _run_1d_core(p, g, 0.0, max_scale, eps, delta)
    update_wd = _run_1d_core(p, g, wd, max_scale, eps, delta)

    update_rms = torch.sqrt(update_no_wd.pow(2).mean() + eps)
    param_rms_smooth = torch.sqrt(p.pow(2).mean() + delta**2)
    uncapped_scale = update_rms / param_rms_smooth
    cap = max_contraction / (lr * wd)

    assert uncapped_scale > cap, "test configuration should bind the cap"

    # wd term = wd * scale * p, so dividing by (wd * p) recovers ``scale``.
    estimated_scale = ((update_wd - update_no_wd) / (wd * p)).mean()
    torch.testing.assert_close(
        estimated_scale,
        torch.tensor(cap, device=DEVICE, dtype=torch.float32),
    )


def test_relative_wd_delta_smooths_zero_param():
    """Near-zero parameters stay finite thanks to the Huber denominator."""
    _requires_cuda()
    torch.manual_seed(2)
    p = torch.zeros(8, device=DEVICE)
    g = torch.randn(8, device=DEVICE)
    max_scale = _relative_wd_max_scale(0.1, 0.5, 0.99)
    update_wd = _run_1d_core(
        p, g, weight_decay=0.5, max_scale=max_scale, eps=1e-8, delta=1e-3
    )
    assert torch.isfinite(update_wd).all()


def test_relative_wd_max_scale_no_cap_when_non_positive():
    """A non-positive lr * weight_decay disables the overshoot cap."""
    assert _relative_wd_max_scale(0.0, 1.0, 0.5) == float("inf")
    assert _relative_wd_max_scale(1.0, 0.0, 0.5) == float("inf")
    assert _relative_wd_max_scale(1.0, 1.0, 0.5) == 0.5


@pytest.mark.parametrize("foreach", [False, True])
@pytest.mark.parametrize("warp_mode", ["dense", "spectral"])
def test_relative_wd_smoke_cuda(foreach, warp_mode):
    """Full WarpAINO steps with relative weight decay run on CUDA."""
    _requires_cuda()
    torch.manual_seed(3)

    lin = torch.nn.Linear(8, 4, device=DEVICE)
    bias_vec = torch.nn.Parameter(torch.randn(6, device=DEVICE))

    params = list(lin.parameters()) + [bias_vec]
    opt = WarpAINO(
        params,
        lr=1e-3,
        weight_decay=1e-2,
        relative_wd=True,
        relative_wd_delta=1e-2,
        relative_wd_max_contraction=0.9,
        meta_lr=1e-2,
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


def test_relative_wd_compile_step_cuda():
    """The compiled step path accepts the new scalar hyperparameters."""
    _requires_cuda()
    torch.manual_seed(4)

    lin = torch.nn.Linear(8, 4, device=DEVICE)
    opt = WarpAINO(
        list(lin.parameters()),
        lr=1e-3,
        weight_decay=1e-2,
        relative_wd=True,
        relative_wd_delta=1e-2,
        relative_wd_max_contraction=0.9,
        compile_step=True,
        warp_mode="spectral",
    )

    x = torch.randn(4, 8, device=DEVICE)
    lin(x).square().mean().backward()
    opt.step()

    assert torch.isfinite(lin.weight).all()
    assert torch.isfinite(lin.bias).all()


def test_relative_wd_compile_step_tensor_lr_cuda():
    """Compiled cores must not branch on a tensor-valued learning rate."""
    _requires_cuda()
    torch.manual_seed(5)

    lin = torch.nn.Linear(8, 4, device=DEVICE)
    opt = WarpAINO(
        list(lin.parameters()),
        lr=1e-3,
        weight_decay=1e-2,
        relative_wd=True,
        relative_wd_delta=1e-2,
        relative_wd_max_contraction=0.9,
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
