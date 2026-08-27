"""CUDA tests for the faithful Simplified-AdEMAMix-AINO hybrid."""

import importlib.util
import os
import sys

import pytest
import torch


sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    from LoraEasyCustomOptimizer.simplified_ademamix_aino import (
        SimplifiedAdEMAMixAINO,
        _prepare_simplified_aino_update_1d,
    )
except ImportError:
    # Match the standalone fallback used by the WarpAINO tests when optional
    # dependencies imported by the package __init__ are unavailable.
    module_dir = os.path.join(
        os.path.dirname(__file__), "..", "LoraEasyCustomOptimizer"
    )
    warp_spec = importlib.util.spec_from_file_location(
        "warpaino_standalone", os.path.join(module_dir, "warpaino.py")
    )
    warp_module = importlib.util.module_from_spec(warp_spec)
    sys.modules[warp_spec.name] = warp_module
    warp_spec.loader.exec_module(warp_module)

    hybrid_spec = importlib.util.spec_from_file_location(
        "simplified_ademamix_aino_standalone",
        os.path.join(module_dir, "simplified_ademamix_aino.py"),
    )
    hybrid_module = importlib.util.module_from_spec(hybrid_spec)
    sys.modules[hybrid_spec.name] = hybrid_module
    hybrid_spec.loader.exec_module(hybrid_module)
    SimplifiedAdEMAMixAINO = hybrid_module.SimplifiedAdEMAMixAINO
    _prepare_simplified_aino_update_1d = (
        hybrid_module._prepare_simplified_aino_update_1d
    )


DEVICE = "cuda"


def _requires_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")


def test_theory_style_momentum_and_second_moment_cuda():
    """The first hybrid step matches the Simplified-AdEMAMix equations."""
    _requires_cuda()
    gradient = torch.tensor([1.0, -1.0, 0.5, -0.5], device=DEVICE)
    momentum = torch.zeros_like(gradient)
    exp_avg_sq = torch.zeros_like(gradient)
    num_sum = torch.ones((), device=DEVICE)
    beta1_t = torch.tensor(0.9, device=DEVICE)
    step_t = torch.tensor(1.0, device=DEVICE)

    update = _prepare_simplified_aino_update_1d(
        gradient,
        momentum,
        exp_avg_sq,
        beta1_t,
        beta2=0.95,
        alpha=0.5,
        eps=1e-8,
        step_t=step_t,
        num_sum=num_sum,
        bias_correction1=False,
        bias_correction2=True,
    )

    grad_norm = gradient / torch.sqrt(gradient.pow(2).mean() + 1e-8)
    expected_momentum = grad_norm
    expected_second_moment = grad_norm.pow(2) * 0.05
    expected_update = (expected_momentum + 0.5 * grad_norm) / (
        torch.sqrt(expected_second_moment / 0.05) + 1e-8
    )

    torch.testing.assert_close(momentum, expected_momentum)
    torch.testing.assert_close(exp_avg_sq, expected_second_moment)
    torch.testing.assert_close(update, expected_update)


@pytest.mark.parametrize("foreach", [False, True])
@pytest.mark.parametrize("warp_mode", ["dense", "spectral"])
def test_hybrid_optimizer_steps_cuda(foreach, warp_mode):
    """Native and foreach paths remain finite with both warp implementations."""
    _requires_cuda()
    torch.manual_seed(41)
    linear = torch.nn.Linear(8, 4, device=DEVICE)
    vector = torch.nn.Parameter(torch.randn(6, device=DEVICE))
    parameters = list(linear.parameters()) + [vector]
    optimizer = SimplifiedAdEMAMixAINO(
        parameters,
        lr=1e-4,
        betas=(0.97, 0.95, 0.999),
        alpha=0.1,
        meta_lr=1e-2,
        warp_mode=warp_mode,
        foreach=foreach,
    )

    before = [parameter.detach().clone() for parameter in parameters]
    for _ in range(3):
        loss = linear(torch.randn(4, 8, device=DEVICE)).square().mean()
        loss = loss + vector.square().mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    for original, parameter in zip(before, parameters):
        assert torch.isfinite(parameter).all()
        assert not torch.equal(original, parameter)

    state = optimizer.state[linear.weight]
    assert state["momentum"].shape == linear.weight.shape
    assert state["exp_avg_sq"].shape == linear.weight.shape
    assert "sign_momentum" not in state
    assert "exp_avg_sq_row" not in state


def test_hybrid_optimizer_unwarped_2d_step_cuda():
    """The plain 2D dispatch returns the same pair as warped dispatches."""
    _requires_cuda()
    torch.manual_seed(43)
    parameter = torch.nn.Parameter(torch.randn(8, 6, device=DEVICE))
    optimizer = SimplifiedAdEMAMixAINO(
        [parameter],
        lr=1e-4,
        betas=(0.97, 0.95, 0.999),
        alpha=0.1,
        meta_lr=0.0,
        warp_mode="spectral",
    )

    parameter.grad = torch.randn_like(parameter)
    optimizer.step()

    state = optimizer.state[parameter]
    assert "spectral_log_left" not in state
    assert torch.isfinite(parameter).all()
    assert torch.isfinite(state["momentum"]).all()
    assert torch.isfinite(state["exp_avg_sq"]).all()


def test_hybrid_optimizer_compile_step_cuda():
    """The faithful hybrid core traces through the compiled execution path."""
    _requires_cuda()
    torch.manual_seed(42)
    linear = torch.nn.Linear(8, 4, device=DEVICE)
    optimizer = SimplifiedAdEMAMixAINO(
        linear.parameters(),
        lr=1e-4,
        betas=(0.97, 0.95, 0.999),
        alpha=0.1,
        beta1_warmup=4,
        min_beta1=0.9,
        compile_step=True,
        meta_lr=0.0,
        warp_mode="spectral",
    )

    for _ in range(3):
        linear(torch.randn(4, 8, device=DEVICE)).square().mean().backward()
        optimizer.step()
        optimizer.zero_grad()

    assert torch.isfinite(linear.weight).all()
    assert torch.isfinite(linear.bias).all()
    assert optimizer.state[linear.weight]["step"] == 3
