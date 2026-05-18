"""Tests for FFTDescent optimizer — compile_step path.

Validates that the ``compile_step=True`` step path:
1. Runs without errors for fp32, fp16, and bf16 parameter dtypes
2. Handles convolution-like tensors (dimcount > 2) that trigger reshape paths
3. Produces numerically equivalent results to the native (``compile_step=False``) path
4. Does not crash with non-contiguous stride patterns (the original issue)

Uses ``_make_compiled_step_opt`` which exercises the ``_step_compiled`` code
path (scalar tensors, FP32 copies, reshape/transpose/spectral-clip) without
requiring ``torch.inductor`` to be functional.

Run with:
    cd backend/sd_scripts
    python -m pytest ../custom_scheduler/tests/test_fftdescent_compile_step.py -v
"""

import sys
import os
import copy
import pytest
import torch

# Ensure the custom_scheduler package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from LoraEasyCustomOptimizer.fftdescent import (
    FFTDescent,
    orthogonalize,
    _spectral_clip,
)

# Default kwargs that avoid torch.compile in CI/test environments
# compile_step=True routes through _step_compiled but uses the uncompiled
# static method (see _make_compiled_step_opt below)
_NO_COMPILE = dict(spectral_clip_compile=False)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_model(seed=42, dtype=torch.float32, sizes=None):
    """Create a small model for testing.

    Args:
        seed: Random seed for initialization.
        dtype: Parameter dtype.
        sizes: Optional list of (out_features, in_features) tuples.
               When None, a default 2-layer MLP is created.
    """
    torch.manual_seed(seed)
    if sizes is not None:
        layers = []
        for i, (out_f, in_f) in enumerate(sizes):
            layers.append(torch.nn.Linear(out_f, in_f, dtype=dtype))
            if i < len(sizes) - 1:
                layers.append(torch.nn.ReLU())
        model = torch.nn.Sequential(*layers)
    else:
        model = torch.nn.Sequential(
            torch.nn.Linear(32, 64, dtype=dtype),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 16, dtype=dtype),
        )
    return model


def _make_conv_model(seed=42, dtype=torch.float32):
    """Create a small conv model to exercise dimcount > 2 reshape path."""
    torch.manual_seed(seed)
    model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 32, kernel_size=3, dtype=dtype),
        torch.nn.ReLU(),
        torch.nn.Conv2d(32, 16, kernel_size=3, dtype=dtype),
    )
    return model


def _make_compiled_step_opt(params, **kwargs):
    """Create an FFTDescent with compile_step=True but using the uncompiled
    static method for _compiled_step.  This exercises the _step_compiled
    code path (scalar tensors, FP32 copies, reshape, transpose, spectral-clip)
    without requiring torch.inductor to be functional."""
    opt = FFTDescent(params, compile_step=True, **kwargs)
    # Override the compiled step with the uncompiled static method
    opt._compiled_step = FFTDescent._fftdescent_step_fp32
    opt._compile_step = True  # Ensure the dispatch uses _step_compiled
    return opt


def _run_steps(model, opt, n_steps=5, input_shape=(8, 32), seed=999,
               input_dtype=None):
    """Run *n_steps* optimizer steps on *model* and return the final loss."""
    if input_dtype is None:
        # Infer input dtype from first parameter
        first_param = next(model.parameters())
        input_dtype = first_param.dtype
    torch.manual_seed(seed)
    loss = None
    for _ in range(n_steps):
        x = torch.randn(*input_shape, dtype=input_dtype)
        loss = model(x).sum()
        loss.backward()
        opt.step()
        opt.zero_grad()
    return loss


def _run_conv_steps(model, opt, n_steps=5, seed=999, input_dtype=None):
    """Run *n_steps* optimizer steps on a conv model."""
    if input_dtype is None:
        first_param = next(model.parameters())
        input_dtype = first_param.dtype
    torch.manual_seed(seed)
    loss = None
    for _ in range(n_steps):
        x = torch.randn(4, 3, 32, 32, dtype=input_dtype)
        loss = model(x).sum()
        loss.backward()
        opt.step()
        opt.zero_grad()
    return loss


def _snapshot_params(model):
    """Return a list of cloned parameter tensors (detached, on CPU)."""
    return [p.detach().cpu().clone() for p in model.parameters()]


def _max_param_diff(snap_a, snap_b):
    """Return the maximum absolute parameter difference between two snapshots."""
    max_diff = 0.0
    for pa, pb in zip(snap_a, snap_b):
        diff = (pa.float() - pb.float()).abs().max().item()
        max_diff = max(max_diff, diff)
    return max_diff


# ---------------------------------------------------------------------------
# Smoke tests — compiled step path runs without errors
# ---------------------------------------------------------------------------

class TestCompiledStepSmoke:
    """Verify that the compiled step path executes without exceptions."""

    def test_fp32_compiled_step_runs(self):
        """Compiled step path should run on fp32 model."""
        model = _make_model(dtype=torch.float32)
        opt = _make_compiled_step_opt(
            model.parameters(), lr=1e-3, **_NO_COMPILE
        )
        _run_steps(model, opt, n_steps=3)
        assert True

    def test_fp16_compiled_step_runs(self):
        """Compiled step path should run on fp16 model (mixed precision scenario)."""
        model = _make_model(dtype=torch.float16)
        opt = _make_compiled_step_opt(
            model.parameters(), lr=1e-3, stochastic_fp=True, **_NO_COMPILE
        )
        _run_steps(model, opt, n_steps=3, input_shape=(8, 32),
                   input_dtype=torch.float16)
        assert True

    def test_bf16_compiled_step_runs(self):
        """Compiled step path should run on bf16 model."""
        model = _make_model(dtype=torch.bfloat16)
        opt = _make_compiled_step_opt(
            model.parameters(), lr=1e-3, stochastic_fp=True, **_NO_COMPILE
        )
        _run_steps(model, opt, n_steps=3, input_shape=(8, 32),
                   input_dtype=torch.bfloat16)
        assert True

    def test_conv_fp16_compiled_step_runs(self):
        """Compiled step path should run on fp16 conv model (dimcount > 2, needs_reshape)."""
        model = _make_conv_model(dtype=torch.float16)
        opt = _make_compiled_step_opt(
            model.parameters(), lr=1e-3, stochastic_fp=True, **_NO_COMPILE
        )
        _run_conv_steps(model, opt, n_steps=3, input_dtype=torch.float16)
        assert True

    def test_conv_fp32_compiled_step_runs(self):
        """Compiled step path should run on fp32 conv model (dimcount > 2)."""
        model = _make_conv_model(dtype=torch.float32)
        opt = _make_compiled_step_opt(
            model.parameters(), lr=1e-3, **_NO_COMPILE
        )
        _run_conv_steps(model, opt, n_steps=3)
        assert True

    def test_no_spectral_clip_compiled_step_runs(self):
        """Compiled step path should run with spectral_clip=False."""
        model = _make_model(dtype=torch.float32)
        opt = _make_compiled_step_opt(
            model.parameters(), lr=1e-3, spectral_clip=False, **_NO_COMPILE
        )
        _run_steps(model, opt, n_steps=3)
        assert True

    def test_with_sign_momentum_compiled_step_runs(self):
        """Compiled step path should run with sign_momentum enabled."""
        model = _make_model(dtype=torch.float32)
        opt = _make_compiled_step_opt(
            model.parameters(), lr=1e-3, sign_momentum=0.9, **_NO_COMPILE
        )
        _run_steps(model, opt, n_steps=3)
        assert True

    def test_with_weight_decay_compiled_step_runs(self):
        """Compiled step path should run with weight decay."""
        model = _make_model(dtype=torch.float32)
        opt = _make_compiled_step_opt(
            model.parameters(), lr=1e-3, weight_decay=0.01, **_NO_COMPILE
        )
        _run_steps(model, opt, n_steps=3)
        assert True


# ---------------------------------------------------------------------------
# Stride-related tests — the original issue
# ---------------------------------------------------------------------------

class TestCompiledStepStrides:
    """Verify that the compiled step path handles various stride patterns.

    These tests specifically target the original ``assert_size_stride`` failure
    in torch.inductor when a (32, 320, 3, 3) tensor had non-contiguous strides
    (1, 288, 96, 32) instead of contiguous (2880, 9, 3, 1).
    """

    def test_non_contiguous_grad_input_fp32(self):
        """Compiled step should handle non-contiguous gradient inputs (fp32)."""
        # Create a parameter and a non-contiguous gradient
        param = torch.nn.Parameter(torch.randn(32, 320, 3, 3))
        # Create a non-contiguous gradient by permuting a contiguous tensor
        grad = torch.randn(3, 3, 320, 32).permute(3, 2, 0, 1)  # (32, 320, 3, 3) non-contiguous
        assert not grad.is_contiguous(), "Test setup: grad should be non-contiguous"

        opt = _make_compiled_step_opt([param], lr=1e-3, **_NO_COMPILE)
        param.grad = grad
        opt.step()
        assert not torch.isnan(param).any(), "NaN in parameter after step"
        assert not torch.isinf(param).any(), "Inf in parameter after step"

    def test_non_contiguous_grad_input_fp16(self):
        """Compiled step should handle non-contiguous gradient inputs (fp16)."""
        param = torch.nn.Parameter(torch.randn(32, 320, 3, 3, dtype=torch.float16))
        grad = torch.randn(3, 3, 320, 32, dtype=torch.float16).permute(3, 2, 0, 1)
        assert not grad.is_contiguous(), "Test setup: grad should be non-contiguous"

        opt = _make_compiled_step_opt(
            [param], lr=1e-3, stochastic_fp=True, **_NO_COMPILE
        )
        param.grad = grad
        opt.step()
        assert not torch.isnan(param).any(), "NaN in parameter after step"
        assert not torch.isinf(param).any(), "Inf in parameter after step"

    def test_transposed_gradient_fp16(self):
        """Compiled step should handle tall-skinny gradient (fp16, dimcount=2)."""
        # Tall-skinny matrix: shape (2880, 32) forces flip=True in spectral clip path
        weight = torch.nn.Parameter(torch.randn(2880, 32, dtype=torch.float16))
        grad = torch.randn(2880, 32, dtype=torch.float16)
        assert grad.is_contiguous()

        # The parameter shape triggers flip=True (2880 > 32), exercising .T paths
        opt = _make_compiled_step_opt(
            [weight], lr=1e-3, stochastic_fp=True, **_NO_COMPILE
        )
        weight.grad = grad
        opt.step()
        assert not torch.isnan(weight).any(), "NaN in parameter after step"
        assert not torch.isinf(weight).any(), "Inf in parameter after step"

    def test_reshape_path_with_non_contiguous_c_t(self):
        """Compiled step should handle c_t that is non-contiguous before reshape."""
        # Create a conv-like parameter where dimcount > 2
        param = torch.nn.Parameter(torch.randn(32, 320, 3, 3, dtype=torch.float32))
        # Non-contiguous gradient
        grad = torch.randn(3, 3, 320, 32).permute(3, 2, 0, 1)
        assert not grad.is_contiguous()

        opt = _make_compiled_step_opt(
            [param], lr=1e-3, spectral_clip=True, **_NO_COMPILE
        )
        param.grad = grad
        opt.step()
        assert not torch.isnan(param).any(), "NaN in parameter after step"
        assert not torch.isinf(param).any(), "Inf in parameter after step"


# ---------------------------------------------------------------------------
# Correctness — compiled step vs native step
# ---------------------------------------------------------------------------

class TestCompiledVsNative:
    """Verify that compiled step and native step produce equivalent results."""

    def _run_native_vs_compiled(self, model_factory, n_steps=5, seed=42,
                                 run_fn=_run_steps, **opt_kwargs):
        """Helper to compare native and compiled-step paths with identical seeds."""
        # Run native path
        torch.manual_seed(seed)
        model_native = model_factory()
        opt_native = FFTDescent(
            model_native.parameters(), lr=1e-3,
            compile_step=False, **opt_kwargs, **_NO_COMPILE
        )
        torch.manual_seed(999)
        run_fn(model_native, opt_native, n_steps=n_steps)
        snap_native = _snapshot_params(model_native)

        # Run compiled-step path (same seed)
        torch.manual_seed(seed)
        model_compiled = model_factory()
        opt_compiled = _make_compiled_step_opt(
            model_compiled.parameters(), lr=1e-3, **opt_kwargs
        )
        torch.manual_seed(999)
        run_fn(model_compiled, opt_compiled, n_steps=n_steps)
        snap_compiled = _snapshot_params(model_compiled)

        return _max_param_diff(snap_native, snap_compiled)

    def test_fp32_equivalent(self):
        """Compiled-step and native paths should produce similar results (fp32)."""
        diff = self._run_native_vs_compiled(
            lambda: _make_model(dtype=torch.float32), n_steps=5
        )
        assert diff < 1e-5, (
            f"Compiled vs native param diff {diff:.2e} exceeds 1e-5 tolerance"
        )

    def test_fp32_with_sign_momentum_equivalent(self):
        """Compiled-step and native should agree with sign_momentum."""
        diff = self._run_native_vs_compiled(
            lambda: _make_model(dtype=torch.float32),
            n_steps=5, sign_momentum=0.9
        )
        assert diff < 1e-5, (
            f"Compiled vs native with sign_momentum diff {diff:.2e}"
        )

    def test_fp32_with_weight_decay_equivalent(self):
        """Compiled-step and native should agree with weight_decay."""
        diff = self._run_native_vs_compiled(
            lambda: _make_model(dtype=torch.float32),
            n_steps=5, weight_decay=0.01
        )
        assert diff < 1e-5, (
            f"Compiled vs native with weight_decay diff {diff:.2e}"
        )

    def test_fp32_no_spectral_clip_equivalent(self):
        """Compiled-step and native should agree with spectral_clip=False."""
        diff = self._run_native_vs_compiled(
            lambda: _make_model(dtype=torch.float32),
            n_steps=5, spectral_clip=False
        )
        assert diff < 1e-5, (
            f"Compiled vs native without spectral_clip diff {diff:.2e}"
        )

    def test_fp16_equivalent(self):
        """Compiled-step and native paths should produce similar results (fp16)."""
        diff = self._run_native_vs_compiled(
            lambda: _make_model(dtype=torch.float16),
            n_steps=5, stochastic_fp=True,
            run_fn=lambda m, o, n_steps: _run_steps(
                m, o, n_steps=n_steps, input_shape=(8, 32), input_dtype=torch.float16
            )
        )
        # fp16 has lower precision — use relaxed tolerance
        assert diff < 1e-2, (
            f"fp16 compiled vs native param diff {diff:.2e} exceeds 1e-2 tolerance"
        )

    def test_conv_fp32_equivalent(self):
        """Compiled-step and native should agree for conv models."""
        diff = self._run_native_vs_compiled(
            lambda: _make_conv_model(dtype=torch.float32),
            n_steps=3, run_fn=_run_conv_steps
        )
        assert diff < 1e-5, (
            f"Conv compiled vs native param diff {diff:.2e}"
        )


# ---------------------------------------------------------------------------
# State and attribute tests
# ---------------------------------------------------------------------------

class TestCompiledStepState:
    """Verify optimizer state is correctly initialized and updated."""

    def test_momentum_state_created(self):
        """Momentum state should be initialized on the first compiled step."""
        model = _make_model()
        opt = _make_compiled_step_opt(
            model.parameters(), lr=1e-3, **_NO_COMPILE
        )
        x = torch.randn(4, 32)
        loss = model(x).sum()
        loss.backward()
        opt.step()

        for p in model.parameters():
            state = opt.state[p]
            assert "momentum" in state
            assert state["momentum"].shape == p.grad.shape

    def test_sign_momentum_state_created(self):
        """Sign-momentum state should be initialized when enabled."""
        model = _make_model()
        opt = _make_compiled_step_opt(
            model.parameters(), lr=1e-3, sign_momentum=0.9, **_NO_COMPILE
        )
        x = torch.randn(4, 32)
        loss = model(x).sum()
        loss.backward()
        opt.step()

        for p in model.parameters():
            state = opt.state[p]
            assert "sign_momentum" in state

    def test_sign_momentum_not_created_when_disabled(self):
        """Sign-momentum should NOT be created when sign_momentum=0."""
        model = _make_model()
        opt = _make_compiled_step_opt(
            model.parameters(), lr=1e-3, sign_momentum=0.0, **_NO_COMPILE
        )
        x = torch.randn(4, 32)
        loss = model(x).sum()
        loss.backward()
        opt.step()

        for p in model.parameters():
            state = opt.state[p]
            assert "sign_momentum" not in state

    def test_step_counter_increments(self):
        """The step counter should increment on each compiled step() call."""
        model = _make_model()
        opt = _make_compiled_step_opt(
            model.parameters(), lr=1e-3, **_NO_COMPILE
        )
        x = torch.randn(4, 32)

        for expected_step in range(1, 4):
            loss = model(x).sum()
            loss.backward()
            opt.step()
            opt.zero_grad()
            assert opt.param_groups[0]["step"] == expected_step

    def test_parameters_change(self):
        """Parameters should be modified after compiled steps."""
        model = _make_model()
        opt = _make_compiled_step_opt(
            model.parameters(), lr=1e-3, **_NO_COMPILE
        )
        before = _snapshot_params(model)
        # Run 2 steps to ensure parameters change (step 1 zeroes the gradient internally)
        _run_steps(model, opt, n_steps=2, input_shape=(4, 32))
        after = _snapshot_params(model)

        diff = _max_param_diff(before, after)
        assert diff > 0, "Parameters did not change after compiled steps"

    def test_fp16_dtype_preserved(self):
        """After compiled-step stochastic update, parameter dtype should remain fp16."""
        model = _make_model(dtype=torch.float16)
        opt = _make_compiled_step_opt(
            model.parameters(), lr=1e-3, stochastic_fp=True, **_NO_COMPILE
        )
        _run_steps(model, opt, n_steps=3, input_shape=(8, 32))
        for p in model.parameters():
            assert p.dtype == torch.float16, (
                f"Parameter dtype changed from float16 to {p.dtype}"
            )

    def test_bf16_dtype_preserved(self):
        """After compiled-step stochastic update, parameter dtype should remain bf16."""
        model = _make_model(dtype=torch.bfloat16)
        opt = _make_compiled_step_opt(
            model.parameters(), lr=1e-3, stochastic_fp=True, **_NO_COMPILE
        )
        _run_steps(model, opt, n_steps=3, input_shape=(8, 32))
        for p in model.parameters():
            assert p.dtype == torch.bfloat16, (
                f"Parameter dtype changed from bfloat16 to {p.dtype}"
            )


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestCompiledStepEdgeCases:
    """Verify the compiled step path handles edge cases correctly."""

    def test_scalar_param(self):
        """Compiled step should handle 0-dim scalar parameters."""
        model = torch.nn.Linear(8, 4, bias=True)
        model.scale = torch.nn.Parameter(torch.tensor(1.0))
        opt = _make_compiled_step_opt(
            model.parameters(), lr=1e-3, **_NO_COMPILE
        )

        for _ in range(5):
            x = torch.randn(4, 8)
            loss = model(x).sum() + model.scale
            loss.backward()
            opt.step()
            opt.zero_grad()

        assert not torch.isnan(model.scale).any(), "NaN in compiled scalar parameter"

    def test_skip_none_grad(self):
        """Parameters without gradients should be skipped gracefully."""
        model = torch.nn.Sequential(
            torch.nn.Linear(8, 8),
            torch.nn.Linear(8, 4),
        )
        opt = _make_compiled_step_opt(
            model.parameters(), lr=1e-3, **_NO_COMPILE
        )

        # Only backprop through the second layer
        x = torch.randn(4, 8)
        out = model[0](x)
        loss = model[1](out).sum()
        loss.backward()
        # model[0].grad is computed (it's in the graph), but let's test
        # that we don't crash if somehow a param has None grad
        opt.step()
        assert True

    def test_first_step_zero_gradient(self):
        """On step==1, gradient should be zeroed inside the step function."""
        model = _make_model(dtype=torch.float32)
        opt = _make_compiled_step_opt(
            model.parameters(), lr=1e-3, **_NO_COMPILE
        )

        # Capture momentum state after first step
        x = torch.randn(4, 32)
        loss = model(x).sum()
        loss.backward()
        opt.step()
        opt.zero_grad()

        for p in model.parameters():
            state = opt.state[p]
            # Momentum should be zero after first step (grad was zeroed)
            assert state["momentum"].abs().max().item() == 0.0, (
                "Momentum should be zero after step 1"
            )

    def test_no_nan_after_multiple_steps(self):
        """Parameters should not contain NaN after many compiled steps."""
        model = _make_model(dtype=torch.float16)
        opt = _make_compiled_step_opt(
            model.parameters(), lr=1e-3, stochastic_fp=True, **_NO_COMPILE
        )
        _run_steps(model, opt, n_steps=20, input_shape=(8, 32))
        for p in model.parameters():
            assert not torch.isnan(p).any(), (
                "NaN detected in fp16 compiled step parameters after 20 steps"
            )


# ---------------------------------------------------------------------------
# orthogonalize() stride correctness
# ---------------------------------------------------------------------------

class TestOrthogonalizeStrides:
    """Verify that orthogonalize() produces contiguous outputs regardless of input strides."""

    def test_contiguous_input_produces_contiguous_output(self):
        """orthogonalize() on contiguous input should return contiguous output."""
        M = torch.randn(32, 2880, dtype=torch.float32)
        result = orthogonalize(M)
        assert result.is_contiguous(), (
            f"orthogonalize output strides: {result.stride()}, expected contiguous"
        )

    def test_non_contiguous_input_handled(self):
        """orthogonalize() on non-contiguous (transposed) input should not crash."""
        M_contiguous = torch.randn(2880, 32, dtype=torch.float32)
        M = M_contiguous.T  # (32, 2880) with non-contiguous strides
        assert not M.is_contiguous()
        result = orthogonalize(M)
        # Output should be usable (contiguous or not, just don't crash)
        assert result.shape == M.shape

    def test_transpose_path_produces_contiguous(self):
        """When transpose=True (M.shape[0] < M.shape[1]), output should be contiguous."""
        # Tall-skinny input: 16 < 288 → transpose path is taken
        M = torch.randn(16, 288, dtype=torch.float32)
        assert M.shape[0] < M.shape[1], "Test setup: should trigger transpose path"
        result = orthogonalize(M)
        assert result.is_contiguous(), (
            f"orthogonalize transpose-path output strides: {result.stride()}"
        )
