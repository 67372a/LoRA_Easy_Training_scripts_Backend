"""Tests for FFTDescent with spectral_clip_compile=True (mode="default").

Validates that the fix changing ``spectral_clip_compiled_func`` from
``mode="reduce-overhead"`` to ``mode="default"`` resolves CUDA graph
capture failures on T4 / limited-SM GPUs when using fp16 mixed precision
and the foreach step path.

The original bug:
    RuntimeError: Expected curr_block->next == nullptr to be true
    (from torch._inductor.cudagraph_trees during CUDA graph capture)

Run with:
    cd backend/sd_scripts
    python -m pytest ../custom_scheduler/tests/test_fftdescent_compile_mode_fix.py -v
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
    spectral_clip_compiled_func,
    spectral_clip_func,
    _spectral_clip,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_device():
    """Return CUDA device if available, else CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _make_model(seed=42, dtype=torch.float32, device=None):
    """Create a small sequential model for testing."""
    if device is None:
        device = _get_device()
    torch.manual_seed(seed)
    model = torch.nn.Sequential(
        torch.nn.Linear(32, 64, dtype=dtype, device=device),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 16, dtype=dtype, device=device),
    )
    return model


def _run_steps(model, opt, n_steps=5, input_dtype=torch.float32, seed=999):
    """Run *n_steps* optimizer steps on *model*, return final loss."""
    device = next(model.parameters()).device
    torch.manual_seed(seed)
    loss = None
    for _ in range(n_steps):
        x = torch.randn(8, 32, dtype=input_dtype, device=device)
        loss = model(x).sum()
        loss.backward()
        opt.step()
        opt.zero_grad()
    return loss


def _max_param_diff(model_a, model_b):
    """Return the maximum absolute parameter difference between two models."""
    max_diff = 0.0
    for pa, pb in zip(model_a.parameters(), model_b.parameters()):
        diff = (pa.data.float() - pb.data.float()).abs().max().item()
        max_diff = max(max_diff, diff)
    return max_diff


# ---------------------------------------------------------------------------
# Compile-mode fix smoke tests
# ---------------------------------------------------------------------------

class TestCompileModeFixSmoke:
    """Verify the compiled spectral clip runs without CUDA graph errors
    on all GPU types (including T4)."""

    def test_foreach_fp16_compiled_spectral_clip(self):
        """The exact scenario from the bug report:
        fp16 params, foreach=True, spectral_clip_compile=True (default),
        with mixed precision stochastic rounding."""
        model = _make_model(dtype=torch.float16)
        opt = FFTDescent(
            list(model.parameters()),
            lr=1e-3,
            foreach=True,
            stochastic_fp=True,
            lowpass_grad=1.0,
            sign_momentum=0.9,
            spectral_clip=True,
            spectral_clip_compile=True,  # <-- uses spectral_clip_compiled_func
        )
        _run_steps(model, opt, n_steps=5, input_dtype=torch.float16)
        assert True  # no exception = pass

    def test_foreach_fp32_compiled_spectral_clip(self):
        """fp32 params, foreach=True, spectral_clip_compile=True."""
        model = _make_model(dtype=torch.float32)
        opt = FFTDescent(
            list(model.parameters()),
            lr=1e-3,
            foreach=True,
            stochastic_fp=True,
            lowpass_grad=1.0,
            sign_momentum=0.9,
            spectral_clip=True,
            spectral_clip_compile=True,
        )
        _run_steps(model, opt, n_steps=5)
        assert True

    def test_foreach_bf16_compiled_spectral_clip(self):
        """bf16 params, foreach=True, spectral_clip_compile=True."""
        model = _make_model(dtype=torch.bfloat16)
        opt = FFTDescent(
            list(model.parameters()),
            lr=1e-3,
            foreach=True,
            stochastic_fp=True,
            lowpass_grad=1.0,
            sign_momentum=0.9,
            spectral_clip=True,
            spectral_clip_compile=True,
        )
        _run_steps(model, opt, n_steps=5, input_dtype=torch.bfloat16)
        assert True

    def test_native_fp16_compiled_spectral_clip(self):
        """Native path (foreach=False) with fp16 and compiled spectral clip.
        The bug was in the foreach path, but verify native still works too."""
        model = _make_model(dtype=torch.float16)
        opt = FFTDescent(
            list(model.parameters()),
            lr=1e-3,
            foreach=False,
            stochastic_fp=True,
            lowpass_grad=1.0,
            sign_momentum=0.9,
            spectral_clip=True,
            spectral_clip_compile=True,
        )
        _run_steps(model, opt, n_steps=5, input_dtype=torch.float16)
        assert True


# ---------------------------------------------------------------------------
# Equivalence: compiled (mode="default") vs uncompiled spectral clip
# ---------------------------------------------------------------------------

class TestCompileModeEquivalence:
    """Verify the compiled spectral_clip_compiled_func (with mode="default")
    produces results numerically equivalent to the uncompiled spectral_clip_func.
    
    All tests place tensors on the detected device (CUDA or CPU) to avoid
    cross-device recompilation issues."""

    @staticmethod
    def _make_test_tensors(shape, dtype, device):
        """Create test tensors on the given device."""
        torch.manual_seed(42)
        W = torch.randn(*shape, dtype=dtype, device=device)
        W_clone = W.clone()
        return W, W_clone

    def test_compiled_vs_uncompiled_fp32(self):
        """Compiled spectral clip should match uncompiled for fp32 inputs."""
        device = _get_device()
        W, W_clone = self._make_test_tensors((16, 64), torch.float32, device)

        # Uncompiled
        result_uncompiled = spectral_clip_func(
            W, sigma_min=-1.0, sigma_max=1.0, ortho_dtype=torch.float32
        )
        # Compiled (mode="default")
        result_compiled = spectral_clip_compiled_func(
            W_clone, sigma_min=-1.0, sigma_max=1.0, ortho_dtype=torch.float32
        )

        max_diff = (result_uncompiled - result_compiled).abs().max().item()
        assert max_diff < 1e-5, (
            f"Compiled vs uncompiled spectral clip diff {max_diff:.2e} exceeds 1e-5"
        )

    def test_compiled_vs_uncompiled_fp16_inputs(self):
        """Compiled spectral clip with fp16 inputs (internally promoted to fp32)."""
        device = _get_device()
        W, W_clone = self._make_test_tensors((16, 64), torch.float16, device)

        result_uncompiled = spectral_clip_func(
            W, sigma_min=-1.0, sigma_max=1.0, ortho_dtype=None
        )
        result_compiled = spectral_clip_compiled_func(
            W_clone, sigma_min=-1.0, sigma_max=1.0, ortho_dtype=None
        )

        max_diff = (result_uncompiled.float() - result_compiled.float()).abs().max().item()
        # fp16 → fp32 internal promotion may introduce small differences
        assert max_diff < 1e-4, (
            f"Compiled vs uncompiled spectral clip (fp16 in) diff {max_diff:.2e}"
        )

    def test_compiled_spectral_clip_adaptive(self):
        """Compiled spectral clip with adaptive=True should match uncompiled."""
        device = _get_device()
        W, W_clone = self._make_test_tensors((32, 128), torch.float32, device)

        result_uncompiled = spectral_clip_func(
            W, sigma_min=-1.0, sigma_max=1.0, ortho_dtype=torch.float32, adaptive=True
        )
        result_compiled = spectral_clip_compiled_func(
            W_clone, sigma_min=-1.0, sigma_max=1.0, ortho_dtype=torch.float32, adaptive=True
        )

        max_diff = (result_uncompiled - result_compiled).abs().max().item()
        # Adaptive path uses einsum, which has larger numerical variance
        assert max_diff < 1e-3, (
            f"Compiled vs uncompiled adaptive spectral clip diff {max_diff:.2e}"
        )


# ---------------------------------------------------------------------------
# Foreach vs Native equivalence (with compiled spectral clip)
# ---------------------------------------------------------------------------

class TestForeachVsNativeWithCompile:
    """Verify foreach and native paths agree when both use compiled spectral clip."""

    def test_foreach_vs_native_fp32_compiled(self):
        """Foreach and native should agree with compiled spectral clip (fp32)."""
        model_ref = _make_model(seed=99, dtype=torch.float32)
        model_fe = copy.deepcopy(model_ref)

        opt_ref = FFTDescent(
            list(model_ref.parameters()),
            lr=1e-3, foreach=False, stochastic_fp=True,
            lowpass_grad=1.0, sign_momentum=0.9,
            spectral_clip=True, spectral_clip_compile=True,
        )
        opt_fe = FFTDescent(
            list(model_fe.parameters()),
            lr=1e-3, foreach=True, stochastic_fp=True,
            lowpass_grad=1.0, sign_momentum=0.9,
            spectral_clip=True, spectral_clip_compile=True,
        )

        torch.manual_seed(777)
        for _ in range(5):
            x = torch.randn(8, 32, device=next(model_ref.parameters()).device)
            loss_ref = model_ref(x).sum()
            loss_ref.backward()
            opt_ref.step()
            opt_ref.zero_grad()

            loss_fe = model_fe(x).sum()
            loss_fe.backward()
            opt_fe.step()
            opt_fe.zero_grad()

        max_diff = _max_param_diff(model_ref, model_fe)
        # Medium tolerance: Newton-Schulz matrix product accumulation order
        # differs between per-tensor loop (native) and batched foreach
        assert max_diff < 5e-2, (
            f"Foreach vs native (compiled spectral clip) diff {max_diff:.2e} exceeds 5e-2"
        )

    def test_foreach_vs_native_fp16_compiled(self):
        """Foreach and native should agree with compiled spectral clip (fp16)."""
        model_ref = _make_model(seed=99, dtype=torch.float16)
        model_fe = copy.deepcopy(model_ref)

        opt_ref = FFTDescent(
            list(model_ref.parameters()),
            lr=1e-3, foreach=False, stochastic_fp=True,
            lowpass_grad=1.0, sign_momentum=0.9,
            spectral_clip=True, spectral_clip_compile=True,
        )
        opt_fe = FFTDescent(
            list(model_fe.parameters()),
            lr=1e-3, foreach=True, stochastic_fp=True,
            lowpass_grad=1.0, sign_momentum=0.9,
            spectral_clip=True, spectral_clip_compile=True,
        )

        device = next(model_ref.parameters()).device
        torch.manual_seed(777)
        for _ in range(5):
            x = torch.randn(8, 32, dtype=torch.float16, device=device)
            loss_ref = model_ref(x).sum()
            loss_ref.backward()
            opt_ref.step()
            opt_ref.zero_grad()

            loss_fe = model_fe(x).sum()
            loss_fe.backward()
            opt_fe.step()
            opt_fe.zero_grad()

        max_diff = _max_param_diff(model_ref, model_fe)
        # fp16 stochastic rounding adds noise — wider tolerance
        assert max_diff < 0.5, (
            f"Foreach vs native fp16 (compiled spectral clip) diff {max_diff:.2e}"
        )


# ---------------------------------------------------------------------------
# Stress tests — many steps to trigger CUDA graph issues
# ---------------------------------------------------------------------------

class TestCompileModeStress:
    """Run many steps to ensure no CUDA graph / allocator issues surface."""

    def test_many_steps_fp16_foreach_compiled(self):
        """50 steps with fp16 + foreach + compiled spectral clip.
        This would have failed on T4 with mode="reduce-overhead"."""
        model = _make_model(seed=123, dtype=torch.float16)
        opt = FFTDescent(
            list(model.parameters()),
            lr=1e-3,
            foreach=True,
            stochastic_fp=True,
            lowpass_grad=1.0,
            sign_momentum=0.9,
            spectral_clip=True,
            spectral_clip_compile=True,
        )
        _run_steps(model, opt, n_steps=50, input_dtype=torch.float16)
        assert True

    def test_many_steps_fp32_foreach_compiled(self):
        """50 steps with fp32 + foreach + compiled spectral clip."""
        model = _make_model(seed=123, dtype=torch.float32)
        opt = FFTDescent(
            list(model.parameters()),
            lr=1e-3,
            foreach=True,
            stochastic_fp=True,
            lowpass_grad=1.0,
            sign_momentum=0.9,
            spectral_clip=True,
            spectral_clip_compile=True,
        )
        _run_steps(model, opt, n_steps=50)
        assert True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
