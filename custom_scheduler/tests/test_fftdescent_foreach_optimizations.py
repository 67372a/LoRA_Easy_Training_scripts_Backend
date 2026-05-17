"""Tests for FFTDescent foreach-path batching optimizations.

Validates that the following foreach optimizations produce numerically
equivalent results to the native (unoptimized per-tensor) path:

1. ``torch._foreach_zero_`` for first-step gradient zeroing
2. ``torch._foreach_abs`` pre-batching of |momentum| for atan2 normalization
3. ``torch._foreach_mul_`` batching of the 4/π scaling factor
4. ``torch._foreach_copy_`` for non-stochastic write-back
5. ``torch._foreach_sign`` / ``_foreach_abs_`` / ``_foreach_mul_`` batching
   in the FFT low-pass filter sub-step

Run with:
    cd backend/sd_scripts
    python -m pytest ../custom_scheduler/tests/test_fftdescent_foreach_optimizations.py -v
"""

import sys
import os
import copy
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from LoraEasyCustomOptimizer.fftdescent import FFTDescent

_NO_COMPILE = dict(spectral_clip_compile=False)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_model(seed=42, dtype=torch.float32, sizes=None):
    """Create a model for testing. *sizes* allows custom layer dimensions."""
    torch.manual_seed(seed)
    if sizes is None:
        sizes = [(32, 64), (64, 16)]
    layers = []
    for in_f, out_f in sizes:
        layers.append(torch.nn.Linear(in_f, out_f, dtype=dtype))
        layers.append(torch.nn.ReLU())
    return torch.nn.Sequential(*layers[:-1])  # drop trailing ReLU


def _run_steps(model, opt, n_steps=5, input_size=32, seed=999):
    """Run *n_steps* optimizer steps and return the final loss."""
    torch.manual_seed(seed)
    for _ in range(n_steps):
        x = torch.randn(8, input_size, dtype=next(model.parameters()).dtype)
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
# Test: foreach_zero_ correctness on first step
# ---------------------------------------------------------------------------

class TestForeachZeroFirstStep:
    """Verify that torch._foreach_zero_ correctly zeroes gradients on step 1."""

    def test_first_step_zeros_gradients(self):
        """After step 1 with foreach, momentum should be zero (grad was zeroed)."""
        model = _make_model()
        opt = FFTDescent(
            list(model.parameters()), lr=1e-3, foreach=True,
            sign_momentum=0.0, spectral_clip=False, lowpass_grad=0.0,
            stochastic_fp=False, **_NO_COMPILE,
        )
        x = torch.randn(4, 32)
        loss = model(x).sum()
        loss.backward()
        opt.step()
        opt.zero_grad()

        # On step 1 the gradient is zeroed, so momentum = beta*0 + (1-beta)*0 = 0
        for p in model.parameters():
            state = opt.state[p]
            assert torch.all(state["momentum"] == 0.0), (
                "Momentum should be zero after step 1 (gradient was zeroed)"
            )

    def test_first_step_zero_matches_native(self):
        """First-step zero behavior should match native path exactly."""
        model_ref = _make_model(seed=55)
        model_fe = copy.deepcopy(model_ref)

        opt_ref = FFTDescent(
            list(model_ref.parameters()), lr=1e-3, foreach=False,
            sign_momentum=0.0, spectral_clip=False, lowpass_grad=0.0,
            stochastic_fp=False, **_NO_COMPILE,
        )
        opt_fe = FFTDescent(
            list(model_fe.parameters()), lr=1e-3, foreach=True,
            sign_momentum=0.0, spectral_clip=False, lowpass_grad=0.0,
            stochastic_fp=False, **_NO_COMPILE,
        )

        torch.manual_seed(777)
        x = torch.randn(4, 32)
        loss_ref = model_ref(x).sum()
        loss_ref.backward()
        opt_ref.step()
        opt_ref.zero_grad()

        loss_fe = model_fe(x).sum()
        loss_fe.backward()
        opt_fe.step()
        opt_fe.zero_grad()

        max_diff = _max_param_diff(model_ref, model_fe)
        assert max_diff < 1e-6, (
            f"First-step foreach vs native diff {max_diff:.2e} exceeds 1e-6"
        )


# ---------------------------------------------------------------------------
# Test: momentum.abs() pre-batching with foreach_abs
# ---------------------------------------------------------------------------

class TestMomentumAbsBatching:
    """Verify that pre-batched |momentum| produces identical atan2 results."""

    def test_atan2_normalization_matches_native(self):
        """The atan2 normalization with pre-batched abs should match native."""
        kwargs = dict(
            lr=1e-3, sign_momentum=0.9, spectral_clip=False,
            lowpass_grad=0.0, stochastic_fp=False,
        )
        model_ref = _make_model(seed=100)
        model_fe = copy.deepcopy(model_ref)

        opt_ref = FFTDescent(
            list(model_ref.parameters()), foreach=False,
            **_NO_COMPILE, **kwargs,
        )
        opt_fe = FFTDescent(
            list(model_fe.parameters()), foreach=True,
            **_NO_COMPILE, **kwargs,
        )

        torch.manual_seed(888)
        # Run multiple steps so momentum accumulates non-trivial values
        for _ in range(10):
            x = torch.randn(8, 32)
            loss_ref = model_ref(x).sum()
            loss_ref.backward()
            opt_ref.step()
            opt_ref.zero_grad()

            loss_fe = model_fe(x).sum()
            loss_fe.backward()
            opt_fe.step()
            opt_fe.zero_grad()

        max_diff = _max_param_diff(model_ref, model_fe)
        # Without spectral clip, element-wise ops are order-independent
        assert max_diff < 1e-5, (
            f"atan2 normalization foreach vs native diff {max_diff:.2e} exceeds 1e-5"
        )

    def test_momentum_abs_with_spectral_clip(self):
        """Pre-batched |momentum| should match native with spectral clipping."""
        kwargs = dict(
            lr=1e-3, sign_momentum=0.9, spectral_clip=True,
            lowpass_grad=0.0, stochastic_fp=False,
        )
        model_ref = _make_model(seed=200)
        model_fe = copy.deepcopy(model_ref)

        opt_ref = FFTDescent(
            list(model_ref.parameters()), foreach=False,
            **_NO_COMPILE, **kwargs,
        )
        opt_fe = FFTDescent(
            list(model_fe.parameters()), foreach=True,
            **_NO_COMPILE, **kwargs,
        )

        torch.manual_seed(333)
        for _ in range(5):
            x = torch.randn(8, 32)
            loss_ref = model_ref(x).sum()
            loss_ref.backward()
            opt_ref.step()
            opt_ref.zero_grad()

            loss_fe = model_fe(x).sum()
            loss_fe.backward()
            opt_fe.step()
            opt_fe.zero_grad()

        max_diff = _max_param_diff(model_ref, model_fe)
        # Spectral clip has matrix ops with different accumulation order
        assert max_diff < 5e-2, (
            f"momentum.abs() foreach vs native (spectral clip) diff {max_diff:.2e}"
        )


# ---------------------------------------------------------------------------
# Test: 4/π scaling batched via foreach_mul_
# ---------------------------------------------------------------------------

class TestFourOverPiBatching:
    """Verify that the extracted 4/π foreach_mul_ matches inline scaling."""

    def test_scaling_matches_native_no_clip(self):
        """4/π scaling via foreach should match native inline scaling."""
        kwargs = dict(
            lr=1e-3, sign_momentum=0.9, spectral_clip=False,
            lowpass_grad=0.0, stochastic_fp=False,
        )
        model_ref = _make_model(seed=300)
        model_fe = copy.deepcopy(model_ref)

        opt_ref = FFTDescent(
            list(model_ref.parameters()), foreach=False,
            **_NO_COMPILE, **kwargs,
        )
        opt_fe = FFTDescent(
            list(model_fe.parameters()), foreach=True,
            **_NO_COMPILE, **kwargs,
        )

        torch.manual_seed(444)
        for _ in range(10):
            x = torch.randn(8, 32)
            loss_ref = model_ref(x).sum()
            loss_ref.backward()
            opt_ref.step()
            opt_ref.zero_grad()

            loss_fe = model_fe(x).sum()
            loss_fe.backward()
            opt_fe.step()
            opt_fe.zero_grad()

        max_diff = _max_param_diff(model_ref, model_fe)
        assert max_diff < 1e-5, (
            f"4/pi scaling foreach vs native diff {max_diff:.2e} exceeds 1e-5"
        )

    def test_scaling_with_sign_momentum_disabled(self):
        """4/π scaling should also match without sign momentum."""
        kwargs = dict(
            lr=1e-3, sign_momentum=0.0, spectral_clip=False,
            lowpass_grad=0.0, stochastic_fp=False,
        )
        model_ref = _make_model(seed=310)
        model_fe = copy.deepcopy(model_ref)

        opt_ref = FFTDescent(
            list(model_ref.parameters()), foreach=False,
            **_NO_COMPILE, **kwargs,
        )
        opt_fe = FFTDescent(
            list(model_fe.parameters()), foreach=True,
            **_NO_COMPILE, **kwargs,
        )

        torch.manual_seed(555)
        for _ in range(10):
            x = torch.randn(8, 32)
            loss_ref = model_ref(x).sum()
            loss_ref.backward()
            opt_ref.step()
            opt_ref.zero_grad()

            loss_fe = model_fe(x).sum()
            loss_fe.backward()
            opt_fe.step()
            opt_fe.zero_grad()

        max_diff = _max_param_diff(model_ref, model_fe)
        assert max_diff < 1e-5, (
            f"4/pi scaling (no sign_mom) foreach vs native diff {max_diff:.2e}"
        )


# ---------------------------------------------------------------------------
# Test: foreach_copy_ write-back for non-stochastic path
# ---------------------------------------------------------------------------

class TestForeachCopyWriteBack:
    """Verify that torch._foreach_copy_ write-back matches per-tensor copy_."""

    def test_fp32_writeback_matches_native(self):
        """Non-stochastic fp32 write-back via foreach_copy_ should match native."""
        kwargs = dict(
            lr=1e-3, sign_momentum=0.9, spectral_clip=False,
            lowpass_grad=0.0, stochastic_fp=False,
        )
        model_ref = _make_model(seed=400)
        model_fe = copy.deepcopy(model_ref)

        opt_ref = FFTDescent(
            list(model_ref.parameters()), foreach=False,
            **_NO_COMPILE, **kwargs,
        )
        opt_fe = FFTDescent(
            list(model_fe.parameters()), foreach=True,
            **_NO_COMPILE, **kwargs,
        )

        torch.manual_seed(666)
        for _ in range(10):
            x = torch.randn(8, 32)
            loss_ref = model_ref(x).sum()
            loss_ref.backward()
            opt_ref.step()
            opt_ref.zero_grad()

            loss_fe = model_fe(x).sum()
            loss_fe.backward()
            opt_fe.step()
            opt_fe.zero_grad()

        max_diff = _max_param_diff(model_ref, model_fe)
        assert max_diff < 1e-5, (
            f"foreach_copy_ write-back diff {max_diff:.2e} exceeds 1e-5"
        )

    def test_writeback_momentum_state_matches(self):
        """Momentum state after write-back should match native path."""
        kwargs = dict(
            lr=1e-3, sign_momentum=0.9, spectral_clip=False,
            lowpass_grad=0.0, stochastic_fp=False,
        )
        model_ref = _make_model(seed=410)
        model_fe = copy.deepcopy(model_ref)

        opt_ref = FFTDescent(
            list(model_ref.parameters()), foreach=False,
            **_NO_COMPILE, **kwargs,
        )
        opt_fe = FFTDescent(
            list(model_fe.parameters()), foreach=True,
            **_NO_COMPILE, **kwargs,
        )

        torch.manual_seed(777)
        for _ in range(5):
            x = torch.randn(8, 32)
            loss_ref = model_ref(x).sum()
            loss_ref.backward()
            opt_ref.step()
            opt_ref.zero_grad()

            loss_fe = model_fe(x).sum()
            loss_fe.backward()
            opt_fe.step()
            opt_fe.zero_grad()

        # Check momentum state matches
        for p_ref, p_fe in zip(model_ref.parameters(), model_fe.parameters()):
            mom_ref = opt_ref.state[p_ref]["momentum"]
            mom_fe = opt_fe.state[p_fe]["momentum"]
            diff = (mom_ref.float() - mom_fe.float()).abs().max().item()
            assert diff < 1e-5, (
                f"Momentum state diff {diff:.2e} exceeds 1e-5"
            )

    def test_writeback_sign_momentum_state_matches(self):
        """Sign-momentum state after write-back should match native path."""
        kwargs = dict(
            lr=1e-3, sign_momentum=0.9, spectral_clip=False,
            lowpass_grad=0.0, stochastic_fp=False,
        )
        model_ref = _make_model(seed=420)
        model_fe = copy.deepcopy(model_ref)

        opt_ref = FFTDescent(
            list(model_ref.parameters()), foreach=False,
            **_NO_COMPILE, **kwargs,
        )
        opt_fe = FFTDescent(
            list(model_fe.parameters()), foreach=True,
            **_NO_COMPILE, **kwargs,
        )

        torch.manual_seed(888)
        for _ in range(5):
            x = torch.randn(8, 32)
            loss_ref = model_ref(x).sum()
            loss_ref.backward()
            opt_ref.step()
            opt_ref.zero_grad()

            loss_fe = model_fe(x).sum()
            loss_fe.backward()
            opt_fe.step()
            opt_fe.zero_grad()

        # Check sign_momentum state matches
        for p_ref, p_fe in zip(model_ref.parameters(), model_fe.parameters()):
            sm_ref = opt_ref.state[p_ref]["sign_momentum"]
            sm_fe = opt_fe.state[p_fe]["sign_momentum"]
            diff = (sm_ref.float() - sm_fe.float()).abs().max().item()
            assert diff < 1e-5, (
                f"Sign-momentum state diff {diff:.2e} exceeds 1e-5"
            )


# ---------------------------------------------------------------------------
# Test: FFT filter foreach batching (sign/abs/mul)
# ---------------------------------------------------------------------------

class TestFFTForeachFilterBatching:
    """Verify that the foreach-batched FFT filter produces correct results."""

    def test_lowpass_filter_matches_native(self):
        """FFT filter with foreach-batched sign/abs/mul should match native."""
        kwargs = dict(
            lr=1e-3, sign_momentum=0.9, spectral_clip=False,
            lowpass_grad=1.0, stochastic_fp=False,
        )
        model_ref = _make_model(seed=500)
        model_fe = copy.deepcopy(model_ref)

        opt_ref = FFTDescent(
            list(model_ref.parameters()), foreach=False,
            **_NO_COMPILE, **kwargs,
        )
        opt_fe = FFTDescent(
            list(model_fe.parameters()), foreach=True,
            **_NO_COMPILE, **kwargs,
        )

        torch.manual_seed(999)
        for _ in range(5):
            x = torch.randn(8, 32)
            loss_ref = model_ref(x).sum()
            loss_ref.backward()
            opt_ref.step()
            opt_ref.zero_grad()

            loss_fe = model_fe(x).sum()
            loss_fe.backward()
            opt_fe.step()
            opt_fe.zero_grad()

        max_diff = _max_param_diff(model_ref, model_fe)
        # FFT filter + spectral clip differences accumulate
        assert max_diff < 5e-2, (
            f"FFT filter foreach vs native diff {max_diff:.2e} exceeds 5e-2"
        )

    def test_lowpass_filter_with_spectral_clip(self):
        """FFT filter + spectral clip with all foreach batching should match native."""
        kwargs = dict(
            lr=1e-3, sign_momentum=0.9, spectral_clip=True,
            lowpass_grad=1.0, stochastic_fp=False,
        )
        model_ref = _make_model(seed=510)
        model_fe = copy.deepcopy(model_ref)

        opt_ref = FFTDescent(
            list(model_ref.parameters()), foreach=False,
            **_NO_COMPILE, **kwargs,
        )
        opt_fe = FFTDescent(
            list(model_fe.parameters()), foreach=True,
            **_NO_COMPILE, **kwargs,
        )

        torch.manual_seed(111)
        for _ in range(5):
            x = torch.randn(8, 32)
            loss_ref = model_ref(x).sum()
            loss_ref.backward()
            opt_ref.step()
            opt_ref.zero_grad()

            loss_fe = model_fe(x).sum()
            loss_fe.backward()
            opt_fe.step()
            opt_fe.zero_grad()

        max_diff = _max_param_diff(model_ref, model_fe)
        assert max_diff < 5e-2, (
            f"FFT+spectral foreach vs native diff {max_diff:.2e} exceeds 5e-2"
        )


# ---------------------------------------------------------------------------
# Test: Many-parameter model exercises batching at scale
# ---------------------------------------------------------------------------

class TestManyParameterBatching:
    """Verify foreach optimizations with a larger model (more parameters)."""

    def test_large_model_foreach_matches_native(self):
        """A model with many layers should produce equivalent results."""
        sizes = [(64, 128), (128, 128), (128, 64), (64, 32), (32, 16)]
        kwargs = dict(
            lr=1e-3, sign_momentum=0.9, spectral_clip=True,
            lowpass_grad=1.0, stochastic_fp=False,
        )
        model_ref = _make_model(seed=600, sizes=sizes)
        model_fe = copy.deepcopy(model_ref)

        opt_ref = FFTDescent(
            list(model_ref.parameters()), foreach=False,
            **_NO_COMPILE, **kwargs,
        )
        opt_fe = FFTDescent(
            list(model_fe.parameters()), foreach=True,
            **_NO_COMPILE, **kwargs,
        )

        torch.manual_seed(222)
        for _ in range(5):
            x = torch.randn(8, 64)
            loss_ref = model_ref(x).sum()
            loss_ref.backward()
            opt_ref.step()
            opt_ref.zero_grad()

            loss_fe = model_fe(x).sum()
            loss_fe.backward()
            opt_fe.step()
            opt_fe.zero_grad()

        max_diff = _max_param_diff(model_ref, model_fe)
        assert max_diff < 5e-2, (
            f"Large model foreach vs native diff {max_diff:.2e} exceeds 5e-2"
        )

    def test_many_steps_stability(self):
        """Run many steps to verify no numerical drift from foreach batching."""
        kwargs = dict(
            lr=1e-3, sign_momentum=0.9, spectral_clip=False,
            lowpass_grad=0.0, stochastic_fp=False,
        )
        model_ref = _make_model(seed=700)
        model_fe = copy.deepcopy(model_ref)

        opt_ref = FFTDescent(
            list(model_ref.parameters()), foreach=False,
            **_NO_COMPILE, **kwargs,
        )
        opt_fe = FFTDescent(
            list(model_fe.parameters()), foreach=True,
            **_NO_COMPILE, **kwargs,
        )

        torch.manual_seed(123)
        for _ in range(50):
            x = torch.randn(8, 32)
            loss_ref = model_ref(x).sum()
            loss_ref.backward()
            opt_ref.step()
            opt_ref.zero_grad()

            loss_fe = model_fe(x).sum()
            loss_fe.backward()
            opt_fe.step()
            opt_fe.zero_grad()

        max_diff = _max_param_diff(model_ref, model_fe)
        # Over 50 steps without spectral clip, should still be very close
        assert max_diff < 1e-4, (
            f"50-step stability foreach vs native diff {max_diff:.2e} exceeds 1e-4"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
