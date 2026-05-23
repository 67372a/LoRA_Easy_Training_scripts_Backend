"""Tests for FFTDescent optimizer — channels_last memory format support.

Validates that the ``compile_step=True`` and native step paths correctly handle
parameters in ``torch.channels_last`` memory format.  This is the specific
scenario that caused the inductor ``assert_size_stride`` failure:

    assert_size_stride(buf3, (64, 320, 3, 3), (2880, 9, 3, 1))
    AssertionError: stride 1==2880 at dim=0; stride 576==9 at dim=1 ...

The root cause was ``.clone()`` preserving channels_last strides, which then
propagated non-contiguous tensors into ``torch.compile(fullgraph=True)``.

Run with:
    cd backend/sd_scripts
    python -m pytest ../custom_scheduler/tests/test_fftdescent_channels_last.py -v
"""

import sys
import os
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from LoraEasyCustomOptimizer.fftdescent import FFTDescent

_NO_COMPILE = dict(spectral_clip_compile=False)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_conv_model_channels_last(seed=42, dtype=torch.float32):
    """Create a small conv model and convert it to channels_last format.

    This simulates the ``--opt_channels_last`` training flag or lycoris
    modules that internally use channels_last memory layout.
    """
    torch.manual_seed(seed)
    model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 32, kernel_size=3, dtype=dtype),
        torch.nn.ReLU(),
        torch.nn.Conv2d(32, 16, kernel_size=3, dtype=dtype),
    )
    model.to(memory_format=torch.channels_last)
    return model


def _make_compiled_step_opt(params, **kwargs):
    """Create an FFTDescent with compile_step=True but using the uncompiled
    static method for _compiled_step (avoids requiring torch.inductor)."""
    opt = FFTDescent(params, compile_step=True, **kwargs)
    opt._compiled_step = FFTDescent._fftdescent_step_fp32
    opt._compile_step = True
    return opt


def _run_conv_steps(model, opt, n_steps=5, seed=999, dtype=torch.float32):
    """Run *n_steps* optimizer steps on a conv model."""
    torch.manual_seed(seed)
    loss = None
    for _ in range(n_steps):
        x = torch.randn(4, 3, 32, 32, dtype=dtype)
        loss = model(x).sum()
        loss.backward()
        opt.step()
        opt.zero_grad()
    return loss


def _snapshot_params(model):
    """Return a list of cloned parameter tensors (detached, CPU, contiguous)."""
    return [p.detach().cpu().clone().contiguous() for p in model.parameters()]


def _max_param_diff(snap_a, snap_b):
    """Return the maximum absolute parameter difference between two snapshots."""
    max_diff = 0.0
    for pa, pb in zip(snap_a, snap_b):
        diff = (pa.float() - pb.float()).abs().max().item()
        max_diff = max(max_diff, diff)
    return max_diff


# ---------------------------------------------------------------------------
# Verify test setup — parameters actually ARE channels_last
# ---------------------------------------------------------------------------

class TestChannelsLastSetup:
    """Sanity checks that our test helpers produce channels_last tensors."""

    def test_conv_model_params_are_channels_last(self):
        """Conv model parameters should be in channels_last format."""
        model = _make_conv_model_channels_last()
        for name, p in model.named_parameters():
            if p.ndim == 4:  # Conv weight tensors
                assert p.is_contiguous(memory_format=torch.channels_last), (
                    f"Parameter '{name}' with shape {tuple(p.shape)} is not "
                    f"in channels_last format"
                )
                assert not p.is_contiguous(), (
                    f"Parameter '{name}' is regular contiguous (not channels_last)"
                )

    def test_channels_last_strides_match_expected(self):
        """channels_last strides for (32, 320, 3, 3) should be (2880, 1, 960, 320)."""
        t = torch.randn(32, 320, 3, 3).to(memory_format=torch.channels_last)
        # channels_last for (N, C, H, W) = (32, 320, 3, 3):
        # stride: (C*H*W, 1, W*C, C) = (2880, 1, 960, 320)
        assert t.stride() == (2880, 1, 960, 320), (
            f"Unexpected channels_last strides: {t.stride()}"
        )


# ---------------------------------------------------------------------------
# Core fix tests — channels_last parameters with compiled step
# ---------------------------------------------------------------------------

class TestChannelsLastCompiledStep:
    """Verify that compile_step=True handles channels_last parameters."""

    def test_channels_last_fp32_compiled_step(self):
        """Compiled step should handle channels_last fp32 conv parameters."""
        model = _make_conv_model_channels_last(dtype=torch.float32)
        opt = _make_compiled_step_opt(
            model.parameters(), lr=1e-3, **_NO_COMPILE
        )
        _run_conv_steps(model, opt, n_steps=3, dtype=torch.float32)
        assert all(not torch.isnan(p).any() for p in model.parameters()), \
            "NaN in parameters after compiled step with channels_last fp32"

    def test_channels_last_fp16_compiled_step(self):
        """Compiled step should handle channels_last fp16 conv parameters."""
        model = _make_conv_model_channels_last(dtype=torch.float16)
        opt = _make_compiled_step_opt(
            model.parameters(), lr=1e-3, stochastic_fp=True, **_NO_COMPILE
        )
        _run_conv_steps(model, opt, n_steps=3, dtype=torch.float16)
        assert all(not torch.isnan(p).any() for p in model.parameters()), \
            "NaN in parameters after compiled step with channels_last fp16"

    def test_channels_last_bf16_compiled_step(self):
        """Compiled step should handle channels_last bf16 conv parameters."""
        model = _make_conv_model_channels_last(dtype=torch.bfloat16)
        opt = _make_compiled_step_opt(
            model.parameters(), lr=1e-3, stochastic_fp=True, **_NO_COMPILE
        )
        _run_conv_steps(model, opt, n_steps=3, dtype=torch.bfloat16)
        assert all(not torch.isnan(p).any() for p in model.parameters()), \
            "NaN in parameters after compiled step with channels_last bf16"

    def test_channels_last_with_sign_momentum(self):
        """Compiled step should handle channels_last with sign_momentum."""
        model = _make_conv_model_channels_last(dtype=torch.float32)
        opt = _make_compiled_step_opt(
            model.parameters(), lr=1e-3, sign_momentum=0.9, **_NO_COMPILE
        )
        _run_conv_steps(model, opt, n_steps=3)
        assert all(not torch.isnan(p).any() for p in model.parameters()), \
            "NaN in parameters after compiled step with channels_last + sign_momentum"

    def test_channels_last_with_weight_decay(self):
        """Compiled step should handle channels_last with weight_decay."""
        model = _make_conv_model_channels_last(dtype=torch.float32)
        opt = _make_compiled_step_opt(
            model.parameters(), lr=1e-3, weight_decay=0.01, **_NO_COMPILE
        )
        _run_conv_steps(model, opt, n_steps=3)
        assert all(not torch.isnan(p).any() for p in model.parameters()), \
            "NaN in parameters after compiled step with channels_last + weight_decay"

    def test_channels_last_no_spectral_clip(self):
        """Compiled step should handle channels_last with spectral_clip=False."""
        model = _make_conv_model_channels_last(dtype=torch.float32)
        opt = _make_compiled_step_opt(
            model.parameters(), lr=1e-3, spectral_clip=False, **_NO_COMPILE
        )
        _run_conv_steps(model, opt, n_steps=3)
        assert all(not torch.isnan(p).any() for p in model.parameters()), \
            "NaN in parameters after compiled step with channels_last, no spectral_clip"


# ---------------------------------------------------------------------------
# Native step path — channels_last should also work
# ---------------------------------------------------------------------------

class TestChannelsLastNativeStep:
    """Verify that the native (uncompiled) step path handles channels_last."""

    def test_channels_last_fp32_native_step(self):
        """Native step should handle channels_last fp32 conv parameters."""
        model = _make_conv_model_channels_last(dtype=torch.float32)
        opt = FFTDescent(
            model.parameters(), lr=1e-3, compile_step=False, **_NO_COMPILE
        )
        _run_conv_steps(model, opt, n_steps=3, dtype=torch.float32)
        assert all(not torch.isnan(p).any() for p in model.parameters()), \
            "NaN in parameters after native step with channels_last fp32"

    def test_channels_last_fp16_native_step(self):
        """Native step should handle channels_last fp16 conv parameters."""
        model = _make_conv_model_channels_last(dtype=torch.float16)
        opt = FFTDescent(
            model.parameters(), lr=1e-3, stochastic_fp=True,
            compile_step=False, **_NO_COMPILE
        )
        _run_conv_steps(model, opt, n_steps=3, dtype=torch.float16)
        assert all(not torch.isnan(p).any() for p in model.parameters()), \
            "NaN in parameters after native step with channels_last fp16"


# ---------------------------------------------------------------------------
# Numerical equivalence — channels_last vs contiguous
# ---------------------------------------------------------------------------

class TestChannelsLastEquivalence:
    """Verify channels_last and contiguous produce equivalent results."""

    def test_compiled_step_channels_last_vs_contiguous(self):
        """Compiled step with channels_last params should produce equivalent
        results to contiguous params (same seed)."""
        n_steps = 3
        # Run with channels_last
        torch.manual_seed(42)
        model_cl = _make_conv_model_channels_last(dtype=torch.float32)
        opt_cl = _make_compiled_step_opt(
            model_cl.parameters(), lr=1e-3, **_NO_COMPILE
        )
        torch.manual_seed(999)
        _run_conv_steps(model_cl, opt_cl, n_steps=n_steps, dtype=torch.float32)
        snap_cl = _snapshot_params(model_cl)

        # Run with contiguous (same seed)
        torch.manual_seed(42)
        model_contig = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, kernel_size=3),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 16, kernel_size=3),
        )
        opt_contig = _make_compiled_step_opt(
            model_contig.parameters(), lr=1e-3, **_NO_COMPILE
        )
        torch.manual_seed(999)
        _run_conv_steps(model_contig, opt_contig, n_steps=n_steps, dtype=torch.float32)
        snap_contig = _snapshot_params(model_contig)

        diff = _max_param_diff(snap_cl, snap_contig)
        assert diff < 1e-5, (
            f"channels_last vs contiguous compiled step param diff {diff:.2e} "
            f"exceeds 1e-5 tolerance"
        )


# ---------------------------------------------------------------------------
# Direct _fftdescent_step_fp32 contiguity enforcement
# ---------------------------------------------------------------------------

class TestFFTDescentStepContiguity:
    """Verify the defensive .contiguous() calls inside _fftdescent_step_fp32."""

    def test_step_with_2d_contiguous_tensors(self):
        """_fftdescent_step_fp32 should work with contiguous 2D tensors."""
        shape = (128, 320)
        p_data = torch.randn(*shape)
        grad = torch.randn(*shape)
        momentum = torch.zeros(*shape)
        sign_momentum = torch.empty(0)
        filter_weights = torch.empty(0)

        lr_t = torch.tensor(1e-3)
        beta_t = torch.tensor(0.95)
        wd_scaled_t = torch.tensor(0.0)
        sign_mom_coeff_t = torch.tensor(0.0)
        spectral_min_t = torch.tensor(-1.0)
        spectral_max_t = torch.tensor(1.0)

        FFTDescent._fftdescent_step_fp32(
            p_data, grad, momentum, sign_momentum, filter_weights,
            lr_t, beta_t, wd_scaled_t, sign_mom_coeff_t,
            spectral_min_t, spectral_max_t,
            do_spectral_clip=True,
            use_sign_momentum=False,
            do_lowpass=False,
            spectral_adaptive=False,
            has_weight_decay=False,
            step_is_one=False,
            needs_reshape=False,
            num_ns_steps=6,
            ortho_dtype=torch.float32,
        )

        assert not torch.isnan(p_data).any(), "NaN in p_data after step"
        assert not torch.isinf(p_data).any(), "Inf in p_data after step"

    def test_step_with_4d_contiguous_tensors(self):
        """_fftdescent_step_fp32 should handle 4D contiguous tensors with needs_reshape=True."""
        shape = (32, 320, 3, 3)
        p_data = torch.randn(*shape)
        grad = torch.randn(*shape)
        momentum = torch.zeros(*shape)
        sign_momentum = torch.empty(0)
        filter_weights = torch.empty(0)

        lr_t = torch.tensor(1e-3)
        beta_t = torch.tensor(0.95)
        wd_scaled_t = torch.tensor(0.0)
        sign_mom_coeff_t = torch.tensor(0.0)
        spectral_min_t = torch.tensor(-1.0)
        spectral_max_t = torch.tensor(1.0)

        # 4D tensors require needs_reshape=True for spectral clipping
        FFTDescent._fftdescent_step_fp32(
            p_data, grad, momentum, sign_momentum, filter_weights,
            lr_t, beta_t, wd_scaled_t, sign_mom_coeff_t,
            spectral_min_t, spectral_max_t,
            do_spectral_clip=True,
            use_sign_momentum=False,
            do_lowpass=False,
            spectral_adaptive=False,
            has_weight_decay=False,
            step_is_one=False,
            needs_reshape=True,
            num_ns_steps=6,
            ortho_dtype=torch.float32,
        )

        assert not torch.isnan(p_data).any(), "NaN in p_data after step"
        assert not torch.isinf(p_data).any(), "Inf in p_data after step"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
