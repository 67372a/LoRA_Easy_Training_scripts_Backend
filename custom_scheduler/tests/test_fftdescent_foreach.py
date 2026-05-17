"""Tests for FFTDescent optimizer — foreach step path.

Validates that the ``foreach=True`` step path:
1. Runs without errors for various configurations
2. Produces numerically equivalent results to the native (``foreach=False``) path
3. Handles edge cases (bf16 params, no sign-momentum, no spectral clip, etc.)

Run with:
    cd backend/sd_scripts
    python -m pytest ../custom_scheduler/tests/test_fftdescent_foreach.py -v
"""

import sys
import os
import copy
import pytest
import torch

# Ensure the custom_scheduler package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from LoraEasyCustomOptimizer.fftdescent import FFTDescent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_model(seed=42, dtype=torch.float32):
    """Create a small sequential model for testing."""
    torch.manual_seed(seed)
    model = torch.nn.Sequential(
        torch.nn.Linear(32, 64, dtype=dtype),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 16, dtype=dtype),
    )
    return model


def _run_steps(model, opt, n_steps=5, input_dtype=torch.float32, seed=999):
    """Run *n_steps* optimizer steps on *model* and return the final loss."""
    torch.manual_seed(seed)
    for _ in range(n_steps):
        x = torch.randn(8, 32, dtype=input_dtype)
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


# Default kwargs that avoid torch.compile issues in CI/test environments
_NO_COMPILE = dict(spectral_clip_compile=False)


# ---------------------------------------------------------------------------
# Smoke tests — foreach path runs without errors
# ---------------------------------------------------------------------------

class TestForeachSmoke:
    """Verify that the foreach step path executes without exceptions."""

    def test_foreach_fp32_full_options(self):
        """foreach=True with all features enabled (fp32, lowpass, sign_momentum, spectral_clip)."""
        model = _make_model()
        opt = FFTDescent(
            list(model.parameters()), lr=1e-3, foreach=True, stochastic_fp=True,
            lowpass_grad=1.0, sign_momentum=0.9, spectral_clip=True,
            **_NO_COMPILE,
        )
        _run_steps(model, opt, n_steps=3)
        assert True  # no exception = pass

    def test_foreach_bf16(self):
        """foreach=True with bf16 parameters and stochastic rounding."""
        model = _make_model(dtype=torch.bfloat16)
        opt = FFTDescent(
            list(model.parameters()), lr=1e-3, foreach=True, stochastic_fp=True,
            lowpass_grad=1.0, sign_momentum=0.9, **_NO_COMPILE,
        )
        _run_steps(model, opt, n_steps=3, input_dtype=torch.bfloat16)
        assert True

    def test_foreach_no_sign_momentum(self):
        """foreach=True with sign_momentum=0 (disabled)."""
        model = _make_model()
        opt = FFTDescent(
            list(model.parameters()), lr=1e-3, foreach=True, sign_momentum=0.0,
            **_NO_COMPILE,
        )
        _run_steps(model, opt, n_steps=3)
        assert True

    def test_foreach_no_spectral_clip(self):
        """foreach=True with spectral_clip=False."""
        model = _make_model()
        opt = FFTDescent(
            list(model.parameters()), lr=1e-3, foreach=True, spectral_clip=False,
        )
        _run_steps(model, opt, n_steps=3)
        assert True

    def test_foreach_no_lowpass(self):
        """foreach=True with lowpass_grad=0 (no FFT filtering)."""
        model = _make_model()
        opt = FFTDescent(
            list(model.parameters()), lr=1e-3, foreach=True, lowpass_grad=0.0,
            **_NO_COMPILE,
        )
        _run_steps(model, opt, n_steps=3)
        assert True

    def test_foreach_weight_decay(self):
        """foreach=True with weight_decay > 0."""
        model = _make_model()
        opt = FFTDescent(
            list(model.parameters()), lr=1e-3, foreach=True, weight_decay=0.01,
            **_NO_COMPILE,
        )
        _run_steps(model, opt, n_steps=3)
        assert True

    def test_foreach_stochastic_fp_false(self):
        """foreach=True with stochastic_fp=False."""
        model = _make_model()
        opt = FFTDescent(
            list(model.parameters()), lr=1e-3, foreach=True, stochastic_fp=False,
            **_NO_COMPILE,
        )
        _run_steps(model, opt, n_steps=3)
        assert True

    def test_foreach_fp16(self):
        """foreach=True with fp16 parameters."""
        model = _make_model(dtype=torch.float16)
        opt = FFTDescent(
            list(model.parameters()), lr=1e-3, foreach=True, stochastic_fp=True,
            lowpass_grad=1.0, sign_momentum=0.9, **_NO_COMPILE,
        )
        _run_steps(model, opt, n_steps=3, input_dtype=torch.float16)
        assert True


# ---------------------------------------------------------------------------
# Numerical equivalence tests — foreach vs native
# ---------------------------------------------------------------------------

class TestForeachEquivalence:
    """Verify that foreach and native paths produce equivalent parameter values.

    Tolerance notes:
    - **Tight (1e-5)**: Configs without spectral clipping — element-wise ops are
      order-independent so foreach and native produce bit-identical results.
    - **Medium (5e-2)**: Configs with spectral clipping — the Newton-Schulz
      orthogonalization involves matrix products whose accumulation order
      differs between the per-tensor loop (native) and the batched foreach
      path.  Also, the native path applies an FFT roundtrip even when
      ``lowpass_grad=0`` (a pre-existing inconsistency), adding further drift.
    - **Wide (0.1)**: bf16 with stochastic rounding — inherent stochasticity.
    """

    @pytest.mark.parametrize("kwargs,label,tol", [
        # No spectral clip, no lowpass → tight tolerance
        (dict(lr=1e-3, sign_momentum=0.0, spectral_clip=False, lowpass_grad=0.0,
              stochastic_fp=False),
         "minimal", 1e-5),
        # No spectral clip, no lowpass, with sign_momentum → tight
        (dict(lr=1e-3, sign_momentum=0.9, spectral_clip=False, lowpass_grad=0.0,
              stochastic_fp=False),
         "sign_mom_only", 1e-5),
        # Spectral clip, no lowpass → medium tolerance (NS matrix ops)
        (dict(lr=1e-3, sign_momentum=0.9, spectral_clip=True, lowpass_grad=0.0,
              stochastic_fp=False),
         "spectral_clip_no_lowpass", 5e-2),
        # With weight decay, no lowpass → medium tolerance
        (dict(lr=1e-3, sign_momentum=0.9, spectral_clip=True, lowpass_grad=0.0,
              weight_decay=0.01, stochastic_fp=False),
         "with_wd_no_lowpass", 5e-2),
        # With lowpass → medium tolerance (FFT CPU vs GPU)
        (dict(lr=1e-3, sign_momentum=0.9, spectral_clip=True, lowpass_grad=1.0,
              stochastic_fp=False),
         "full_with_lowpass", 5e-2),
        # No spectral clip but with lowpass → medium tolerance
        (dict(lr=1e-3, sign_momentum=0.9, spectral_clip=False, lowpass_grad=1.0,
              stochastic_fp=False),
         "no_spectral_clip_with_lowpass", 5e-2),
    ])
    def test_foreach_matches_native(self, kwargs, label, tol):
        """foreach=True and foreach=False should produce equivalent parameters."""
        # Build two identical models from the same seed
        model_ref = _make_model(seed=123)
        model_fe = copy.deepcopy(model_ref)

        opt_ref = FFTDescent(list(model_ref.parameters()), foreach=False,
                             **_NO_COMPILE, **kwargs)
        opt_fe = FFTDescent(list(model_fe.parameters()), foreach=True,
                            **_NO_COMPILE, **kwargs)

        # Run the same number of steps with the same data
        torch.manual_seed(999)
        for step in range(5):
            x = torch.randn(8, 32)

            # Native path
            loss_ref = model_ref(x).sum()
            loss_ref.backward()
            opt_ref.step()
            opt_ref.zero_grad()

            # Foreach path
            loss_fe = model_fe(x).sum()
            loss_fe.backward()
            opt_fe.step()
            opt_fe.zero_grad()

        max_diff = _max_param_diff(model_ref, model_fe)
        assert max_diff < tol, (
            f"[{label}] foreach vs native param diff {max_diff:.2e} exceeds {tol:.0e} tolerance"
        )

    def test_foreach_matches_native_bf16(self):
        """foreach=True and foreach=False produce equivalent bf16 params (stochastic_fp=True)."""
        kwargs = dict(lr=1e-3, sign_momentum=0.9, spectral_clip=True,
                      lowpass_grad=0.0, stochastic_fp=True)

        torch.manual_seed(123)
        model_ref = _make_model(seed=123, dtype=torch.bfloat16)
        model_fe = copy.deepcopy(model_ref)

        opt_ref = FFTDescent(list(model_ref.parameters()), foreach=False,
                             **_NO_COMPILE, **kwargs)
        opt_fe = FFTDescent(list(model_fe.parameters()), foreach=True,
                            **_NO_COMPILE, **kwargs)

        torch.manual_seed(999)
        for _ in range(5):
            x = torch.randn(8, 32, dtype=torch.bfloat16)

            loss_ref = model_ref(x).sum()
            loss_ref.backward()
            opt_ref.step()
            opt_ref.zero_grad()

            loss_fe = model_fe(x).sum()
            loss_fe.backward()
            opt_fe.step()
            opt_fe.zero_grad()

        max_diff = _max_param_diff(model_ref, model_fe)
        # bf16 + stochastic rounding has larger variance, allow wider tolerance
        assert max_diff < 0.1, (
            f"bf16 foreach vs native param diff {max_diff:.2e} exceeds 0.1 tolerance"
        )


# ---------------------------------------------------------------------------
# Dispatch tests
# ---------------------------------------------------------------------------

class TestStepDispatch:
    """Verify the three-way dispatch in step()."""

    def test_compile_step_flag_overrides_foreach(self):
        """When compile_step=True and foreach=True, the _compile_step flag takes priority."""
        model = _make_model()
        opt = FFTDescent(list(model.parameters()), lr=1e-3, compile_step=True,
                         foreach=True, spectral_clip_compile=False)
        # Verify the flags are stored correctly
        assert opt._compile_step is True
        assert opt._foreach is True
        # The step() dispatch checks _compile_step first, so _step_compiled
        # would be called.  We don't run steps here because torch.compile
        # may not be available in all test environments.

    def test_foreach_dispatch(self):
        """When foreach=True and compile_step=False, foreach path should be used."""
        model = _make_model()
        opt = FFTDescent(list(model.parameters()), lr=1e-3, foreach=True,
                         compile_step=False, **_NO_COMPILE)
        assert opt._compile_step is False
        assert opt._foreach is True
        _run_steps(model, opt, n_steps=2)
        assert True

    def test_native_dispatch(self):
        """When both compile_step and foreach are False, native path should be used."""
        model = _make_model()
        opt = FFTDescent(list(model.parameters()), lr=1e-3, foreach=False,
                         compile_step=False, **_NO_COMPILE)
        assert opt._compile_step is False
        assert opt._foreach is False
        _run_steps(model, opt, n_steps=2)
        assert True


# ---------------------------------------------------------------------------
# State and attribute tests
# ---------------------------------------------------------------------------

class TestStateAndAttributes:
    """Verify optimizer state is correctly initialized and updated."""

    def test_momentum_state_created(self):
        """Momentum state should be initialized on the first step."""
        model = _make_model()
        opt = FFTDescent(list(model.parameters()), lr=1e-3, foreach=True,
                         **_NO_COMPILE)
        x = torch.randn(4, 32)
        loss = model(x).sum()
        loss.backward()
        opt.step()

        for p in model.parameters():
            state = opt.state[p]
            assert "momentum" in state
            assert state["momentum"].shape == p.grad.shape

    def test_sign_momentum_state_created(self):
        """Sign-momentum state should be initialized when sign_momentum != 0."""
        model = _make_model()
        opt = FFTDescent(list(model.parameters()), lr=1e-3, foreach=True,
                         sign_momentum=0.9, **_NO_COMPILE)
        x = torch.randn(4, 32)
        loss = model(x).sum()
        loss.backward()
        opt.step()

        for p in model.parameters():
            state = opt.state[p]
            assert "sign_momentum" in state

    def test_sign_momentum_state_not_created_when_disabled(self):
        """Sign-momentum state should NOT be created when sign_momentum=0."""
        model = _make_model()
        opt = FFTDescent(list(model.parameters()), lr=1e-3, foreach=True,
                         sign_momentum=0.0, **_NO_COMPILE)
        x = torch.randn(4, 32)
        loss = model(x).sum()
        loss.backward()
        opt.step()

        for p in model.parameters():
            state = opt.state[p]
            assert "sign_momentum" not in state

    def test_foreach_flag_stored_in_defaults(self):
        """The foreach flag should be stored in param group defaults."""
        m = torch.nn.Linear(4, 2)
        opt = FFTDescent(list(m.parameters()), foreach=True, **_NO_COMPILE)
        assert opt.param_groups[0]["foreach"] is True

        m2 = torch.nn.Linear(4, 2)
        opt2 = FFTDescent(list(m2.parameters()), foreach=False, **_NO_COMPILE)
        assert opt2.param_groups[0]["foreach"] is False

    def test_step_counter_increments(self):
        """The step counter should increment on each step() call."""
        model = _make_model()
        opt = FFTDescent(list(model.parameters()), lr=1e-3, foreach=True,
                         **_NO_COMPILE)
        x = torch.randn(4, 32)

        for expected_step in range(1, 4):
            loss = model(x).sum()
            loss.backward()
            opt.step()
            opt.zero_grad()
            assert opt.param_groups[0]["step"] == expected_step


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
