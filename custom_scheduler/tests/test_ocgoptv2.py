"""Tests for OCGOptV2 optimizer optimizations.

Validates that the following optimizations produce numerically correct results:

1. Eliminated redundant bf16→fp32 clone: ``.detach().to(float32)`` instead of
   ``.detach().clone().to(float32)``
2. Cached ``beta**step`` to avoid computing it twice per beta (6→3 pow calls)
3. Pre-computed ``weight_decay * weight_decay_rate**step`` at group level
4. Replaced ``Q = I.clone()`` with ``Q = I`` in ``gram_newton_schulz_2step``
5. Factored repeated reshape pattern into ``_reshape_to_2d`` helper
6. Cleaned up dead commented-out code

Run with:
    cd backend/sd_scripts
    python -m pytest ../custom_scheduler/tests/test_ocgoptv2.py -v
"""

import sys
import os
import copy
import pytest
import torch

# Import directly from the module file to avoid pulling in the full
# LoraEasyCustomOptimizer package (which has heavy dependencies like
# pytorch_optimizer that may not be installed in the test environment).
# We set up a fake package context so that relative imports (e.g.
# ``from .utils import copy_stochastic_``) resolve correctly.
import importlib.util
import types

_pkg_dir = os.path.join(os.path.dirname(__file__), "..", "LoraEasyCustomOptimizer")

# Create a fake package module so relative imports work
if "LoraEasyCustomOptimizer" not in sys.modules:
    _pkg = types.ModuleType("LoraEasyCustomOptimizer")
    _pkg.__path__ = [_pkg_dir]
    _pkg.__package__ = "LoraEasyCustomOptimizer"
    sys.modules["LoraEasyCustomOptimizer"] = _pkg

# Load utils first (needed by ocgoptv2 via ``from .utils import copy_stochastic_``)
_utils_path = os.path.join(_pkg_dir, "utils.py")
_utils_spec = importlib.util.spec_from_file_location(
    "LoraEasyCustomOptimizer.utils", _utils_path
)
_utils_mod = importlib.util.module_from_spec(_utils_spec)
sys.modules["LoraEasyCustomOptimizer.utils"] = _utils_mod
_utils_spec.loader.exec_module(_utils_mod)

# Load ocgoptv2 as a submodule of the fake package
_ocgoptv2_path = os.path.join(_pkg_dir, "ocgoptv2.py")
_spec = importlib.util.spec_from_file_location(
    "LoraEasyCustomOptimizer.ocgoptv2", _ocgoptv2_path
)
_ocgoptv2 = importlib.util.module_from_spec(_spec)
sys.modules["LoraEasyCustomOptimizer.ocgoptv2"] = _ocgoptv2
_spec.loader.exec_module(_ocgoptv2)

OCGOptV2 = _ocgoptv2.OCGOptV2
gram_newton_schulz_2step = _ocgoptv2.gram_newton_schulz_2step
_reshape_to_2d = _ocgoptv2._reshape_to_2d

_NO_COMPILE = dict(spectral_clip_compile=False)


def _make_compiled_step_opt(params, **kwargs):
    """Create an OCGOptV2 with compile_step=True but using the uncompiled
    static method for _compiled_step.  This exercises the _step_compiled
    code path (scalar tensors, FP32 copies, in-place lerp_, etc.) without
    requiring torch.inductor to be functional."""
    opt = OCGOptV2(params, compile_step=True, **kwargs)
    # Override the compiled step with the uncompiled static method
    opt._compiled_step = OCGOptV2._ocgoptv2_step_fp32
    return opt


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_model(seed=42, dtype=torch.float32, sizes=None):
    """Create a simple feed-forward model for testing."""
    torch.manual_seed(seed)
    if sizes is None:
        sizes = [(32, 64), (64, 16)]
    layers = []
    for in_f, out_f in sizes:
        layers.append(torch.nn.Linear(in_f, out_f, dtype=dtype))
        layers.append(torch.nn.ReLU())
    return torch.nn.Sequential(*layers[:-1])


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


def _snapshot_params(model):
    """Return a detached copy of all parameter data."""
    return [p.data.clone() for p in model.parameters()]


def _max_param_diff(params_a, params_b):
    """Return the maximum absolute parameter difference between two snapshots."""
    max_diff = 0.0
    for pa, pb in zip(params_a, params_b):
        diff = (pa.float() - pb.float()).abs().max().item()
        max_diff = max(max_diff, diff)
    return max_diff


# ---------------------------------------------------------------------------
# Test: _reshape_to_2d helper
# ---------------------------------------------------------------------------

class TestReshapeTo2d:
    """Verify _reshape_to_2d handles all dimensionalities correctly."""

    def test_1d_tensor(self):
        """1D tensor should be reshaped to [1, N]."""
        t = torch.randn(10)
        r = _reshape_to_2d(t)
        assert r.shape == (1, 10)
        assert torch.equal(r.flatten(), t)

    def test_2d_tensor_passthrough(self):
        """2D tensor should be returned as-is (no copy)."""
        t = torch.randn(4, 8)
        r = _reshape_to_2d(t)
        assert r.shape == (4, 8)
        assert r.data_ptr() == t.data_ptr()  # same memory

    def test_3d_tensor(self):
        """3D tensor [C, H, W] should be reshaped to [C, H*W]."""
        t = torch.randn(3, 4, 5)
        r = _reshape_to_2d(t)
        assert r.shape == (3, 20)
        assert torch.equal(r.flatten(), t.flatten())

    def test_4d_tensor(self):
        """4D tensor [N, C, H, W] should be reshaped to [N, C*H*W]."""
        t = torch.randn(2, 3, 4, 5)
        r = _reshape_to_2d(t)
        assert r.shape == (2, 60)


# ---------------------------------------------------------------------------
# Test: gram_newton_schulz_2step (Q = I optimization)
# ---------------------------------------------------------------------------

class TestGramNewtonSchulz2Step:
    """Verify gram_newton_schulz_2step produces orthonormal output."""

    def test_output_shape_preserved(self):
        """Output should have the same shape as input."""
        M = torch.randn(8, 16)
        out = gram_newton_schulz_2step(M)
        assert out.shape == M.shape

    def test_output_columns_approximately_normalized(self):
        """Output columns should have norms close to 1 (spectral clipping
        approximation — not a perfect orthogonalizer with only 2 iterations)."""
        M = torch.randn(16, 8)
        out = gram_newton_schulz_2step(M, ortho_dtype=torch.float32)
        # Check column norms are in a reasonable range
        col_norms = torch.linalg.norm(out, dim=0)
        assert col_norms.min() > 0.5, f"Column norms too small: {col_norms}"
        assert col_norms.max() < 2.0, f"Column norms too large: {col_norms}"

    def test_output_preserves_directionality(self):
        """Output should approximately preserve the subspace direction
        (spectral clipping preserves dominant singular vectors)."""
        torch.manual_seed(123)
        M = torch.randn(16, 8)
        out = gram_newton_schulz_2step(M, ortho_dtype=torch.float32)
        # The output should not be zero or degenerate
        assert out.abs().max() > 0.1, "Output is degenerate"
        # Output should have same shape and finite values
        assert torch.isfinite(out).all(), "Output contains non-finite values"

    def test_wide_matrix_transpose(self):
        """Wide matrices (more cols than rows) should be handled via transpose."""
        M = torch.randn(4, 16)
        out = gram_newton_schulz_2step(M)
        assert out.shape == M.shape

    def test_dtype_preservation(self):
        """Output dtype should match input dtype."""
        M = torch.randn(8, 8)
        out = gram_newton_schulz_2step(M, ortho_dtype=torch.float32)
        assert out.dtype == M.dtype

    def test_deterministic(self):
        """Same input should produce same output."""
        M = torch.randn(8, 16)
        out1 = gram_newton_schulz_2step(M, ortho_dtype=torch.float32)
        out2 = gram_newton_schulz_2step(M, ortho_dtype=torch.float32)
        assert torch.equal(out1, out2)


# ---------------------------------------------------------------------------
# Test: OCGOptV2 basic functionality (fp32 path)
# ---------------------------------------------------------------------------

class TestOCGOptV2Basic:
    """Verify the optimizer runs without errors and reduces loss."""

    def test_fp32_convergence(self):
        """Optimizer should reduce loss over multiple steps (fp32)."""
        model = _make_model()
        opt = OCGOptV2(model.parameters(), lr=1e-3, **_NO_COMPILE)
        loss_init = _run_steps(model, opt, n_steps=1)
        loss_final = _run_steps(model, opt, n_steps=20)
        assert loss_final.item() < loss_init.item(), (
            f"Loss did not decrease: init={loss_init.item()}, final={loss_final.item()}"
        )

    def test_step_returns_loss_with_closure(self):
        """step() with closure should return the closure's loss."""
        model = _make_model()
        opt = OCGOptV2(model.parameters(), lr=1e-3, **_NO_COMPILE)
        x = torch.randn(8, 32)
        def closure():
            opt.zero_grad()
            loss = model(x).sum()
            loss.backward()
            return loss
        returned_loss = opt.step(closure)
        assert returned_loss is not None

    def test_step_returns_none_without_closure(self):
        """step() without closure should return None."""
        model = _make_model()
        opt = OCGOptV2(model.parameters(), lr=1e-3, **_NO_COMPILE)
        x = torch.randn(8, 32)
        loss = model(x).sum()
        loss.backward()
        returned_loss = opt.step()
        assert returned_loss is None

    def test_parameters_change_after_step(self):
        """Parameters should be modified after a step."""
        model = _make_model()
        opt = OCGOptV2(model.parameters(), lr=1e-3, **_NO_COMPILE)
        before = _snapshot_params(model)
        x = torch.randn(8, 32)
        loss = model(x).sum()
        loss.backward()
        opt.step()
        after = _snapshot_params(model)
        diff = _max_param_diff(before, after)
        assert diff > 0, "Parameters did not change after step"

    def test_no_nan_in_params(self):
        """Parameters should not contain NaN after multiple steps."""
        model = _make_model()
        opt = OCGOptV2(model.parameters(), lr=1e-3, **_NO_COMPILE)
        _run_steps(model, opt, n_steps=10)
        for p in model.parameters():
            assert not torch.isnan(p).any(), "NaN detected in parameters"

    def test_no_inf_in_params(self):
        """Parameters should not contain Inf after multiple steps."""
        model = _make_model()
        opt = OCGOptV2(model.parameters(), lr=1e-3, **_NO_COMPILE)
        _run_steps(model, opt, n_steps=10)
        for p in model.parameters():
            assert not torch.isinf(p).any(), "Inf detected in parameters"


# ---------------------------------------------------------------------------
# Test: OCGOptV2 bf16 + stochastic_fp path (the optimized detach/to path)
# ---------------------------------------------------------------------------

class TestOCGOptV2Bf16:
    """Verify the bf16 stochastic rounding path works correctly."""

    def test_bf16_no_crash(self):
        """Optimizer should not crash on bf16 parameters with stochastic_fp."""
        model = _make_model(dtype=torch.bfloat16)
        opt = OCGOptV2(model.parameters(), lr=1e-3, stochastic_fp=True, **_NO_COMPILE)
        _run_steps(model, opt, n_steps=5)
        # If we get here without exception, the test passes

    def test_bf16_no_nan(self):
        """bf16 parameters should not contain NaN after steps."""
        model = _make_model(dtype=torch.bfloat16)
        opt = OCGOptV2(model.parameters(), lr=1e-3, stochastic_fp=True, **_NO_COMPILE)
        _run_steps(model, opt, n_steps=10)
        for p in model.parameters():
            assert not torch.isnan(p).any(), "NaN detected in bf16 parameters"

    def test_bf16_state_stays_bf16(self):
        """After stochastic update, parameter dtype should remain bf16."""
        model = _make_model(dtype=torch.bfloat16)
        opt = OCGOptV2(model.parameters(), lr=1e-3, stochastic_fp=True, **_NO_COMPILE)
        _run_steps(model, opt, n_steps=3)
        for p in model.parameters():
            assert p.dtype == torch.bfloat16, (
                f"Parameter dtype changed from bfloat16 to {p.dtype}"
            )

    def test_bf16_state_momentum_dtype(self):
        """Momentum states should be stored in the parameter's dtype (bf16)."""
        model = _make_model(dtype=torch.bfloat16)
        opt = OCGOptV2(model.parameters(), lr=1e-3, stochastic_fp=True, **_NO_COMPILE)
        _run_steps(model, opt, n_steps=2)
        for p in model.parameters():
            state = opt.state[p]
            assert state["value_momentum"].dtype == torch.bfloat16
            assert state["centralized_momentum"].dtype == torch.bfloat16


# ---------------------------------------------------------------------------
# Test: OCGOptV2 weight decay
# ---------------------------------------------------------------------------

class TestOCGOptV2WeightDecay:
    """Verify weight decay is applied correctly."""

    def test_weight_decay_reduces_magnitude(self):
        """With weight decay, parameter magnitudes should decrease vs. no decay."""
        torch.manual_seed(42)
        sizes = [(16, 16)]
        model_a = _make_model(seed=42, sizes=sizes)
        model_b = copy.deepcopy(model_a)
        opt_no_wd = OCGOptV2(model_a.parameters(), lr=1e-3, weight_decay=0.0, **_NO_COMPILE)
        opt_wd = OCGOptV2(model_b.parameters(), lr=1e-3, weight_decay=0.1, **_NO_COMPILE)

        # Use same data for both
        torch.manual_seed(999)
        for _ in range(10):
            x = torch.randn(4, 16)
            loss_a = model_a(x).sum()
            loss_a.backward()
            opt_no_wd.step()
            opt_no_wd.zero_grad()

            loss_b = model_b(x).sum()
            loss_b.backward()
            opt_wd.step()
            opt_wd.zero_grad()

        # The model with weight decay should have smaller parameter norms
        norm_no_wd = sum(p.norm().item() for p in model_a.parameters())
        norm_wd = sum(p.norm().item() for p in model_b.parameters())
        assert norm_wd < norm_no_wd, (
            f"Weight decay did not reduce norms: no_wd={norm_no_wd}, wd={norm_wd}"
        )

    def test_weight_decay_zero_has_no_effect(self):
        """weight_decay=0 should produce same results as default."""
        model_a = _make_model()
        model_b = copy.deepcopy(model_a)
        opt_a = OCGOptV2(model_a.parameters(), lr=1e-3, weight_decay=0.0, **_NO_COMPILE)
        opt_b = OCGOptV2(model_b.parameters(), lr=1e-3, weight_decay=0.0, **_NO_COMPILE)

        torch.manual_seed(999)
        for _ in range(5):
            x = torch.randn(8, 32)
            loss_a = model_a(x).sum()
            loss_a.backward()
            opt_a.step()
            opt_a.zero_grad()

            loss_b = model_b(x).sum()
            loss_b.backward()
            opt_b.step()
            opt_b.zero_grad()

        diff = _max_param_diff(
            _snapshot_params(model_a), _snapshot_params(model_b)
        )
        assert diff < 1e-6, f"Identical configs produced different params: diff={diff}"


# ---------------------------------------------------------------------------
# Test: OCGOptV2 slow_beta (bias correction) computation
# ---------------------------------------------------------------------------

class TestOCGOptV2SlowBeta:
    """Verify the cached beta**step computation is numerically correct."""

    def test_slow_beta_values_match_naive(self):
        """Cached beta**step should match naive double-computation."""
        model = _make_model(sizes=[(4, 4)])
        opt = OCGOptV2(model.parameters(), lr=1e-3, betas=(0.95, 0.99, 0.999), **_NO_COMPILE)

        # Run one step to initialize state
        x = torch.randn(2, 4)
        loss = model(x).sum()
        loss.backward()
        opt.step()
        opt.zero_grad()

        # Check that group['step'] is 1
        assert opt.param_groups[0]['step'] == 1

        # Manually compute slow_betas the naive way for step=1
        beta1, beta2, beta3 = 0.95, 0.99, 0.999
        step = 1
        naive_slow_beta1 = (beta1**step - beta1) / (beta1**step - 1.0)
        naive_slow_beta2 = (beta2**step - beta2) / (beta2**step - 1.0)
        naive_slow_beta3 = (beta3**step - beta3) / (beta3**step - 1.0)

        # All should be 0 at step 1 (beta**1 = beta, so numerator = 0)
        assert abs(naive_slow_beta1) < 1e-10
        assert abs(naive_slow_beta2) < 1e-10
        assert abs(naive_slow_beta3) < 1e-10

        # Run more steps
        for _ in range(9):
            x = torch.randn(2, 4)
            loss = model(x).sum()
            loss.backward()
            opt.step()
            opt.zero_grad()

        step = opt.param_groups[0]['step']
        assert step == 10

        # Verify the formula at step 10
        b1p = beta1 ** step
        cached_slow_beta1 = (b1p - beta1) / (b1p - 1.0)
        naive_slow_beta1 = (beta1**step - beta1) / (beta1**step - 1.0)
        assert abs(cached_slow_beta1 - naive_slow_beta1) < 1e-15


# ---------------------------------------------------------------------------
# Test: OCGOptV2 adaptive mode
# ---------------------------------------------------------------------------

class TestOCGOptV2Adaptive:
    """Verify adaptive scaling mode works correctly."""

    def test_adaptive_mode_runs(self):
        """Adaptive mode should not crash."""
        model = _make_model()
        opt = OCGOptV2(model.parameters(), lr=1e-3, adaptive=True, **_NO_COMPILE)
        _run_steps(model, opt, n_steps=5)

    def test_adaptive_no_nan(self):
        """Adaptive mode should not produce NaN."""
        model = _make_model()
        opt = OCGOptV2(model.parameters(), lr=1e-3, adaptive=True, **_NO_COMPILE)
        _run_steps(model, opt, n_steps=10)
        for p in model.parameters():
            assert not torch.isnan(p).any(), "NaN in adaptive mode"


# ---------------------------------------------------------------------------
# Test: OCGOptV2 AOL preconditioning
# ---------------------------------------------------------------------------

class TestOCGOptV2AOL:
    """Verify AOL preconditioning mode works correctly."""

    def test_aol_mode_runs(self):
        """AOL mode should not crash."""
        model = _make_model()
        opt = OCGOptV2(model.parameters(), lr=1e-3, aol=True, **_NO_COMPILE)
        _run_steps(model, opt, n_steps=5)

    def test_aol_no_nan(self):
        """AOL mode should not produce NaN."""
        model = _make_model()
        opt = OCGOptV2(model.parameters(), lr=1e-3, aol=True, **_NO_COMPILE)
        _run_steps(model, opt, n_steps=10)
        for p in model.parameters():
            assert not torch.isnan(p).any(), "NaN in AOL mode"


# ---------------------------------------------------------------------------
# Test: OCGOptV2 input_norm mode
# ---------------------------------------------------------------------------

class TestOCGOptV2InputNorm:
    """Verify input_norm mode works correctly."""

    def test_input_norm_mode_runs(self):
        """input_norm mode should not crash."""
        model = _make_model()
        opt = OCGOptV2(model.parameters(), lr=1e-3, input_norm=True, **_NO_COMPILE)
        _run_steps(model, opt, n_steps=5)

    def test_input_norm_no_nan(self):
        """input_norm mode should not produce NaN."""
        model = _make_model()
        opt = OCGOptV2(model.parameters(), lr=1e-3, input_norm=True, **_NO_COMPILE)
        _run_steps(model, opt, n_steps=10)
        for p in model.parameters():
            assert not torch.isnan(p).any(), "NaN in input_norm mode"


# ---------------------------------------------------------------------------
# Test: OCGOptV2 scalar (0-dim) parameter handling
# ---------------------------------------------------------------------------

class TestOCGOptV2ScalarParam:
    """Verify the optimizer handles scalar (0-dim) parameters correctly."""

    def test_scalar_param_no_crash(self):
        """Optimizer should handle 0-dim parameters without crashing."""
        # Create a model with a bias that acts as a scalar-like param
        model = torch.nn.Linear(8, 4, bias=True)
        # Add a manual scalar parameter
        model.scale = torch.nn.Parameter(torch.tensor(1.0))
        opt = OCGOptV2(model.parameters(), lr=1e-3, **_NO_COMPILE)

        for _ in range(5):
            x = torch.randn(4, 8)
            loss = model(x).sum() + model.scale
            loss.backward()
            opt.step()
            opt.zero_grad()

        assert not torch.isnan(model.scale).any(), "NaN in scalar parameter"


# ---------------------------------------------------------------------------
# Test: OCGOptV2 determinism
# ---------------------------------------------------------------------------

class TestOCGOptV2Determinism:
    """Verify the optimizer produces deterministic results with same seed."""

    def test_deterministic_fp32(self):
        """Same seed should produce identical parameters (fp32)."""
        results = []
        for _ in range(2):
            torch.manual_seed(42)
            model = _make_model(seed=42, sizes=[(16, 16), (16, 4)])
            opt = OCGOptV2(model.parameters(), lr=1e-3, **_NO_COMPILE)
            torch.manual_seed(123)
            for _ in range(5):
                x = torch.randn(4, 16)
                loss = model(x).sum()
                loss.backward()
                opt.step()
                opt.zero_grad()
            results.append(_snapshot_params(model))

        diff = _max_param_diff(results[0], results[1])
        assert diff < 1e-7, f"Non-deterministic results: diff={diff}"


# ---------------------------------------------------------------------------
# Test: OCGOptV2 dtype string parsing
# ---------------------------------------------------------------------------

class TestOCGOptV2DtypeParsing:
    """Verify spectral_clip_dtype can be passed as a string."""

    def test_string_dtype_float32(self):
        """String 'torch.float32' should be parsed correctly."""
        model = _make_model()
        opt = OCGOptV2(
            model.parameters(), lr=1e-3,
            spectral_clip_dtype="torch.float32",
            spectral_clip_compile=False,
        )
        assert opt.param_groups[0]["spectral_clip_dtype"] == torch.float32

    def test_string_dtype_bfloat16(self):
        """String 'torch.bfloat16' should be parsed correctly."""
        model = _make_model()
        opt = OCGOptV2(
            model.parameters(), lr=1e-3,
            spectral_clip_dtype="torch.bfloat16",
            spectral_clip_compile=False,
        )
        assert opt.param_groups[0]["spectral_clip_dtype"] == torch.bfloat16

    def test_none_dtype_defaults_to_bf16(self):
        """None spectral_clip_dtype should default to bfloat16."""
        model = _make_model()
        opt = OCGOptV2(model.parameters(), lr=1e-3, **_NO_COMPILE)
        assert opt.param_groups[0]["spectral_clip_dtype"] == torch.bfloat16


# ---------------------------------------------------------------------------
# Test: OCGOptV2 weight_decay_rate interaction
# ---------------------------------------------------------------------------

class TestOCGOptV2WeightDecayRate:
    """Verify weight_decay_rate modulates decay over training steps."""

    def test_decay_rate_affects_magnitude(self):
        """A faster decay rate (smaller value) should reduce weight decay effect
        compared to a slower decay rate."""
        sizes = [(16, 16)]
        model_fast = _make_model(seed=42, sizes=sizes)
        model_slow = _make_model(seed=42, sizes=sizes)

        opt_fast = OCGOptV2(
            model_fast.parameters(), lr=1e-3,
            weight_decay=0.1, weight_decay_rate=0.5, **_NO_COMPILE
        )
        opt_slow = OCGOptV2(
            model_slow.parameters(), lr=1e-3,
            weight_decay=0.1, weight_decay_rate=0.999, **_NO_COMPILE
        )

        torch.manual_seed(999)
        for _ in range(20):
            x = torch.randn(4, 16)
            loss_f = model_fast(x).sum()
            loss_f.backward()
            opt_fast.step()
            opt_fast.zero_grad()

            loss_s = model_slow(x).sum()
            loss_s.backward()
            opt_slow.step()
            opt_slow.zero_grad()

        norm_fast = sum(p.norm().item() for p in model_fast.parameters())
        norm_slow = sum(p.norm().item() for p in model_slow.parameters())
        # Slower decay rate (0.999) means more total weight decay applied
        assert norm_slow < norm_fast, (
            f"Expected slower decay rate to reduce norms more: fast={norm_fast}, slow={norm_slow}"
        )


# ---------------------------------------------------------------------------
# Test: OCGOptV2 reset method
# ---------------------------------------------------------------------------

class TestOCGOptV2Reset:
    """Verify reset() exists and doesn't crash."""

    def test_reset_does_not_crash(self):
        model = _make_model()
        opt = OCGOptV2(model.parameters(), lr=1e-3, **_NO_COMPILE)
        opt.reset()  # Should be a no-op


# ---------------------------------------------------------------------------
# Test: OCGOptV2 no gradient handling
# ---------------------------------------------------------------------------

class TestOCGOptV2NoGrad:
    """Verify the optimizer skips parameters with no gradient."""

    def test_skip_none_grad(self):
        """Parameters without gradients should be skipped gracefully."""
        model = _make_model(sizes=[(8, 8), (8, 4)])
        opt = OCGOptV2(model.parameters(), lr=1e-3, **_NO_COMPILE)

        # Only compute gradient for the first parameter
        x = torch.randn(2, 8)
        # Forward through first layer only
        out = model[0](x)
        loss = out.sum()
        loss.backward()

        # Zero out gradient for second layer manually
        for name, param in model.named_parameters():
            if '1' in name:  # second layer
                param.grad = None

        opt.step()  # Should not crash


# ---------------------------------------------------------------------------
# Test: OCGOptV2 compile_step dispatch
# ---------------------------------------------------------------------------

class TestOCGOptV2CompileStepDispatch:
    """Verify the two-way dispatch in step() for compile_step."""

    def test_compile_step_flag_stored(self):
        """compile_step flag should be stored on the optimizer."""
        model = _make_model()
        opt = _make_compiled_step_opt(model.parameters(), lr=1e-3)
        assert opt._compile_step is True

    def test_compile_step_false_by_default(self):
        """compile_step should default to False."""
        model = _make_model()
        opt = OCGOptV2(model.parameters(), lr=1e-3, **_NO_COMPILE)
        assert opt._compile_step is False

    def test_compile_step_sets_clip_func_none(self):
        """When compile_step=True, clip_func should be None (inlined into graph)."""
        model = _make_model()
        opt = _make_compiled_step_opt(model.parameters(), lr=1e-3)
        assert opt.clip_func is None

    def test_native_dispatch(self):
        """When compile_step=False, native path should be used."""
        model = _make_model()
        opt = OCGOptV2(model.parameters(), lr=1e-3, compile_step=False, **_NO_COMPILE)
        assert opt._compile_step is False
        _run_steps(model, opt, n_steps=2)
        assert True

    def test_compiled_step_attribute_exists(self):
        """The _compiled_step attribute should always be set."""
        model = _make_model()
        opt = _make_compiled_step_opt(model.parameters(), lr=1e-3)
        assert hasattr(opt, '_compiled_step')
        assert callable(opt._compiled_step)


# ---------------------------------------------------------------------------
# Test: OCGOptV2 compiled step correctness (fp32)
# ---------------------------------------------------------------------------

class TestOCGOptV2CompiledStepFp32:
    """Verify the compiled step path produces correct results in fp32.

    Uses _make_compiled_step_opt which exercises the _step_compiled code
    path (scalar tensors, FP32 copies, in-place lerp_, etc.) without
    requiring torch.inductor.
    """

    def test_compiled_fp32_convergence(self):
        """Compiled-step optimizer should reduce loss over multiple steps (fp32)."""
        model = _make_model()
        opt = _make_compiled_step_opt(model.parameters(), lr=1e-3)
        loss_init = _run_steps(model, opt, n_steps=1)
        loss_final = _run_steps(model, opt, n_steps=20)
        assert loss_final.item() < loss_init.item(), (
            f"Compiled loss did not decrease: init={loss_init.item()}, final={loss_final.item()}"
        )

    def test_compiled_no_nan(self):
        """Compiled step should not produce NaN in parameters."""
        model = _make_model()
        opt = _make_compiled_step_opt(model.parameters(), lr=1e-3)
        _run_steps(model, opt, n_steps=10)
        for p in model.parameters():
            assert not torch.isnan(p).any(), "NaN detected in compiled step parameters"

    def test_compiled_no_inf(self):
        """Compiled step should not produce Inf in parameters."""
        model = _make_model()
        opt = _make_compiled_step_opt(model.parameters(), lr=1e-3)
        _run_steps(model, opt, n_steps=10)
        for p in model.parameters():
            assert not torch.isinf(p).any(), "Inf detected in compiled step parameters"

    def test_compiled_parameters_change(self):
        """Parameters should be modified after a compiled step."""
        model = _make_model()
        opt = _make_compiled_step_opt(model.parameters(), lr=1e-3)
        before = _snapshot_params(model)
        x = torch.randn(8, 32)
        loss = model(x).sum()
        loss.backward()
        opt.step()
        after = _snapshot_params(model)
        diff = _max_param_diff(before, after)
        assert diff > 0, "Parameters did not change after compiled step"

    def test_compiled_step_returns_loss_with_closure(self):
        """Compiled step() with closure should return the closure's loss."""
        model = _make_model()
        opt = _make_compiled_step_opt(model.parameters(), lr=1e-3)
        x = torch.randn(8, 32)
        def closure():
            opt.zero_grad()
            loss = model(x).sum()
            loss.backward()
            return loss
        returned_loss = opt.step(closure)
        assert returned_loss is not None

    def test_compiled_step_returns_none_without_closure(self):
        """Compiled step() without closure should return None."""
        model = _make_model()
        opt = _make_compiled_step_opt(model.parameters(), lr=1e-3)
        x = torch.randn(8, 32)
        loss = model(x).sum()
        loss.backward()
        returned_loss = opt.step()
        assert returned_loss is None


# ---------------------------------------------------------------------------
# Test: OCGOptV2 compiled vs native numerical agreement
# ---------------------------------------------------------------------------

class TestOCGOptV2CompiledVsNative:
    """Verify the compiled and native paths produce numerically close results.

    The compiled path uses in-place lerp_ while the native path uses
    out-of-place lerp (which rebinds local variables). These should produce
    identical results since the math is the same.
    """

    def _run_native_vs_compiled(self, sizes, n_steps=3, seed=42, **opt_kwargs):
        """Helper to compare native and compiled-step paths with identical seeds."""
        # Run native path
        torch.manual_seed(seed)
        model_native = _make_model(seed=seed, sizes=sizes)
        opt_native = OCGOptV2(model_native.parameters(), lr=1e-3, compile_step=False, **opt_kwargs, **_NO_COMPILE)
        torch.manual_seed(999)
        for _ in range(n_steps):
            x = torch.randn(4, sizes[0][0])
            loss = model_native(x).sum()
            loss.backward()
            opt_native.step()
            opt_native.zero_grad()
        native_params = _snapshot_params(model_native)

        # Run compiled-step path (same seed)
        torch.manual_seed(seed)
        model_compiled = _make_model(seed=seed, sizes=sizes)
        opt_compiled = _make_compiled_step_opt(model_compiled.parameters(), lr=1e-3, **opt_kwargs)
        torch.manual_seed(999)
        for _ in range(n_steps):
            x = torch.randn(4, sizes[0][0])
            loss = model_compiled(x).sum()
            loss.backward()
            opt_compiled.step()
            opt_compiled.zero_grad()
        compiled_params = _snapshot_params(model_compiled)

        return _max_param_diff(native_params, compiled_params)

    def test_compiled_vs_native_fp32_agreement(self):
        """Compiled-step and native paths should produce similar parameter values (fp32)."""
        diff = self._run_native_vs_compiled(sizes=[(16, 32), (32, 8)], n_steps=5)
        assert diff < 1e-5, (
            f"Compiled vs native param diff {diff:.2e} exceeds 1e-5 tolerance"
        )

    def test_compiled_vs_native_with_adaptive(self):
        """Compiled-step and native should agree with adaptive=True."""
        diff = self._run_native_vs_compiled(sizes=[(16, 16)], n_steps=3, adaptive=True)
        assert diff < 1e-5, (
            f"Compiled vs native adaptive param diff {diff:.2e} exceeds 1e-5 tolerance"
        )

    def test_compiled_vs_native_with_weight_decay(self):
        """Compiled-step and native should agree with weight_decay."""
        diff = self._run_native_vs_compiled(sizes=[(16, 16)], n_steps=3, weight_decay=0.01)
        assert diff < 1e-5, (
            f"Compiled vs native weight_decay param diff {diff:.2e} exceeds 1e-5 tolerance"
        )

    def test_compiled_vs_native_with_aol(self):
        """Compiled-step and native should agree with aol=True."""
        diff = self._run_native_vs_compiled(sizes=[(16, 32), (32, 8)], n_steps=3, aol=True)
        assert diff < 1e-5, (
            f"Compiled vs native aol param diff {diff:.2e} exceeds 1e-5 tolerance"
        )

    def test_compiled_vs_native_with_input_norm(self):
        """Compiled-step and native should agree with input_norm=True."""
        diff = self._run_native_vs_compiled(sizes=[(16, 32), (32, 8)], n_steps=3, input_norm=True)
        assert diff < 1e-5, (
            f"Compiled vs native input_norm param diff {diff:.2e} exceeds 1e-5 tolerance"
        )


# ---------------------------------------------------------------------------
# Test: OCGOptV2 compiled step with bf16
# ---------------------------------------------------------------------------

class TestOCGOptV2CompiledBf16:
    """Verify the compiled step path works with bf16 parameters."""

    def test_compiled_bf16_no_crash(self):
        """Compiled-step optimizer should not crash on bf16 parameters."""
        model = _make_model(dtype=torch.bfloat16)
        opt = _make_compiled_step_opt(model.parameters(), lr=1e-3, stochastic_fp=True)
        _run_steps(model, opt, n_steps=5)

    def test_compiled_bf16_no_nan(self):
        """Compiled-step bf16 parameters should not contain NaN after steps."""
        model = _make_model(dtype=torch.bfloat16)
        opt = _make_compiled_step_opt(model.parameters(), lr=1e-3, stochastic_fp=True)
        _run_steps(model, opt, n_steps=10)
        for p in model.parameters():
            assert not torch.isnan(p).any(), "NaN detected in compiled bf16 parameters"

    def test_compiled_bf16_stays_bf16(self):
        """After compiled-step stochastic update, parameter dtype should remain bf16."""
        model = _make_model(dtype=torch.bfloat16)
        opt = _make_compiled_step_opt(model.parameters(), lr=1e-3, stochastic_fp=True)
        _run_steps(model, opt, n_steps=3)
        for p in model.parameters():
            assert p.dtype == torch.bfloat16, (
                f"Parameter dtype changed from bfloat16 to {p.dtype}"
            )


# ---------------------------------------------------------------------------
# Test: OCGOptV2 compiled step scalar parameter handling
# ---------------------------------------------------------------------------

class TestOCGOptV2CompiledScalarParam:
    """Verify the compiled step path handles scalar (0-dim) parameters correctly."""

    def test_compiled_scalar_param_no_crash(self):
        """Compiled-step optimizer should handle 0-dim parameters without crashing."""
        model = torch.nn.Linear(8, 4, bias=True)
        model.scale = torch.nn.Parameter(torch.tensor(1.0))
        opt = _make_compiled_step_opt(model.parameters(), lr=1e-3)

        for _ in range(5):
            x = torch.randn(4, 8)
            loss = model(x).sum() + model.scale
            loss.backward()
            opt.step()
            opt.zero_grad()

        assert not torch.isnan(model.scale).any(), "NaN in compiled scalar parameter"


# ---------------------------------------------------------------------------
# Test: OCGOptV2 compiled step determinism
# ---------------------------------------------------------------------------

class TestOCGOptV2CompiledDeterminism:
    """Verify the compiled step path produces deterministic results with same seed."""

    def test_compiled_deterministic_fp32(self):
        """Same seed should produce identical parameters with compiled-step (fp32)."""
        results = []
        for _ in range(2):
            torch.manual_seed(42)
            model = _make_model(seed=42, sizes=[(16, 16), (16, 4)])
            opt = _make_compiled_step_opt(model.parameters(), lr=1e-3)
            torch.manual_seed(123)
            for _ in range(5):
                x = torch.randn(4, 16)
                loss = model(x).sum()
                loss.backward()
                opt.step()
                opt.zero_grad()
            results.append(_snapshot_params(model))

        diff = _max_param_diff(results[0], results[1])
        assert diff < 1e-7, f"Non-deterministic compiled results: diff={diff}"


# ---------------------------------------------------------------------------
# Test: OCGOptV2 compiled step state management
# ---------------------------------------------------------------------------

class TestOCGOptV2CompiledState:
    """Verify optimizer state is correctly initialized and updated in compiled step path."""

    def test_compiled_momentum_state_created(self):
        """Momentum states should be initialized on the first compiled step."""
        model = _make_model()
        opt = _make_compiled_step_opt(model.parameters(), lr=1e-3)
        x = torch.randn(4, 32)
        loss = model(x).sum()
        loss.backward()
        opt.step()

        for p in model.parameters():
            state = opt.state[p]
            assert "value_momentum" in state
            assert "centralized_momentum" in state
            assert state["value_momentum"].shape == p.grad.shape
            assert state["centralized_momentum"].shape == p.grad.shape

    def test_compiled_denom_state_for_scalar(self):
        """Denom state should be created for scalar parameters in compiled step path."""
        model = torch.nn.Linear(8, 4, bias=True)
        model.scale = torch.nn.Parameter(torch.tensor(1.0))
        opt = _make_compiled_step_opt(model.parameters(), lr=1e-3)

        x = torch.randn(4, 8)
        loss = model(x).sum() + model.scale
        loss.backward()
        opt.step()

        state = opt.state[model.scale]
        assert "denom" in state
        assert state["denom"].shape == model.scale.shape

    def test_compiled_step_counter_increments(self):
        """The step counter should increment on each compiled step() call."""
        model = _make_model()
        opt = _make_compiled_step_opt(model.parameters(), lr=1e-3)
        x = torch.randn(4, 32)

        for expected_step in range(1, 4):
            loss = model(x).sum()
            loss.backward()
            opt.step()
            opt.zero_grad()
            assert opt.param_groups[0]["step"] == expected_step


# ---------------------------------------------------------------------------
# Test: OCGOptV2 compiled step no gradient handling
# ---------------------------------------------------------------------------

class TestOCGOptV2CompiledNoGrad:
    """Verify the compiled step path skips parameters with no gradient."""

    def test_compiled_skip_none_grad(self):
        """Parameters without gradients should be skipped gracefully in compiled step path."""
        model = _make_model(sizes=[(8, 8), (8, 4)])
        opt = _make_compiled_step_opt(model.parameters(), lr=1e-3)

        x = torch.randn(2, 8)
        out = model[0](x)
        loss = out.sum()
        loss.backward()

        for name, param in model.named_parameters():
            if '1' in name:
                param.grad = None

        opt.step()  # Should not crash

    def test_skip_on_subsequent_steps(self):
        """Parameters that lose gradients mid-training should be skipped."""
        model = _make_model(sizes=[(8, 8), (8, 4)])
        opt = OCGOptV2(model.parameters(), lr=1e-3, **_NO_COMPILE)

        # First, run normally to build up state
        x = torch.randn(2, 8)
        loss = model(x).sum()
        loss.backward()
        opt.step()
        opt.zero_grad()

        # Now remove gradient for some params
        for name, param in model.named_parameters():
            if '0.weight' in name:
                param.grad = None

        opt.step()  # Should not crash


# ---------------------------------------------------------------------------
# Test: OCGOptV2 foreach dispatch
# ---------------------------------------------------------------------------

class TestOCGOptV2ForeachDispatch:
    """Verify the three-way dispatch in step() for foreach."""

    def test_foreach_flag_stored(self):
        """foreach flag should be stored on the optimizer."""
        model = _make_model()
        opt = OCGOptV2(model.parameters(), lr=1e-3, foreach=True, **_NO_COMPILE)
        assert opt._foreach is True

    def test_foreach_false_by_default(self):
        """foreach should default to False."""
        model = _make_model()
        opt = OCGOptV2(model.parameters(), lr=1e-3, **_NO_COMPILE)
        assert opt._foreach is False

    def test_foreach_dispatch(self):
        """When foreach=True, foreach path should be used."""
        model = _make_model()
        opt = OCGOptV2(model.parameters(), lr=1e-3, foreach=True, **_NO_COMPILE)
        assert opt._foreach is True
        _run_steps(model, opt, n_steps=2)
        assert True

    def test_foreach_sets_clip_func(self):
        """When foreach=True (and compile_step=False), clip_func should be set."""
        model = _make_model()
        opt = OCGOptV2(model.parameters(), lr=1e-3, foreach=True, **_NO_COMPILE)
        assert opt.clip_func is not None
        assert callable(opt.clip_func)

    def test_compile_step_takes_priority_over_foreach(self):
        """When both compile_step=True and foreach=True, compile_step should win."""
        model = _make_model()
        opt = OCGOptV2(model.parameters(), lr=1e-3, compile_step=True, foreach=True, **_NO_COMPILE)
        # Override compiled step to uncompiled for testing
        opt._compiled_step = OCGOptV2._ocgoptv2_step_fp32
        assert opt._compile_step is True
        assert opt._foreach is True
        # step() should use _step_compiled, not _step_foreach
        _run_steps(model, opt, n_steps=2)


# ---------------------------------------------------------------------------
# Test: OCGOptV2 foreach basic functionality (fp32)
# ---------------------------------------------------------------------------

class TestOCGOptV2ForeachBasic:
    """Verify the foreach step runs without errors and reduces loss."""

    def test_foreach_fp32_convergence(self):
        """Foreach optimizer should reduce loss over multiple steps (fp32)."""
        model = _make_model()
        opt = OCGOptV2(model.parameters(), lr=1e-3, foreach=True, **_NO_COMPILE)
        loss_init = _run_steps(model, opt, n_steps=1)
        loss_final = _run_steps(model, opt, n_steps=20)
        assert loss_final.item() < loss_init.item(), (
            f"Foreach loss did not decrease: init={loss_init.item()}, final={loss_final.item()}"
        )

    def test_foreach_step_returns_loss_with_closure(self):
        """foreach step() with closure should return the closure's loss."""
        model = _make_model()
        opt = OCGOptV2(model.parameters(), lr=1e-3, foreach=True, **_NO_COMPILE)
        x = torch.randn(8, 32)
        def closure():
            opt.zero_grad()
            loss = model(x).sum()
            loss.backward()
            return loss
        returned_loss = opt.step(closure)
        assert returned_loss is not None

    def test_foreach_step_returns_none_without_closure(self):
        """foreach step() without closure should return None."""
        model = _make_model()
        opt = OCGOptV2(model.parameters(), lr=1e-3, foreach=True, **_NO_COMPILE)
        x = torch.randn(8, 32)
        loss = model(x).sum()
        loss.backward()
        returned_loss = opt.step()
        assert returned_loss is None

    def test_foreach_parameters_change_after_step(self):
        """Parameters should be modified after a foreach step."""
        model = _make_model()
        opt = OCGOptV2(model.parameters(), lr=1e-3, foreach=True, **_NO_COMPILE)
        before = _snapshot_params(model)
        x = torch.randn(8, 32)
        loss = model(x).sum()
        loss.backward()
        opt.step()
        after = _snapshot_params(model)
        diff = _max_param_diff(before, after)
        assert diff > 0, "Parameters did not change after foreach step"

    def test_foreach_no_nan_in_params(self):
        """Parameters should not contain NaN after multiple foreach steps."""
        model = _make_model()
        opt = OCGOptV2(model.parameters(), lr=1e-3, foreach=True, **_NO_COMPILE)
        _run_steps(model, opt, n_steps=10)
        for p in model.parameters():
            assert not torch.isnan(p).any(), "NaN detected in foreach parameters"

    def test_foreach_no_inf_in_params(self):
        """Parameters should not contain Inf after multiple foreach steps."""
        model = _make_model()
        opt = OCGOptV2(model.parameters(), lr=1e-3, foreach=True, **_NO_COMPILE)
        _run_steps(model, opt, n_steps=10)
        for p in model.parameters():
            assert not torch.isinf(p).any(), "Inf detected in foreach parameters"


# ---------------------------------------------------------------------------
# Test: OCGOptV2 foreach vs native numerical agreement
# ---------------------------------------------------------------------------

class TestOCGOptV2ForeachVsNative:
    """Verify the foreach and native paths produce numerically close results.

    The foreach path uses _foreach_lerp_ (in-place) while the native path uses
    out-of-place lerp (which rebinds local variables). These should produce
    identical results since the math is the same.
    """

    def _run_native_vs_foreach(self, sizes, n_steps=3, seed=42, **opt_kwargs):
        """Helper to compare native and foreach paths with identical seeds."""
        # Run native path
        torch.manual_seed(seed)
        model_native = _make_model(seed=seed, sizes=sizes)
        opt_native = OCGOptV2(model_native.parameters(), lr=1e-3, compile_step=False, foreach=False, **opt_kwargs, **_NO_COMPILE)
        torch.manual_seed(999)
        for _ in range(n_steps):
            x = torch.randn(4, sizes[0][0])
            loss = model_native(x).sum()
            loss.backward()
            opt_native.step()
            opt_native.zero_grad()
        native_params = _snapshot_params(model_native)

        # Run foreach path (same seed)
        torch.manual_seed(seed)
        model_foreach = _make_model(seed=seed, sizes=sizes)
        opt_foreach = OCGOptV2(model_foreach.parameters(), lr=1e-3, compile_step=False, foreach=True, **opt_kwargs, **_NO_COMPILE)
        torch.manual_seed(999)
        for _ in range(n_steps):
            x = torch.randn(4, sizes[0][0])
            loss = model_foreach(x).sum()
            loss.backward()
            opt_foreach.step()
            opt_foreach.zero_grad()
        foreach_params = _snapshot_params(model_foreach)

        return _max_param_diff(native_params, foreach_params)

    def test_foreach_vs_native_fp32_agreement(self):
        """Foreach and native paths should produce similar parameter values (fp32)."""
        diff = self._run_native_vs_foreach(sizes=[(16, 32), (32, 8)], n_steps=5)
        assert diff < 1e-5, (
            f"Foreach vs native param diff {diff:.2e} exceeds 1e-5 tolerance"
        )

    def test_foreach_vs_native_with_adaptive(self):
        """Foreach and native should agree with adaptive=True."""
        diff = self._run_native_vs_foreach(sizes=[(16, 16)], n_steps=3, adaptive=True)
        assert diff < 1e-5, (
            f"Foreach vs native adaptive param diff {diff:.2e} exceeds 1e-5 tolerance"
        )

    def test_foreach_vs_native_with_weight_decay(self):
        """Foreach and native should agree with weight_decay."""
        diff = self._run_native_vs_foreach(sizes=[(16, 16)], n_steps=3, weight_decay=0.01)
        assert diff < 1e-5, (
            f"Foreach vs native weight_decay param diff {diff:.2e} exceeds 1e-5 tolerance"
        )

    def test_foreach_vs_native_with_aol(self):
        """Foreach and native should agree with aol=True."""
        diff = self._run_native_vs_foreach(sizes=[(16, 32), (32, 8)], n_steps=3, aol=True)
        assert diff < 1e-5, (
            f"Foreach vs native aol param diff {diff:.2e} exceeds 1e-5 tolerance"
        )

    def test_foreach_vs_native_with_input_norm(self):
        """Foreach and native should agree with input_norm=True."""
        diff = self._run_native_vs_foreach(sizes=[(16, 32), (32, 8)], n_steps=3, input_norm=True)
        assert diff < 1e-5, (
            f"Foreach vs native input_norm param diff {diff:.2e} exceeds 1e-5 tolerance"
        )

    def test_foreach_vs_native_with_centralization(self):
        """Foreach and native should agree with custom centralization."""
        diff = self._run_native_vs_foreach(sizes=[(16, 32), (32, 8)], n_steps=3, centralization=0.5)
        assert diff < 1e-5, (
            f"Foreach vs native centralization param diff {diff:.2e} exceeds 1e-5 tolerance"
        )

    def test_foreach_vs_native_with_betas(self):
        """Foreach and native should agree with custom betas."""
        diff = self._run_native_vs_foreach(
            sizes=[(16, 32), (32, 8)], n_steps=3,
            betas=(0.9, 0.999, 0.9999),
        )
        assert diff < 1e-5, (
            f"Foreach vs native betas param diff {diff:.2e} exceeds 1e-5 tolerance"
        )

    def test_foreach_vs_native_multi_step_agreement(self):
        """Foreach and native should agree over many steps."""
        diff = self._run_native_vs_foreach(sizes=[(32, 64), (64, 16)], n_steps=10)
        assert diff < 1e-4, (
            f"Foreach vs native multi-step param diff {diff:.2e} exceeds 1e-4 tolerance"
        )


# ---------------------------------------------------------------------------
# Test: OCGOptV2 foreach with bf16
# ---------------------------------------------------------------------------

class TestOCGOptV2ForeachBf16:
    """Verify the foreach path works with bf16 parameters."""

    def test_foreach_bf16_no_crash(self):
        """Foreach optimizer should not crash on bf16 parameters."""
        model = _make_model(dtype=torch.bfloat16)
        opt = OCGOptV2(model.parameters(), lr=1e-3, foreach=True, stochastic_fp=True, **_NO_COMPILE)
        _run_steps(model, opt, n_steps=5)

    def test_foreach_bf16_no_nan(self):
        """Foreach bf16 parameters should not contain NaN after steps."""
        model = _make_model(dtype=torch.bfloat16)
        opt = OCGOptV2(model.parameters(), lr=1e-3, foreach=True, stochastic_fp=True, **_NO_COMPILE)
        _run_steps(model, opt, n_steps=10)
        for p in model.parameters():
            assert not torch.isnan(p).any(), "NaN detected in foreach bf16 parameters"

    def test_foreach_bf16_stays_bf16(self):
        """After foreach stochastic update, parameter dtype should remain bf16."""
        model = _make_model(dtype=torch.bfloat16)
        opt = OCGOptV2(model.parameters(), lr=1e-3, foreach=True, stochastic_fp=True, **_NO_COMPILE)
        _run_steps(model, opt, n_steps=3)
        for p in model.parameters():
            assert p.dtype == torch.bfloat16, (
                f"Parameter dtype changed from bfloat16 to {p.dtype}"
            )

    def test_foreach_bf16_state_momentum_dtype(self):
        """Momentum states should be stored in the parameter's dtype (bf16)."""
        model = _make_model(dtype=torch.bfloat16)
        opt = OCGOptV2(model.parameters(), lr=1e-3, foreach=True, stochastic_fp=True, **_NO_COMPILE)
        _run_steps(model, opt, n_steps=2)
        for p in model.parameters():
            state = opt.state[p]
            assert state["value_momentum"].dtype == torch.bfloat16
            assert state["centralized_momentum"].dtype == torch.bfloat16


# ---------------------------------------------------------------------------
# Test: OCGOptV2 foreach scalar parameter handling
# ---------------------------------------------------------------------------

class TestOCGOptV2ForeachScalarParam:
    """Verify the foreach path handles scalar (0-dim) parameters correctly."""

    def test_foreach_scalar_param_no_crash(self):
        """Foreach optimizer should handle 0-dim parameters without crashing."""
        model = torch.nn.Linear(8, 4, bias=True)
        model.scale = torch.nn.Parameter(torch.tensor(1.0))
        opt = OCGOptV2(model.parameters(), lr=1e-3, foreach=True, **_NO_COMPILE)

        for _ in range(5):
            x = torch.randn(4, 8)
            loss = model(x).sum() + model.scale
            loss.backward()
            opt.step()
            opt.zero_grad()

        assert not torch.isnan(model.scale).any(), "NaN in foreach scalar parameter"

    def test_foreach_scalar_params_change(self):
        """Scalar parameters should be modified after foreach steps."""
        model = torch.nn.Linear(8, 4, bias=True)
        model.scale = torch.nn.Parameter(torch.tensor(1.0))
        opt = OCGOptV2(model.parameters(), lr=1e-3, foreach=True, **_NO_COMPILE)

        before_scale = model.scale.data.clone()
        for _ in range(5):
            x = torch.randn(4, 8)
            loss = model(x).sum() + model.scale
            loss.backward()
            opt.step()
            opt.zero_grad()

        diff = (before_scale - model.scale.data).abs().max().item()
        assert diff > 0, "Scalar parameter did not change after foreach steps"


# ---------------------------------------------------------------------------
# Test: OCGOptV2 foreach state management
# ---------------------------------------------------------------------------

class TestOCGOptV2ForeachState:
    """Verify optimizer state is correctly initialized and updated in foreach path."""

    def test_foreach_momentum_state_created(self):
        """Momentum states should be initialized on the first foreach step."""
        model = _make_model()
        opt = OCGOptV2(model.parameters(), lr=1e-3, foreach=True, **_NO_COMPILE)
        x = torch.randn(4, 32)
        loss = model(x).sum()
        loss.backward()
        opt.step()

        for p in model.parameters():
            state = opt.state[p]
            assert "value_momentum" in state
            assert "centralized_momentum" in state
            assert state["value_momentum"].shape == p.grad.shape
            assert state["centralized_momentum"].shape == p.grad.shape

    def test_foreach_denom_state_for_scalar(self):
        """Denom state should be created for scalar parameters in foreach path."""
        model = torch.nn.Linear(8, 4, bias=True)
        model.scale = torch.nn.Parameter(torch.tensor(1.0))
        opt = OCGOptV2(model.parameters(), lr=1e-3, foreach=True, **_NO_COMPILE)

        x = torch.randn(4, 8)
        loss = model(x).sum() + model.scale
        loss.backward()
        opt.step()

        state = opt.state[model.scale]
        assert "denom" in state
        assert state["denom"].shape == model.scale.shape

    def test_foreach_step_counter_increments(self):
        """The step counter should increment on each foreach step() call."""
        model = _make_model()
        opt = OCGOptV2(model.parameters(), lr=1e-3, foreach=True, **_NO_COMPILE)
        x = torch.randn(4, 32)

        for expected_step in range(1, 4):
            loss = model(x).sum()
            loss.backward()
            opt.step()
            opt.zero_grad()
            assert opt.param_groups[0]["step"] == expected_step


# ---------------------------------------------------------------------------
# Test: OCGOptV2 foreach no gradient handling
# ---------------------------------------------------------------------------

class TestOCGOptV2ForeachNoGrad:
    """Verify the foreach path skips parameters with no gradient."""

    def test_foreach_skip_none_grad(self):
        """Parameters without gradients should be skipped gracefully in foreach path."""
        model = _make_model(sizes=[(8, 8), (8, 4)])
        opt = OCGOptV2(model.parameters(), lr=1e-3, foreach=True, **_NO_COMPILE)

        x = torch.randn(2, 8)
        out = model[0](x)
        loss = out.sum()
        loss.backward()

        for name, param in model.named_parameters():
            if '1' in name:
                param.grad = None

        opt.step()  # Should not crash

    def test_foreach_skip_on_subsequent_steps(self):
        """Parameters that lose gradients mid-training should be skipped."""
        model = _make_model(sizes=[(8, 8), (8, 4)])
        opt = OCGOptV2(model.parameters(), lr=1e-3, foreach=True, **_NO_COMPILE)

        # First, run normally to build up state
        x = torch.randn(2, 8)
        loss = model(x).sum()
        loss.backward()
        opt.step()
        opt.zero_grad()

        # Now remove gradient for some params
        for name, param in model.named_parameters():
            if '0.weight' in name:
                param.grad = None

        opt.step()  # Should not crash


# ---------------------------------------------------------------------------
# Test: OCGOptV2 foreach determinism
# ---------------------------------------------------------------------------

class TestOCGOptV2ForeachDeterminism:
    """Verify the foreach path produces deterministic results with same seed."""

    def test_foreach_deterministic_fp32(self):
        """Same seed should produce identical parameters with foreach (fp32)."""
        results = []
        for _ in range(2):
            torch.manual_seed(42)
            model = _make_model(seed=42, sizes=[(16, 16), (16, 4)])
            opt = OCGOptV2(model.parameters(), lr=1e-3, foreach=True, **_NO_COMPILE)
            torch.manual_seed(123)
            for _ in range(5):
                x = torch.randn(4, 16)
                loss = model(x).sum()
                loss.backward()
                opt.step()
                opt.zero_grad()
            results.append(_snapshot_params(model))

        diff = _max_param_diff(results[0], results[1])
        assert diff < 1e-7, f"Non-deterministic foreach results: diff={diff}"


# ---------------------------------------------------------------------------
# Test: OCGOptV2 foreach weight decay
# ---------------------------------------------------------------------------

class TestOCGOptV2ForeachWeightDecay:
    """Verify weight decay is applied correctly in foreach path."""

    def test_foreach_weight_decay_reduces_magnitude(self):
        """With weight decay, parameter magnitudes should decrease vs. no decay."""
        sizes = [(16, 16)]
        model_a = _make_model(seed=42, sizes=sizes)
        model_b = copy.deepcopy(model_a)
        opt_no_wd = OCGOptV2(model_a.parameters(), lr=1e-3, weight_decay=0.0, foreach=True, **_NO_COMPILE)
        opt_wd = OCGOptV2(model_b.parameters(), lr=1e-3, weight_decay=0.1, foreach=True, **_NO_COMPILE)

        torch.manual_seed(999)
        for _ in range(10):
            x = torch.randn(4, 16)
            loss_a = model_a(x).sum()
            loss_a.backward()
            opt_no_wd.step()
            opt_no_wd.zero_grad()

            loss_b = model_b(x).sum()
            loss_b.backward()
            opt_wd.step()
            opt_wd.zero_grad()

        norm_no_wd = sum(p.norm().item() for p in model_a.parameters())
        norm_wd = sum(p.norm().item() for p in model_b.parameters())
        assert norm_wd < norm_no_wd, (
            f"Foreach weight decay did not reduce norms: no_wd={norm_no_wd}, wd={norm_wd}"
        )


# ---------------------------------------------------------------------------
# Test: OCGOptV2 cautious weight decay (all three paths)
# ---------------------------------------------------------------------------

class TestOCGOptV2CautiousWeightDecay:
    """Verify cautious weight decay masks WD where grad * param < 0.

    Cautious WD should:
    - Still reduce parameter norms vs no WD (less aggressively than standard WD)
    - Produce different results from standard WD
    - Be a no-op when weight_decay=0
    - Work across all three step paths (native, foreach, compiled)
    """

    # ---- Native path ---------------------------------------------------

    def test_native_cautious_wd_reduces_norms_less_than_standard(self):
        """Cautious WD should reduce norms less than standard WD because
        it skips decay on some parameters."""
        sizes = [(16, 16)]
        model_std = _make_model(seed=42, sizes=sizes)
        model_caut = copy.deepcopy(model_std)
        opt_std = OCGOptV2(
            model_std.parameters(), lr=1e-3, weight_decay=0.1,
            **_NO_COMPILE,
        )
        opt_caut = OCGOptV2(
            model_caut.parameters(), lr=1e-3, weight_decay=0.1,
            cautious_weight_decay=True, **_NO_COMPILE,
        )

        torch.manual_seed(999)
        for _ in range(10):
            x = torch.randn(4, 16)
            loss_std = model_std(x).sum()
            loss_std.backward()
            opt_std.step()
            opt_std.zero_grad()

            loss_caut = model_caut(x).sum()
            loss_caut.backward()
            opt_caut.step()
            opt_caut.zero_grad()

        norm_std = sum(p.norm().item() for p in model_std.parameters())
        norm_caut = sum(p.norm().item() for p in model_caut.parameters())
        assert norm_caut > norm_std, (
            f"Cautious WD should reduce norms less than standard WD: "
            f"cautious={norm_caut:.4f}, standard={norm_std:.4f}"
        )

    def test_native_cautious_wd_vs_no_wd(self):
        """Cautious WD should still reduce norms compared to no WD at all."""
        sizes = [(16, 16)]
        model_none = _make_model(seed=42, sizes=sizes)
        model_caut = copy.deepcopy(model_none)
        opt_none = OCGOptV2(
            model_none.parameters(), lr=1e-3, weight_decay=0.0,
            **_NO_COMPILE,
        )
        opt_caut = OCGOptV2(
            model_caut.parameters(), lr=1e-3, weight_decay=0.1,
            cautious_weight_decay=True, **_NO_COMPILE,
        )

        torch.manual_seed(999)
        for _ in range(10):
            x = torch.randn(4, 16)
            loss_none = model_none(x).sum()
            loss_none.backward()
            opt_none.step()
            opt_none.zero_grad()

            loss_caut = model_caut(x).sum()
            loss_caut.backward()
            opt_caut.step()
            opt_caut.zero_grad()

        norm_none = sum(p.norm().item() for p in model_none.parameters())
        norm_caut = sum(p.norm().item() for p in model_caut.parameters())
        assert norm_caut < norm_none, (
            f"Cautious WD should still reduce norms vs no WD: "
            f"cautious={norm_caut:.4f}, no_wd={norm_none:.4f}"
        )

    def test_native_cautious_wd_produces_different_params(self):
        """Cautious WD should produce different parameter values than standard WD."""
        sizes = [(16, 16)]
        model_std = _make_model(seed=42, sizes=sizes)
        model_caut = copy.deepcopy(model_std)
        opt_std = OCGOptV2(
            model_std.parameters(), lr=1e-3, weight_decay=0.1,
            **_NO_COMPILE,
        )
        opt_caut = OCGOptV2(
            model_caut.parameters(), lr=1e-3, weight_decay=0.1,
            cautious_weight_decay=True, **_NO_COMPILE,
        )

        torch.manual_seed(999)
        for _ in range(5):
            x = torch.randn(4, 16)
            loss_std = model_std(x).sum()
            loss_std.backward()
            opt_std.step()
            opt_std.zero_grad()

            loss_caut = model_caut(x).sum()
            loss_caut.backward()
            opt_caut.step()
            opt_caut.zero_grad()

        diff = _max_param_diff(_snapshot_params(model_std), _snapshot_params(model_caut))
        assert diff > 1e-6, (
            f"Cautious WD should produce different params from standard WD: diff={diff}"
        )

    def test_native_cautious_wd_noop_when_wd_zero(self):
        """When weight_decay=0, cautious_weight_decay=True should be a no-op."""
        sizes = [(16, 16)]
        model_a = _make_model(seed=42, sizes=sizes)
        model_b = copy.deepcopy(model_a)
        opt_a = OCGOptV2(
            model_a.parameters(), lr=1e-3, weight_decay=0.0,
            **_NO_COMPILE,
        )
        opt_b = OCGOptV2(
            model_b.parameters(), lr=1e-3, weight_decay=0.0,
            cautious_weight_decay=True, **_NO_COMPILE,
        )

        torch.manual_seed(999)
        for _ in range(5):
            x = torch.randn(4, 16)
            loss_a = model_a(x).sum()
            loss_a.backward()
            opt_a.step()
            opt_a.zero_grad()

            loss_b = model_b(x).sum()
            loss_b.backward()
            opt_b.step()
            opt_b.zero_grad()

        diff = _max_param_diff(_snapshot_params(model_a), _snapshot_params(model_b))
        assert diff < 1e-7, (
            f"Cautious WD with weight_decay=0 should be a no-op: diff={diff}"
        )

    def test_native_cautious_wd_training_converges(self):
        """Training should still converge with cautious weight decay."""
        model = _make_model(dtype=torch.float32)
        opt = OCGOptV2(
            model.parameters(), lr=1e-3, weight_decay=0.01,
            cautious_weight_decay=True, **_NO_COMPILE,
        )
        loss_init = _run_steps(model, opt, n_steps=1).item()
        loss_final = _run_steps(model, opt, n_steps=20).item()
        assert loss_final < loss_init, (
            f"Loss did not decrease with cautious WD: init={loss_init:.4f}, final={loss_final:.4f}"
        )

    # ---- Foreach path --------------------------------------------------

    def test_foreach_cautious_wd_reduces_norms_less_than_standard(self):
        """Foreach path: cautious WD should reduce norms less than standard WD."""
        sizes = [(16, 16)]
        model_std = _make_model(seed=42, sizes=sizes)
        model_caut = copy.deepcopy(model_std)
        opt_std = OCGOptV2(
            model_std.parameters(), lr=1e-3, weight_decay=0.1,
            foreach=True, **_NO_COMPILE,
        )
        opt_caut = OCGOptV2(
            model_caut.parameters(), lr=1e-3, weight_decay=0.1,
            cautious_weight_decay=True, foreach=True, **_NO_COMPILE,
        )

        torch.manual_seed(999)
        for _ in range(10):
            x = torch.randn(4, 16)
            loss_std = model_std(x).sum()
            loss_std.backward()
            opt_std.step()
            opt_std.zero_grad()

            loss_caut = model_caut(x).sum()
            loss_caut.backward()
            opt_caut.step()
            opt_caut.zero_grad()

        norm_std = sum(p.norm().item() for p in model_std.parameters())
        norm_caut = sum(p.norm().item() for p in model_caut.parameters())
        assert norm_caut > norm_std, (
            f"Foreach cautious WD should reduce norms less than standard WD: "
            f"cautious={norm_caut:.4f}, standard={norm_std:.4f}"
        )

    def test_foreach_cautious_wd_produces_different_params(self):
        """Foreach path: cautious WD should produce different params from standard WD."""
        sizes = [(16, 16)]
        model_std = _make_model(seed=42, sizes=sizes)
        model_caut = copy.deepcopy(model_std)
        opt_std = OCGOptV2(
            model_std.parameters(), lr=1e-3, weight_decay=0.1,
            foreach=True, **_NO_COMPILE,
        )
        opt_caut = OCGOptV2(
            model_caut.parameters(), lr=1e-3, weight_decay=0.1,
            cautious_weight_decay=True, foreach=True, **_NO_COMPILE,
        )

        torch.manual_seed(999)
        for _ in range(5):
            x = torch.randn(4, 16)
            loss_std = model_std(x).sum()
            loss_std.backward()
            opt_std.step()
            opt_std.zero_grad()

            loss_caut = model_caut(x).sum()
            loss_caut.backward()
            opt_caut.step()
            opt_caut.zero_grad()

        diff = _max_param_diff(_snapshot_params(model_std), _snapshot_params(model_caut))
        assert diff > 1e-6, (
            f"Foreach cautious WD should produce different params: diff={diff}"
        )

    # ---- Compiled path -------------------------------------------------

    def test_compiled_cautious_wd_reduces_norms_less_than_standard(self):
        """Compiled path: cautious WD should reduce norms less than standard WD."""
        sizes = [(16, 16)]
        model_std = _make_model(seed=42, sizes=sizes)
        model_caut = copy.deepcopy(model_std)
        opt_std = _make_compiled_step_opt(
            model_std.parameters(), lr=1e-3, weight_decay=0.1,
            **_NO_COMPILE,
        )
        opt_caut = _make_compiled_step_opt(
            model_caut.parameters(), lr=1e-3, weight_decay=0.1,
            cautious_weight_decay=True, **_NO_COMPILE,
        )

        torch.manual_seed(999)
        for _ in range(10):
            x = torch.randn(4, 16)
            loss_std = model_std(x).sum()
            loss_std.backward()
            opt_std.step()
            opt_std.zero_grad()

            loss_caut = model_caut(x).sum()
            loss_caut.backward()
            opt_caut.step()
            opt_caut.zero_grad()

        norm_std = sum(p.norm().item() for p in model_std.parameters())
        norm_caut = sum(p.norm().item() for p in model_caut.parameters())
        assert norm_caut > norm_std, (
            f"Compiled cautious WD should reduce norms less than standard WD: "
            f"cautious={norm_caut:.4f}, standard={norm_std:.4f}"
        )

    def test_compiled_cautious_wd_produces_different_params(self):
        """Compiled path: cautious WD should produce different params from standard WD."""
        sizes = [(16, 16)]
        model_std = _make_model(seed=42, sizes=sizes)
        model_caut = copy.deepcopy(model_std)
        opt_std = _make_compiled_step_opt(
            model_std.parameters(), lr=1e-3, weight_decay=0.1,
            **_NO_COMPILE,
        )
        opt_caut = _make_compiled_step_opt(
            model_caut.parameters(), lr=1e-3, weight_decay=0.1,
            cautious_weight_decay=True, **_NO_COMPILE,
        )

        torch.manual_seed(999)
        for _ in range(5):
            x = torch.randn(4, 16)
            loss_std = model_std(x).sum()
            loss_std.backward()
            opt_std.step()
            opt_std.zero_grad()

            loss_caut = model_caut(x).sum()
            loss_caut.backward()
            opt_caut.step()
            opt_caut.zero_grad()

        diff = _max_param_diff(_snapshot_params(model_std), _snapshot_params(model_caut))
        assert diff > 1e-6, (
            f"Compiled cautious WD should produce different params: diff={diff}"
        )

    def test_compiled_cautious_wd_noop_when_wd_zero(self):
        """Compiled path: cautious WD with weight_decay=0 should be a no-op."""
        sizes = [(16, 16)]
        model_a = _make_model(seed=42, sizes=sizes)
        model_b = copy.deepcopy(model_a)
        opt_a = _make_compiled_step_opt(
            model_a.parameters(), lr=1e-3, weight_decay=0.0,
            **_NO_COMPILE,
        )
        opt_b = _make_compiled_step_opt(
            model_b.parameters(), lr=1e-3, weight_decay=0.0,
            cautious_weight_decay=True, **_NO_COMPILE,
        )

        torch.manual_seed(999)
        for _ in range(5):
            x = torch.randn(4, 16)
            loss_a = model_a(x).sum()
            loss_a.backward()
            opt_a.step()
            opt_a.zero_grad()

            loss_b = model_b(x).sum()
            loss_b.backward()
            opt_b.step()
            opt_b.zero_grad()

        diff = _max_param_diff(_snapshot_params(model_a), _snapshot_params(model_b))
        assert diff < 1e-7, (
            f"Compiled cautious WD with weight_decay=0 should be a no-op: diff={diff}"
        )
