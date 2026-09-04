"""Tests for OCGOptV2 compiled spectral clip fallback.

On some environments (notably Windows with certain Triton/Python
combinations), ``torch.compile`` of ``gram_newton_schulz_2step`` fails at the
*first call* with errors such as::

    torch._inductor.exc.InductorError: SystemError:
    PY_SSIZE_T_CLEAN macro must be defined for '#' formats

Since ``spectral_clip_compile=True`` is the OCGOptV2 default, this crashed
training at the very first optimizer step.  The fix wraps every
``self.clip_func(...)`` call site in ``_call_clip_func`` which, on the first
failure of the compiled function, logs a one-time warning and permanently
swaps in the uncompiled ``gram_newton_schulz_2step``.

Run with:
    cd backend/sd_scripts
    python -m pytest ../custom_scheduler/tests/test_ocgoptv2_compile_fallback.py -v
"""

import importlib.util
import logging
import os
import sys
import types

import pytest
import torch

# Import directly from the module file to avoid pulling in the full
# LoraEasyCustomOptimizer package (which has heavy dependencies like
# pytorch_optimizer that may not be installed in the test environment).
_pkg_dir = os.path.join(os.path.dirname(__file__), "..", "LoraEasyCustomOptimizer")

if "LoraEasyCustomOptimizer" not in sys.modules:
    _pkg = types.ModuleType("LoraEasyCustomOptimizer")
    _pkg.__path__ = [_pkg_dir]
    _pkg.__package__ = "LoraEasyCustomOptimizer"
    sys.modules["LoraEasyCustomOptimizer"] = _pkg

# Load utils first (needed by ocgoptv2 via ``from .utils import copy_stochastic_``)
_utils_spec = importlib.util.spec_from_file_location(
    "LoraEasyCustomOptimizer.utils", os.path.join(_pkg_dir, "utils.py")
)
_utils_mod = importlib.util.module_from_spec(_utils_spec)
sys.modules["LoraEasyCustomOptimizer.utils"] = _utils_mod
_utils_spec.loader.exec_module(_utils_mod)

_ocgoptv2_spec = importlib.util.spec_from_file_location(
    "LoraEasyCustomOptimizer.ocgoptv2", os.path.join(_pkg_dir, "ocgoptv2.py")
)
_ocgoptv2 = importlib.util.module_from_spec(_ocgoptv2_spec)
sys.modules["LoraEasyCustomOptimizer.ocgoptv2"] = _ocgoptv2
_ocgoptv2_spec.loader.exec_module(_ocgoptv2)

OCGOptV2 = _ocgoptv2.OCGOptV2
gram_newton_schulz_2step = _ocgoptv2.gram_newton_schulz_2step


class _SimulatedCompileError(RuntimeError):
    """Stands in for torch._inductor.exc.InductorError raised at first call."""


def _failing_compiled_clip(M, ortho_dtype=None):
    """Mimics a compiled clip_func whose first invocation fails to build/load
    its Triton kernel (e.g. the PY_SSIZE_T_CLEAN Windows error)."""
    raise _SimulatedCompileError(
        "InductorError: SystemError: PY_SSIZE_T_CLEAN macro must be defined "
        "for '#' formats"
    )


def _make_model(seed=42, sizes=((16, 16), (16, 8)), device="cuda"):
    torch.manual_seed(seed)
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(torch.nn.Linear(sizes[i][0], sizes[i][1], device=device))
    return torch.nn.Sequential(*layers)


def _run_steps(model, opt, n_steps=3):
    loss = None
    for _ in range(n_steps):
        opt.zero_grad()
        x = torch.randn(4, model[0].in_features, device=next(model.parameters()).device)
        loss = model(x).pow(2).mean()
        loss.backward()
        opt.step()
    return loss


# ---------------------------------------------------------------------------
# __init__ flag wiring
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestClipFuncFlagWiring:
    def test_default_is_compiled(self):
        model = _make_model()
        opt = OCGOptV2(model.parameters(), lr=1e-3)
        assert opt.clip_func is not None
        assert opt._clip_func_compiled is True

    def test_uncompiled_flag(self):
        model = _make_model()
        opt = OCGOptV2(model.parameters(), lr=1e-3, spectral_clip_compile=False)
        assert opt.clip_func is gram_newton_schulz_2step
        assert opt._clip_func_compiled is False

    def test_compile_step_no_clip_func(self):
        model = _make_model()
        opt = OCGOptV2(model.parameters(), lr=1e-3, compile_step=True)
        assert opt.clip_func is None
        assert opt._clip_func_compiled is False


# ---------------------------------------------------------------------------
# Fallback behavior (native + foreach paths)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestCompileFailureFallback:
    @pytest.mark.parametrize("foreach", [False, True])
    def test_fallback_after_first_failure(self, foreach, caplog):
        """A compiled clip_func that fails at first call must be swapped for
        the uncompiled implementation and training must continue."""
        model = _make_model()
        opt = OCGOptV2(model.parameters(), lr=1e-3, foreach=foreach)
        # Simulate an environment where the compiled kernel fails at runtime.
        opt.clip_func = _failing_compiled_clip
        opt._clip_func_compiled = True

        with caplog.at_level(logging.WARNING):
            _run_steps(model, opt, n_steps=3)

        assert opt._clip_func_compiled is False
        assert opt.clip_func is gram_newton_schulz_2step
        # One-time warning mentioning the fallback.
        fallback_warnings = [
            r for r in caplog.records
            if "compiled spectral clip" in r.getMessage()
            and "Falling back" in r.getMessage()
        ]
        assert len(fallback_warnings) == 1
        for p in model.parameters():
            assert torch.isfinite(p).all()

    @pytest.mark.parametrize("foreach", [False, True])
    def test_fallback_results_match_uncompiled(self, foreach, caplog):
        """Training with the fallback must match training with
        spectral_clip_compile=False from identical seeds."""
        kwargs = dict(lr=1e-3, foreach=foreach)

        torch.manual_seed(123)
        model_a = _make_model(seed=7, sizes=((16, 16), (16, 8)))
        opt_a = OCGOptV2(model_a.parameters(), **kwargs)
        opt_a.clip_func = _failing_compiled_clip  # forces the fallback path
        opt_a._clip_func_compiled = True
        with caplog.at_level(logging.WARNING):
            _run_steps(model_a, opt_a, n_steps=5)

        torch.manual_seed(123)
        model_b = _make_model(seed=7, sizes=((16, 16), (16, 8)))
        opt_b = OCGOptV2(model_b.parameters(), spectral_clip_compile=False, **kwargs)
        _run_steps(model_b, opt_b, n_steps=5)

        for pa, pb in zip(model_a.parameters(), model_b.parameters()):
            assert torch.allclose(pa, pb, atol=1e-6), (
                "Fallback path diverged from the uncompiled spectral clip path."
            )

    def test_warning_is_one_time(self, caplog):
        """The warning must be logged exactly once, even across many steps."""
        model = _make_model()
        opt = OCGOptV2(model.parameters(), lr=1e-3)
        opt.clip_func = _failing_compiled_clip
        opt._clip_func_compiled = True

        with caplog.at_level(logging.WARNING):
            _run_steps(model, opt, n_steps=8)

        fallback_warnings = [
            r for r in caplog.records if "Falling back" in r.getMessage()
        ]
        assert len(fallback_warnings) == 1

    def test_uncompiled_failure_reraises(self):
        """If clip_func is already uncompiled, failures must propagate."""
        model = _make_model()
        opt = OCGOptV2(model.parameters(), lr=1e-3, spectral_clip_compile=False)
        opt.clip_func = _failing_compiled_clip  # pretend this is "the" impl

        with pytest.raises(_SimulatedCompileError):
            opt._call_clip_func(torch.randn(4, 4, device="cuda"))

    def test_no_failure_keeps_compiled(self, caplog):
        """When the compiled function works, no fallback occurs."""
        model = _make_model()
        opt = OCGOptV2(model.parameters(), lr=1e-3)

        with caplog.at_level(logging.WARNING):
            _run_steps(model, opt, n_steps=2)

        assert opt._clip_func_compiled is True
        assert not [r for r in caplog.records if "Falling back" in r.getMessage()]


# ---------------------------------------------------------------------------
# End-to-end with the real torch.compile wrapper on CUDA
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestRealCompiledClipEndToEnd:
    def test_default_config_completes(self):
        """The default OCGOptV2 config (spectral_clip_compile=True) must
        complete training steps: either the compiled kernel works, or the
        fallback kicks in.  Either way, parameters stay finite and the
        optimizer advances."""
        model = _make_model()
        before = [p.detach().clone() for p in model.parameters()]
        opt = OCGOptV2(model.parameters(), lr=1e-3)

        _run_steps(model, opt, n_steps=3)

        assert opt._clip_func_compiled in (True, False)
        for p, p0 in zip(model.parameters(), before):
            assert torch.isfinite(p).all()
            assert not torch.equal(p, p0), "Parameter was not updated."

    def test_bf16_dtype_default_clip_runs(self):
        """Default spectral_clip_dtype (bfloat16 since the 2-step NS change)
        must run through the (possibly compiled) clip function without
        crashing, on both the native and foreach paths."""
        for foreach in (False, True):
            model = _make_model()
            opt = OCGOptV2(
                model.parameters(), lr=1e-3, foreach=foreach,
                spectral_clip_dtype="torch.bfloat16",
            )
            assert opt.param_groups[0]["spectral_clip_dtype"] == torch.bfloat16
            _run_steps(model, opt, n_steps=2)
            for p in model.parameters():
                assert torch.isfinite(p).all()
