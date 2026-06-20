"""Tests for OCGOptV2 fixes: A1 (reset), A2 (fp16 upcast), A3 (slow_beta hoist), C1 (slow_beta stability).

Run from the repo root:
    python -m pytest backend/custom_scheduler/tests/test_ocgoptv2_fixes2.py -v

These tests assume CUDA availability (per project convention).
"""

import math
import sys
import os

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from backend.custom_scheduler.LoraEasyCustomOptimizer.ocgoptv2 import (
    OCGOptV2,
    _slow_beta,
    _LOW_PRECISION_DTYPES,
)


# ---------------------------------------------------------------------------
# C1: _slow_beta numerical stability
# ---------------------------------------------------------------------------


class TestSlowBetaStability:
    """Verify _slow_beta matches the naive formula and is stable near beta≈1."""

    @pytest.mark.parametrize("beta", [0.9, 0.95, 0.99, 0.999, 0.9999])
    @pytest.mark.parametrize("step", [1, 2, 3, 5, 10, 50, 100])
    def test_slow_beta_matches_naive_formula(self, beta, step):
        """_slow_beta must equal (beta**step - beta)/(beta**step - 1) for step>1."""
        if step <= 1:
            assert _slow_beta(beta, step) == 0.0
            return
        naive = (beta ** step - beta) / (beta ** step - 1.0)
        stable = _slow_beta(beta, step)
        assert stable == pytest.approx(naive, rel=1e-9, abs=1e-12), (
            f"beta={beta} step={step}: naive={naive!r} stable={stable!r}"
        )

    def test_slow_beta_step1_is_zero(self):
        """At step 1 there is no averaging yet — slow_beta must be 0."""
        for beta in [0.5, 0.9, 0.999, 0.9999999]:
            assert _slow_beta(beta, 1) == 0.0

    def test_slow_beta_stable_near_one(self):
        """For beta very close to 1, the naive formula loses precision but
        _slow_beta must remain finite and in [0, 1]."""
        beta = 0.9999999  # documented default
        for step in [2, 3, 10, 100, 1000]:
            val = _slow_beta(beta, step)
            assert math.isfinite(val), f"non-finite at beta={beta} step={step}: {val}"
            assert 0.0 <= val <= 1.0, f"out of range at beta={beta} step={step}: {val}"

    def test_slow_beta_monotonic_in_step(self):
        """slow_beta should be non-decreasing in step for 0<beta<1
        (more averaging as steps accumulate)."""
        beta = 0.99
        prev = _slow_beta(beta, 1)
        for step in range(2, 50):
            cur = _slow_beta(beta, step)
            assert cur >= prev - 1e-12, f"decreased at step={step}: {prev} -> {cur}"
            prev = cur

    def test_slow_beta_beta_ge_one(self):
        """beta >= 1 is degenerate (pure momentum); should return 1.0."""
        assert _slow_beta(1.0, 5) == 1.0
        assert _slow_beta(1.5, 5) == 1.0


# ---------------------------------------------------------------------------
# A3: native slow_beta hoisted to group level (consistency + perf)
# ---------------------------------------------------------------------------


class TestSlowBetaHoist:
    """Verify native path produces identical results to compiled/foreach paths
    (which already hoisted slow_beta), confirming the hoist didn't change math."""

    @pytest.mark.parametrize("path", ["native", "foreach"])
    def test_native_matches_foreach_after_hoist(self, path):
        """Run a step on identical models; native and foreach must agree."""
        torch.manual_seed(42)
        sizes = [(32, 16), (16, 32), (8, 8)]
        # Build two identical models
        params_a = [torch.randn(s, device="cuda", requires_grad=True) for s in sizes]
        params_b = [p.detach().clone().requires_grad_(True) for p in params_a]

        # Identical gradients
        torch.manual_seed(7)
        grads = [torch.randn(s, device="cuda") for s in sizes]
        for p, g in zip(params_a, grads):
            p.grad = g.clone()
        for p, g in zip(params_b, grads):
            p.grad = g.clone()

        opt_a = OCGOptV2(params_a, lr=1e-3, compile_step=False, foreach=False)
        opt_b = OCGOptV2(params_b, lr=1e-3, compile_step=False, foreach=True)
        opt_a.step()
        opt_b.step()

        for a, b in zip(params_a, params_b):
            assert torch.allclose(a, b, atol=1e-5), (
                f"native vs foreach mismatch, max diff {(a - b).abs().max().item()}"
            )

    def test_native_slow_beta_not_recomputed_per_param(self):
        """Smoke test: native path runs multi-step without error after hoist,
        exercising the group-level slow_beta for several step counts."""
        torch.manual_seed(0)
        model = torch.nn.Linear(64, 32, device="cuda")
        opt = OCGOptV2(model.parameters(), lr=1e-3, compile_step=False, foreach=False)
        for _ in range(5):
            x = torch.randn(4, 64, device="cuda")
            model(x).sum().backward()
            opt.step()
            opt.zero_grad()
        # After 5 steps, group step counter should be 5
        assert opt.param_groups[0]["step"] == 5


# ---------------------------------------------------------------------------
# A2: fp16 parameters upcast to fp32 for compute
# ---------------------------------------------------------------------------


class TestFp16Upcast:
    """Verify fp16 params are upcast to fp32 for compute (no overflow) and
    written back via plain cast (not stochastic rounding)."""

    def test_fp16_param_runs_native(self):
        """fp16 param must not overflow during a step (would if computed in fp16)."""
        torch.manual_seed(42)
        # Large-magnitude gradient that would overflow fp16 (max ~65504)
        param = torch.randn(64, 32, device="cuda", dtype=torch.float16, requires_grad=True)
        param.grad = (torch.randn(64, 32, device="cuda") * 1000.0).to(torch.float16)
        opt = OCGOptV2([param], lr=1e-4, compile_step=False, foreach=False, stochastic_fp=True)
        opt.step()  # should not raise / not produce inf
        assert torch.isfinite(param).all(), "fp16 param became non-finite"

    def test_fp16_param_runs_foreach(self):
        torch.manual_seed(42)
        param = torch.randn(64, 32, device="cuda", dtype=torch.float16, requires_grad=True)
        param.grad = (torch.randn(64, 32, device="cuda") * 1000.0).to(torch.float16)
        opt = OCGOptV2([param], lr=1e-4, compile_step=False, foreach=True, stochastic_fp=True)
        opt.step()
        assert torch.isfinite(param).all()

    def test_fp16_param_runs_compiled(self):
        torch.manual_seed(42)
        param = torch.randn(64, 32, device="cuda", dtype=torch.float16, requires_grad=True)
        param.grad = (torch.randn(64, 32, device="cuda") * 1000.0).to(torch.float16)
        opt = OCGOptV2([param], lr=1e-4, compile_step=True, foreach=False, stochastic_fp=True)
        opt.step()
        assert torch.isfinite(param).all()

    def test_fp16_uses_plain_cast_not_stochastic(self):
        """fp16 writeback should use plain .copy_ (deterministic given the fp32
        source), not stochastic rounding. We verify by checking that two runs
        with the same seed produce identical fp16 results (stochastic rounding
        would introduce randomness)."""
        def run_once():
            torch.manual_seed(42)
            param = torch.randn(32, 16, device="cuda", dtype=torch.float16, requires_grad=True)
            param.grad = torch.randn(32, 16, device="cuda", dtype=torch.float16)
            opt = OCGOptV2([param], lr=1e-3, compile_step=False, foreach=False, stochastic_fp=True)
            opt.step()
            return param.detach().clone()

        a = run_once()
        b = run_once()
        assert torch.equal(a, b), "fp16 writeback should be deterministic (plain cast)"

    def test_low_precision_dtypes_set(self):
        """_LOW_PRECISION_DTYPES must include both bf16 and fp16."""
        assert torch.bfloat16 in _LOW_PRECISION_DTYPES
        assert torch.float16 in _LOW_PRECISION_DTYPES
        assert torch.float32 not in _LOW_PRECISION_DTYPES

    def test_fp16_state_stored_in_fp16(self):
        """State buffers should match the param dtype (fp16), not be upcast
        permanently — upcast is only for the compute working copy."""
        torch.manual_seed(0)
        param = torch.randn(16, 8, device="cuda", dtype=torch.float16, requires_grad=True)
        param.grad = torch.randn(16, 8, device="cuda", dtype=torch.float16)
        opt = OCGOptV2([param], lr=1e-3, compile_step=False, foreach=False, stochastic_fp=True)
        opt.step()
        state = opt.state[param]
        assert state["value_momentum"].dtype == torch.float16
        assert state["centralized_momentum"].dtype == torch.float16


# ---------------------------------------------------------------------------
# A1: reset() clears state and step counters
# ---------------------------------------------------------------------------


class TestReset:
    """Verify reset() actually clears optimizer state and step counters."""

    def test_reset_clears_state(self):
        torch.manual_seed(0)
        model = torch.nn.Linear(32, 16, device="cuda")
        opt = OCGOptV2(model.parameters(), lr=1e-3, compile_step=False, foreach=False)
        # Run a few steps to populate state
        for _ in range(3):
            x = torch.randn(4, 32, device="cuda")
            model(x).sum().backward()
            opt.step()
            opt.zero_grad()
        # State should be populated
        assert any(len(s) > 0 for s in opt.state.values()), "state should be populated"
        opt.reset()
        # All state cleared
        for s in opt.state.values():
            assert len(s) == 0, f"state not cleared: {s}"

    def test_reset_clears_step_counter(self):
        torch.manual_seed(0)
        model = torch.nn.Linear(32, 16, device="cuda")
        opt = OCGOptV2(model.parameters(), lr=1e-3, compile_step=False, foreach=False)
        for _ in range(5):
            x = torch.randn(4, 32, device="cuda")
            model(x).sum().backward()
            opt.step()
            opt.zero_grad()
        assert opt.param_groups[0]["step"] == 5
        opt.reset()
        assert opt.param_groups[0]["step"] == 0

    def test_reset_allows_clean_restart(self):
        """After reset, the next step should behave like step 1 (step counter
        goes 0 -> 1, slow_beta == 0)."""
        torch.manual_seed(0)
        model = torch.nn.Linear(32, 16, device="cuda")
        opt = OCGOptV2(model.parameters(), lr=1e-3, compile_step=False, foreach=False)
        for _ in range(4):
            x = torch.randn(4, 32, device="cuda")
            model(x).sum().backward()
            opt.step()
            opt.zero_grad()
        opt.reset()
        # One more step — should be step 1
        x = torch.randn(4, 32, device="cuda")
        model(x).sum().backward()
        opt.step()
        assert opt.param_groups[0]["step"] == 1
        # State should be freshly reinitialized with the correct shape/dtype.
        # (Momentum is NOT zero after step 1: slow_beta1 == 0 at step 1, so the
        # lerp weight is 1.0 and momentum is set to the full centralized_grad.)
        for p in model.parameters():
            if p.grad is not None and p in opt.state:
                vm = opt.state[p]["value_momentum"]
                assert vm.shape == p.shape, "momentum shape mismatch after reset"
                assert vm.dtype == p.dtype, "momentum dtype mismatch after reset"
                assert torch.isfinite(vm).all(), "momentum non-finite after reset"

    def test_reset_clears_scalar_and_srng_caches(self):
        torch.manual_seed(0)
        model = torch.nn.Linear(32, 16, device="cuda")
        opt = OCGOptV2(model.parameters(), lr=1e-3, compile_step=False, foreach=False)
        for _ in range(2):
            x = torch.randn(4, 32, device="cuda")
            model(x).sum().backward()
            opt.step()
            opt.zero_grad()
        # Touch the scalar-tensor cache via the compiled-path helper
        _ = opt._get_scalar_tensors(model.weight.device)
        assert len(opt._scalar_tensors) > 0
        opt.reset()
        assert opt._scalar_tensors == {}
        assert opt._srng_buf is None

    def test_reset_all_three_paths(self):
        """reset() must work regardless of which step path is active."""
        for compile_step, foreach in [(False, False), (False, True), (True, False)]:
            torch.manual_seed(0)
            model = torch.nn.Linear(32, 16, device="cuda")
            opt = OCGOptV2(
                model.parameters(), lr=1e-3,
                compile_step=compile_step, foreach=foreach,
            )
            x = torch.randn(4, 32, device="cuda")
            model(x).sum().backward()
            opt.step()
            opt.zero_grad()
            assert opt.param_groups[0]["step"] == 1
            opt.reset()
            assert opt.param_groups[0]["step"] == 0
            assert all(len(s) == 0 for s in opt.state.values())


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
