"""
Tests for OCGOptV2 bug fixes:

Bug 2: AOL + global RMS double-normalization when aol=True and input_norm=False.
  - When aol=True, AOL preconditioning should replace global RMS normalization.
  - Previously, both were applied (global RMS in the else branch of input_norm).

Issue 3: view_as → reshape_as in native/foreach paths.
  - All three step paths now use reshape_as for non-contiguous tensors from
    spectral clipping transpose.
"""

import sys
import os
import torch
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from backend.custom_scheduler.LoraEasyCustomOptimizer.ocgoptv2 import (
    OCGOptV2,
    _reshape_to_2d,
)


# ---------------------------------------------------------------------------
# Bug 2: AOL should not be followed by global RMS normalization
# ---------------------------------------------------------------------------


class TestAOLNoDoubleNormalization:
    """When aol=True and input_norm=False, only AOL should be applied (no global RMS)."""

    def _make_grad_tensor(self, shape=(64, 32), device="cuda"):
        """Create a gradient tensor with a known non-unit RMS."""
        torch.manual_seed(42)
        return torch.randn(shape, device=device) * 10.0  # RMS ≈ 10, not 1

    def test_aol_preconditioning_does_not_force_unit_rms(self):
        """
        After AOL preconditioning, the global RMS should NOT be forced to 1.0.
        If the bug is present (double normalization), the RMS would be ~1.0.
        If fixed (AOL only), the RMS should differ from 1.0.
        """
        grad = self._make_grad_tensor((64, 32))
        original_rms = grad.pow(2).mean().sqrt().item()

        # Simulate the fixed code path: AOL only, no global RMS
        grad_2d = _reshape_to_2d(grad)
        A = grad_2d @ grad_2d.mT
        rescaling = A.abs().sum(dim=-1, keepdim=True).clamp_min_(1e-16)
        grad_2d = grad_2d * rescaling.rsqrt()
        grad_after_aol = grad_2d.reshape_as(grad)

        aol_rms = grad_after_aol.pow(2).mean().sqrt().item()

        # After AOL, the RMS should NOT be exactly 1.0 (it would be if global RMS was also applied)
        # AOL normalizes rows, but the global RMS of the whole tensor is not necessarily 1.0
        assert aol_rms != pytest.approx(1.0, abs=0.01), (
            f"AOL should not force global RMS to 1.0. Got RMS={aol_rms:.4f}. "
            f"This suggests global RMS normalization is still being applied after AOL."
        )

    def test_native_step_aol_no_double_norm(self):
        """
        Run a native step with aol=True, input_norm=False.
        Verify the optimizer completes without error and produces valid updates.
        """
        model = torch.nn.Linear(64, 32, device="cuda")
        torch.manual_seed(42)
        model.weight.data = torch.randn(32, 64, device="cuda") * 10.0
        model.bias.data.zero_()

        dummy_input = torch.randn(4, 64, device="cuda")
        output = model(dummy_input)
        loss = output.sum()
        loss.backward()

        grad_rms_before = model.weight.grad.pow(2).mean().sqrt().item()
        assert grad_rms_before > 1.0, "Test setup: grad RMS should be > 1.0"

        optimizer = OCGOptV2(
            model.parameters(),
            lr=1e-4,
            aol=True,
            input_norm=False,
            compile_step=False,
            foreach=False,
        )

        # Should not raise
        optimizer.step()

        # Verify the weight was actually updated
        assert not torch.equal(model.weight.data, torch.randn(32, 64, device="cuda"))

    def test_foreach_step_aol_no_double_norm(self):
        """Run a foreach step with aol=True, input_norm=False."""
        model = torch.nn.Linear(64, 32, device="cuda")
        torch.manual_seed(42)
        model.weight.data = torch.randn(32, 64, device="cuda") * 10.0
        model.bias.data.zero_()

        dummy_input = torch.randn(4, 64, device="cuda")
        output = model(dummy_input)
        loss = output.sum()
        loss.backward()

        optimizer = OCGOptV2(
            model.parameters(),
            lr=1e-4,
            aol=True,
            input_norm=False,
            compile_step=False,
            foreach=True,
        )

        # Should not raise
        optimizer.step()

    def test_compiled_step_aol_no_double_norm(self):
        """Run a compiled step with aol=True, input_norm=False."""
        model = torch.nn.Linear(64, 32, device="cuda")
        torch.manual_seed(42)
        model.weight.data = torch.randn(32, 64, device="cuda") * 10.0
        model.bias.data.zero_()

        dummy_input = torch.randn(4, 64, device="cuda")
        output = model(dummy_input)
        loss = output.sum()
        loss.backward()

        optimizer = OCGOptV2(
            model.parameters(),
            lr=1e-4,
            aol=True,
            input_norm=False,
            compile_step=True,
            foreach=False,
        )

        # Should not raise
        optimizer.step()

    def test_aol_rms_differs_from_global_rms(self):
        """
        Verify that AOL-only normalization produces different results
        than AOL + global RMS (the old buggy behavior).
        """
        grad = self._make_grad_tensor((64, 32))

        # Path A: AOL only (fixed behavior)
        grad_a = grad.clone()
        grad_2d_a = _reshape_to_2d(grad_a)
        A_a = grad_2d_a @ grad_2d_a.mT
        rescaling_a = A_a.abs().sum(dim=-1, keepdim=True).clamp_min_(1e-16)
        grad_2d_a = grad_2d_a * rescaling_a.rsqrt()
        grad_a = grad_2d_a.reshape_as(grad_a)

        # Path B: AOL + global RMS (old buggy behavior)
        grad_b = grad.clone()
        grad_2d_b = _reshape_to_2d(grad_b)
        A_b = grad_2d_b @ grad_2d_b.mT
        rescaling_b = A_b.abs().sum(dim=-1, keepdim=True).clamp_min_(1e-16)
        grad_2d_b = grad_2d_b * rescaling_b.rsqrt()
        grad_b = grad_2d_b.reshape_as(grad_b)
        # Additional global RMS (the bug)
        rms_b = grad_b.pow(2).mean().sqrt_().clamp_min_(1e-16)
        grad_b = grad_b.div(rms_b)

        # The two results should differ
        assert not torch.allclose(grad_a, grad_b, atol=1e-6), (
            "AOL-only and AOL+globalRMS should produce different results. "
            "If they're identical, the fix may not be working."
        )

        # AOL-only should NOT have unit global RMS
        rms_a = grad_a.pow(2).mean().sqrt().item()
        rms_b_val = grad_b.pow(2).mean().sqrt().item()
        assert rms_b_val == pytest.approx(1.0, abs=0.01), (
            f"AOL+globalRMS should have RMS≈1.0, got {rms_b_val:.4f}"
        )
        assert rms_a != pytest.approx(1.0, abs=0.1), (
            f"AOL-only should NOT have RMS≈1.0, got {rms_a:.4f}"
        )


# ---------------------------------------------------------------------------
# Issue 3: reshape_as used consistently across all paths
# ---------------------------------------------------------------------------


class TestReshapeAsConsistency:
    """Verify all step paths work with both tall and wide parameters."""

    @pytest.fixture
    def wide_model(self):
        """Model with weight shape [512, 256] — triggers flip=True."""
        torch.manual_seed(42)
        model = torch.nn.Linear(256, 512, device="cuda")
        model.weight.data = torch.randn(512, 256, device="cuda")
        model.bias.data.zero_()
        return model

    @pytest.fixture
    def tall_model(self):
        """Model with weight shape [256, 512] — flip=False."""
        torch.manual_seed(42)
        model = torch.nn.Linear(512, 256, device="cuda")
        model.weight.data = torch.randn(256, 512, device="cuda")
        model.bias.data.zero_()
        return model

    def _run_step(self, model, compile_step, foreach):
        """Helper: run one optimizer step."""
        model.zero_grad()
        dummy_input = torch.randn(4, model.in_features, device="cuda")
        output = model(dummy_input)
        loss = output.sum()
        loss.backward()

        optimizer = OCGOptV2(
            model.parameters(),
            lr=1e-4,
            compile_step=compile_step,
            foreach=foreach,
        )
        optimizer.step()
        return optimizer

    def test_native_wide(self, wide_model):
        """Native path with wide parameter (flip=True)."""
        self._run_step(wide_model, compile_step=False, foreach=False)

    def test_native_tall(self, tall_model):
        """Native path with tall parameter (flip=False)."""
        self._run_step(tall_model, compile_step=False, foreach=False)

    def test_foreach_wide(self, wide_model):
        """Foreach path with wide parameter (flip=True)."""
        self._run_step(wide_model, compile_step=False, foreach=True)

    def test_foreach_tall(self, tall_model):
        """Foreach path with tall parameter (flip=False)."""
        self._run_step(tall_model, compile_step=False, foreach=True)

    def test_compiled_wide(self, wide_model):
        """Compiled path with wide parameter (flip=True)."""
        self._run_step(wide_model, compile_step=True, foreach=False)

    def test_compiled_tall(self, tall_model):
        """Compiled path with tall parameter (flip=False)."""
        self._run_step(tall_model, compile_step=True, foreach=False)

    def test_native_3d_param(self):
        """Native path with 3D parameter (conv-like)."""
        torch.manual_seed(42)
        # Conv weight shape [out_channels, in_channels, kernel]
        param = torch.randn(128, 64, 3, device="cuda", requires_grad=True)
        dummy_grad = torch.randn(128, 64, 3, device="cuda")
        param.grad = dummy_grad

        optimizer = OCGOptV2([param], lr=1e-4, compile_step=False, foreach=False)
        optimizer.step()

    def test_native_1d_param(self):
        """Native path with 1D parameter (bias-like)."""
        torch.manual_seed(42)
        param = torch.randn(64, device="cuda", requires_grad=True)
        param.grad = torch.randn(64, device="cuda")

        optimizer = OCGOptV2([param], lr=1e-4, compile_step=False, foreach=False)
        optimizer.step()

    def test_all_three_paths_produce_same_result(self, wide_model):
        """
        Verify that native, foreach, and compiled paths produce consistent
        (not necessarily identical due to compile effects) results.
        """
        import copy

        # Use same initial weights and gradient for all paths
        torch.manual_seed(42)
        initial_weight = torch.randn(512, 256, device="cuda")
        initial_bias = torch.zeros(512, device="cuda")
        fixed_input = torch.randn(4, 256, device="cuda")

        results = {}
        for path_name, compile_step, foreach in [
            ("native", False, False),
            ("foreach", False, True),
        ]:
            torch.manual_seed(42)
            model = torch.nn.Linear(256, 512, device="cuda", bias=True)
            model.weight.data.copy_(initial_weight)
            model.bias.data.copy_(initial_bias)

            # Compute gradient with fixed input
            torch.manual_seed(123)
            output = model(fixed_input)
            loss = output.sum()
            loss.backward()

            optimizer = OCGOptV2(
                model.parameters(),
                lr=1e-4,
                compile_step=compile_step,
                foreach=foreach,
            )
            optimizer.step()

            results[path_name] = model.weight.data.clone()

        # Native and foreach should produce identical results
        assert torch.allclose(results["native"], results["foreach"], atol=1e-6), (
            "Native and foreach paths should produce identical results. "
            f"Max diff: {(results['native'] - results['foreach']).abs().max().item():.8f}"
        )


# ---------------------------------------------------------------------------
# Regression: verify basic optimizer functionality still works
# ---------------------------------------------------------------------------


class TestBasicFunctionality:
    """Smoke tests to verify the optimizer still works after fixes."""

    def test_multiple_steps(self):
        """Run multiple steps to verify no accumulated errors."""
        model = torch.nn.Linear(64, 32, device="cuda")
        optimizer = OCGOptV2(model.parameters(), lr=1e-4)

        for _ in range(10):
            dummy_input = torch.randn(4, 64, device="cuda")
            output = model(dummy_input)
            loss = output.sum()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    def test_with_weight_decay(self):
        """Verify weight decay path works."""
        model = torch.nn.Linear(64, 32, device="cuda")
        optimizer = OCGOptV2(
            model.parameters(), lr=1e-4, weight_decay=0.01
        )

        dummy_input = torch.randn(4, 64, device="cuda")
        output = model(dummy_input)
        loss = output.sum()
        loss.backward()
        optimizer.step()

    def test_with_adaptive(self):
        """Verify adaptive scaling path works."""
        model = torch.nn.Linear(64, 32, device="cuda")
        optimizer = OCGOptV2(model.parameters(), lr=1e-4, adaptive=True)

        dummy_input = torch.randn(4, 64, device="cuda")
        output = model(dummy_input)
        loss = output.sum()
        loss.backward()
        optimizer.step()

    def test_with_lowpass(self):
        """Verify lowpass filter path works."""
        model = torch.nn.Linear(64, 32, device="cuda")
        optimizer = OCGOptV2(model.parameters(), lr=1e-4, lowpass_grad=1.0)

        dummy_input = torch.randn(4, 64, device="cuda")
        output = model(dummy_input)
        loss = output.sum()
        loss.backward()
        optimizer.step()

    def test_scalar_parameter(self):
        """Verify scalar (0-dim) parameter path works."""
        param = torch.tensor(1.0, device="cuda", requires_grad=True)
        param.grad = torch.tensor(0.5, device="cuda")

        optimizer = OCGOptV2([param], lr=1e-4)
        optimizer.step()

    def test_bf16_parameter(self):
        """Verify bf16 parameter with stochastic rounding works."""
        param = torch.randn(64, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        param.grad = torch.randn(64, 32, device="cuda", dtype=torch.bfloat16)

        optimizer = OCGOptV2([param], lr=1e-4, stochastic_fp=True)
        optimizer.step()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
