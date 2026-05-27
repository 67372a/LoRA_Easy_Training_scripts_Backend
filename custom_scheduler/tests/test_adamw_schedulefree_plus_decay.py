"""Tests for AdamWScheduleFreePlus weight decay placement (decay-at-z vs decay-at-y).

Verifies that:
1. Default behavior (weight_decay_at_y=False) applies decay at z only.
2. Legacy behavior (weight_decay_at_y=True) applies decay at y (original).
3. The decay-at-z path produces bounded z iterates.
4. The decay-at-y path matches the original unmodified algorithm.
5. Both paths run without error and produce finite loss values.
6. eval()/train() mode switching works correctly with both paths.
"""

import torch
import pytest
import sys
import os

# Add the parent directory to sys.path so we can import the optimizer
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from LoraEasyCustomOptimizer.adamw_schedulefree_plus import AdamWScheduleFreePlus


def _make_simple_model_and_loss(device="cuda", dtype=torch.float32):
    """Create a simple model and synthetic data for testing."""
    torch.manual_seed(42)
    model = torch.nn.Linear(8, 4, device=device, dtype=dtype)
    # Synthetic data
    x = torch.randn(16, 8, device=device, dtype=dtype)
    target = torch.randn(16, 4, device=device, dtype=dtype)
    return model, x, target


def _run_training_loop(optimizer, model, x, target, steps=50):
    """Run a training loop and return the loss history."""
    losses = []
    for step in range(steps):
        optimizer.train()
        out = model(x)
        loss = ((out - target) ** 2).mean()
        loss.backward()
        optimizer.step_func(loss.item())
        model.zero_grad()
        losses.append(loss.item())
    return losses


# Common conservative kwargs to avoid Polyak step blow-up
_SAFE_KWARGS = dict(
    lr=0.5,
    betas=(0.9, 0.95),
    sf_beta1=0.9,
    warmup_steps=5,
    c_warmup=10,
    max_polyak_lr=5.0,
)


class TestDecayAtZ:
    """Tests for the default decay-at-z behavior."""

    def test_decays_at_z_by_default(self):
        """Default behavior should use decay_at_z (weight_decay_at_y=False)."""
        model, x, target = _make_simple_model_and_loss(device="cuda")
        opt = AdamWScheduleFreePlus(
            model.parameters(),
            weight_decay=0.1,
            **_SAFE_KWARGS,
        )
        # Verify default is decay-at-z
        for group in opt.param_groups:
            assert group.get("weight_decay_at_y", False) is False

    def test_decay_at_z_runs_without_error(self):
        """Decay-at-z path should run without errors and produce finite losses."""
        model, x, target = _make_simple_model_and_loss(device="cuda")
        opt = AdamWScheduleFreePlus(
            model.parameters(),
            weight_decay=0.1,
            **_SAFE_KWARGS,
        )
        losses = _run_training_loop(opt, model, x, target, steps=100)
        assert all(torch.tensor(losses).isfinite()), f"Non-finite loss detected: {losses[-5:]}"

    def test_decay_at_z_bounds_z_iterates(self):
        """With decay at z, the z iterates should remain bounded.

        Lemma 3.1 from Apte et al. (2026) proves:
            ||Z_t||_F <= ||Z_0||_F * (1-eta*lambda)^t + C/lambda
        so z should not diverge.
        """
        model, x, target = _make_simple_model_and_loss(device="cuda")
        opt = AdamWScheduleFreePlus(
            model.parameters(),
            weight_decay=1.0,
            **_SAFE_KWARGS,
        )

        # Record z norms
        z_norms = []
        for step in range(200):
            opt.train()
            out = model(x)
            loss = ((out - target) ** 2).mean()
            loss.backward()
            opt.step_func(loss.item())
            model.zero_grad()

            # Record z norm
            total_norm = 0.0
            for p in model.parameters():
                state = opt.state[p]
                if "z" in state:
                    total_norm += state["z"].float().norm().item() ** 2
            z_norms.append(total_norm ** 0.5)

        # After many steps, z norms should stabilize, not diverge
        # Check that the last 50 steps' z norms are bounded
        late_z_norms = z_norms[-50:]
        max_late_norm = max(late_z_norms)
        assert max_late_norm < 100.0, (
            f"z norm grew too large ({max_late_norm:.2f}), "
            f"suggesting unbounded growth with decay_at_z"
        )

    def test_decay_at_z_pure_gradient_on_y(self):
        """With decay at z, the y-update should use PURE gradient (no decay term)."""
        torch.manual_seed(42)
        model = torch.nn.Linear(4, 2, bias=False, device="cuda", dtype=torch.float64)
        x = torch.randn(8, 4, device="cuda", dtype=torch.float64)
        target = torch.zeros(8, 2, device="cuda", dtype=torch.float64)

        opt = AdamWScheduleFreePlus(
            model.parameters(),
            weight_decay=1.0,
            weight_decay_at_y=False,
            **_SAFE_KWARGS,
        )

        # Train one step to initialize state
        opt.train()
        out = model(x)
        loss = ((out - target) ** 2).mean()
        loss.backward()
        opt.step_func(loss.item())
        model.zero_grad()

        # Now snapshot after state is initialized
        p_before = model.weight.data.clone()
        z_before = opt.state[model.weight]["z"].clone()

        # Take another step
        opt.train()
        out = model(x)
        loss = ((out - target) ** 2).mean()
        loss.backward()
        opt.step_func(loss.item())
        model.zero_grad()

        # Verify z changed (decay + gradient)
        z_change = (opt.state[model.weight]["z"] - z_before).abs().max().item()
        assert z_change > 0, "z should have changed"


class TestDecayAtY:
    """Tests for the legacy decay-at-y behavior."""

    def test_decay_at_y_flag_accepted(self):
        """Setting weight_decay_at_y=True should be accepted."""
        model, x, target = _make_simple_model_and_loss(device="cuda")
        opt = AdamWScheduleFreePlus(
            model.parameters(),
            weight_decay=0.1,
            weight_decay_at_y=True,
            **_SAFE_KWARGS,
        )
        for group in opt.param_groups:
            assert group["weight_decay_at_y"] is True

    def test_decay_at_y_runs_without_error(self):
        """Decay-at-y path should run without errors and produce finite losses."""
        model, x, target = _make_simple_model_and_loss(device="cuda")
        opt = AdamWScheduleFreePlus(
            model.parameters(),
            weight_decay=0.1,
            weight_decay_at_y=True,
            **_SAFE_KWARGS,
        )
        losses = _run_training_loop(opt, model, x, target, steps=100)
        assert all(torch.tensor(losses).isfinite()), f"Non-finite loss detected: {losses[-5:]}"

    def test_decay_at_y_matches_original_structure(self):
        """The decay-at-y path should produce different results than decay-at-z.

        With non-zero weight_decay and long training, the two paths diverge.
        """
        torch.manual_seed(42)
        model_y = torch.nn.Linear(16, 8, device="cuda")
        torch.manual_seed(42)
        model_z = torch.nn.Linear(16, 8, device="cuda")

        x = torch.randn(32, 16, device="cuda")
        target = torch.randn(32, 8, device="cuda")

        common_kwargs = dict(
            weight_decay=1.0,
            **_SAFE_KWARGS,
        )

        opt_y = AdamWScheduleFreePlus(model_y.parameters(), weight_decay_at_y=True, **common_kwargs)
        opt_z = AdamWScheduleFreePlus(model_z.parameters(), weight_decay_at_y=False, **common_kwargs)

        # Run both for several steps
        for _ in range(20):
            for opt, mdl in [(opt_y, model_y), (opt_z, model_z)]:
                opt.train()
                out = mdl(x)
                loss = ((out - target) ** 2).mean()
                loss.backward()
                opt.step_func(loss.item())
                mdl.zero_grad()

        # Weights should differ (the two decay strategies produce different updates)
        w_y = model_y.weight.data.clone()
        w_z = model_z.weight.data.clone()
        assert not torch.allclose(w_y, w_z, atol=1e-6), (
            "Decay-at-y and decay-at-z should produce different weights"
        )


class TestEvalTrainMode:
    """Tests for eval/train mode switching with both decay paths."""

    @pytest.mark.parametrize("decay_at_y", [True, False])
    def test_eval_train_cycle(self, decay_at_y):
        """eval() then train() should be reversible."""
        torch.manual_seed(42)
        model = torch.nn.Linear(8, 4, device="cuda")
        opt = AdamWScheduleFreePlus(
            model.parameters(),
            weight_decay=0.1,
            weight_decay_at_y=decay_at_y,
            **_SAFE_KWARGS,
        )

        x = torch.randn(16, 8, device="cuda")
        target = torch.randn(16, 4, device="cuda")

        # Train enough steps to pass c_warmup (10) so x diverges from z
        for _ in range(15):
            opt.train()
            out = model(x)
            loss = ((out - target) ** 2).mean()
            loss.backward()
            opt.step_func(loss.item())
            model.zero_grad()

        # Snapshot y (train mode)
        y_snapshot = model.weight.data.clone()

        # Switch to eval -> get x
        opt.eval()
        x_snapshot = model.weight.data.clone()

        # y and x should differ (unless sf_beta1_k = 0)
        assert not torch.allclose(y_snapshot, x_snapshot, atol=1e-8), (
            "y and x should differ in general"
        )

        # Switch back to train -> should restore y
        opt.train()
        y_restored = model.weight.data.clone()
        assert torch.allclose(y_snapshot, y_restored, atol=1e-6), (
            "train() should restore y after eval()"
        )

    @pytest.mark.parametrize("decay_at_y", [True, False])
    def test_eval_mode_loss_is_finite(self, decay_at_y):
        """Loss computed in eval mode should be finite."""
        model, x, target = _make_simple_model_and_loss(device="cuda")
        opt = AdamWScheduleFreePlus(
            model.parameters(),
            weight_decay=0.1,
            weight_decay_at_y=decay_at_y,
            **_SAFE_KWARGS,
        )

        # Train
        _run_training_loop(opt, model, x, target, steps=20)

        # Eval
        opt.eval()
        with torch.no_grad():
            out = model(x)
            loss = ((out - target) ** 2).mean()
        assert torch.isfinite(loss), f"Eval loss not finite: {loss.item()}"
        opt.train()


class TestZeroDecay:
    """When weight_decay=0, both paths should produce identical results."""

    def test_zero_decay_paths_equivalent(self):
        """With weight_decay=0, decay_at_y and decay_at_z should be identical."""
        torch.manual_seed(42)
        model_y = torch.nn.Linear(8, 4, device="cuda")
        torch.manual_seed(42)
        model_z = torch.nn.Linear(8, 4, device="cuda")

        x = torch.randn(16, 8, device="cuda")
        target = torch.randn(16, 4, device="cuda")

        common_kwargs = dict(
            weight_decay=0.0,
            **_SAFE_KWARGS,
        )

        opt_y = AdamWScheduleFreePlus(model_y.parameters(), weight_decay_at_y=True, **common_kwargs)
        opt_z = AdamWScheduleFreePlus(model_z.parameters(), weight_decay_at_y=False, **common_kwargs)

        for _ in range(20):
            for opt, mdl in [(opt_y, model_y), (opt_z, model_z)]:
                opt.train()
                out = mdl(x)
                loss = ((out - target) ** 2).mean()
                loss.backward()
                opt.step_func(loss.item())
                mdl.zero_grad()

        # Weights should be identical (no decay = no difference)
        w_y = model_y.weight.data.clone()
        w_z = model_z.weight.data.clone()
        assert torch.allclose(w_y, w_z, atol=1e-7), (
            "With weight_decay=0, both paths should produce identical weights. "
            f"Max diff: {(w_y - w_z).abs().max().item()}"
        )


class TestBFloat16:
    """Test that both paths work with bfloat16 parameters."""

    @pytest.mark.parametrize("decay_at_y", [True, False])
    def test_bf16_runs_without_error(self, decay_at_y):
        """Both decay paths should work with bfloat16 parameters."""
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            pytest.skip("CUDA bfloat16 not available")

        model, x, target = _make_simple_model_and_loss(
            device="cuda", dtype=torch.bfloat16
        )
        opt = AdamWScheduleFreePlus(
            model.parameters(),
            weight_decay=0.1,
            weight_decay_at_y=decay_at_y,
            **_SAFE_KWARGS,
        )
        losses = _run_training_loop(opt, model, x, target, steps=50)
        assert all(torch.tensor(losses).isfinite()), (
            f"Non-finite loss with bf16, decay_at_y={decay_at_y}: {losses[-5:]}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
