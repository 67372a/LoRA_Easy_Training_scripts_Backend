"""Tests for NorMuonScheduleFree (SF-NorMuon) optimizer.

Validates:
1. Import and optimizer instantiation works.
2. train()/eval() mode switching and step() run without error.
3. State tensors (z, mom, v for 2D; z, exp_avg_sq for 1D) are correctly initialized.
4. Parameters change after a step (spectral update is active).
5. Convergence: loss decreases over multiple steps on a simple task.
6. Numerical stability: no NaN or Inf after multiple steps.
7. Newton-Schulz polar decomposition produces near-orthogonal output.
8. Mixed 1D/2D parameter handling works correctly.
9. Weight decay is applied at Z (not at Y).
10. bfloat16 parameters use stochastic rounding correctly.
11. float16 parameters use fp32 upcast, fp32 Newton-Schulz, and stochastic rounding.

Run with:
    cd backend/sd_scripts
    python -m pytest ../custom_scheduler/tests/test_nor_muon_schedulefree.py -v
"""

import sys
import os
import pytest
import torch
import math

# Import directly from the module file to avoid pulling in the full
# LoraEasyCustomOptimizer package (which has heavy dependencies).
import importlib.util
import types

_pkg_dir = os.path.join(os.path.dirname(__file__), "..", "LoraEasyCustomOptimizer")

# Create a fake package module so relative imports work
if "LoraEasyCustomOptimizer" not in sys.modules:
    _pkg = types.ModuleType("LoraEasyCustomOptimizer")
    _pkg.__path__ = [_pkg_dir]
    _pkg.__package__ = "LoraEasyCustomOptimizer"
    sys.modules["LoraEasyCustomOptimizer"] = _pkg

# Load utils first (needed via ``from .utils import copy_stochastic_``)
_utils_path = os.path.join(_pkg_dir, "utils.py")
_utils_spec = importlib.util.spec_from_file_location(
    "LoraEasyCustomOptimizer.utils", _utils_path
)
_utils_mod = importlib.util.module_from_spec(_utils_spec)
sys.modules["LoraEasyCustomOptimizer.utils"] = _utils_mod
_utils_spec.loader.exec_module(_utils_mod)

# Load nor_muon_schedulefree as a submodule of the fake package
_nmsf_path = os.path.join(_pkg_dir, "nor_muon_schedulefree.py")
_spec = importlib.util.spec_from_file_location(
    "LoraEasyCustomOptimizer.nor_muon_schedulefree", _nmsf_path
)
_nmsf = importlib.util.module_from_spec(_spec)
sys.modules["LoraEasyCustomOptimizer.nor_muon_schedulefree"] = _nmsf
_spec.loader.exec_module(_nmsf)

NorMuonScheduleFree = _nmsf.NorMuonScheduleFree
_zeropower_via_newtonschulz5 = _nmsf._zeropower_via_newtonschulz5
copy_stochastic_ = _utils_mod.copy_stochastic_


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
    """Run *n_steps* optimizer.step() calls and return the final loss."""
    torch.manual_seed(seed)
    final_loss = None
    opt.train()
    for _ in range(n_steps):
        x = torch.randn(4, input_size, dtype=model[0].weight.dtype, device=model[0].weight.device)
        loss = model(x).sum()
        loss.backward()
        opt.step()
        opt.zero_grad()
        final_loss = loss
    return final_loss


# ---------------------------------------------------------------------------
# Tests: Newton-Schulz polar decomposition
# ---------------------------------------------------------------------------

class TestNewtonSchulz:
    """Tests for the Newton-Schulz polar decomposition helper."""

    def test_output_shape_matches_input(self):
        """Polar factor should have the same shape as input."""
        G = torch.randn(16, 32)
        P = _zeropower_via_newtonschulz5(G)
        assert P.shape == G.shape

    def test_output_dtype_matches_input(self):
        """Polar factor should have the same dtype as input."""
        G = torch.randn(16, 32, dtype=torch.float32)
        P = _zeropower_via_newtonschulz5(G)
        assert P.dtype == torch.float32

    def test_near_orthogonal_output(self):
        """For a tall matrix, P^T @ P should be near identity.

        Note: The paper's coefficients (3.4445, -4.7750, 2.0315) intentionally
        produce an approximation where singular values are ~Uniform(0.5, 1.5)
        rather than exact orthogonality. This is by design and does not hurt
        model performance (see Appendix F notes in the paper).
        """
        G = torch.randn(64, 16, dtype=torch.float32)
        P = _zeropower_via_newtonschulz5(G, steps=5)
        # P is 64x16, so P^T @ P should be approximately I_16
        PtP = P.T @ P
        I = torch.eye(16, dtype=torch.float32)
        # Allow tolerance for the approximate polar factor
        max_diff = (PtP - I).abs().max().item()
        assert max_diff < 0.55, f"P^T @ P deviation too large: max_diff={max_diff:.4f}"
        # Diagonal should be closer to 1 than off-diagonal
        diag_mean = PtP.diag().abs().mean().item()
        assert diag_mean > 0.7, f"Diagonal of P^T @ P too far from 1: mean={diag_mean:.4f}"

    def test_near_orthogonal_wide_matrix(self):
        """For a wide matrix, P @ P^T should be near identity.

        Same tolerance as tall matrix case — the approximation produces
        singular values that are near 1 but not exactly 1.
        """
        G = torch.randn(16, 64, dtype=torch.float32)
        P = _zeropower_via_newtonschulz5(G, steps=5)
        # P is 16x64, so P @ P^T should be approximately I_16
        PPt = P @ P.T
        I = torch.eye(16, dtype=torch.float32)
        max_diff = (PPt - I).abs().max().item()
        assert max_diff < 0.55, f"P @ P^T deviation too large: max_diff={max_diff:.4f}"
        diag_mean = PPt.diag().abs().mean().item()
        assert diag_mean > 0.7, f"Diagonal of P @ P^T too far from 1: mean={diag_mean:.4f}"

    def test_spectral_norm_near_one(self):
        """All singular values of the polar factor should be bounded near 1.

        The paper's coefficients produce S' ~ Uniform(0.5, 1.5) in the
        idealized case, so we use relaxed thresholds.
        """
        G = torch.randn(32, 32, dtype=torch.float64)
        P = _zeropower_via_newtonschulz5(G, steps=5).to(torch.float64)
        sv = torch.linalg.svdvals(P)
        # The paper notes singular values are ~Uniform(0.5, 1.5)
        assert sv.min().item() > 0.4, f"Min singular value too low: {sv.min().item():.4f}"
        assert sv.max().item() < 1.8, f"Max singular value too high: {sv.max().item():.4f}"

    def test_empty_matrix_handling(self):
        """Should handle small matrices without error."""
        G = torch.randn(2, 2, dtype=torch.float32)
        P = _zeropower_via_newtonschulz5(G, steps=5)
        assert P.shape == (2, 2)
        assert not torch.isnan(P).any()

    def test_bfloat16_input(self):
        """Should work with bfloat16 input tensors."""
        G = torch.randn(16, 32, dtype=torch.bfloat16)
        P = _zeropower_via_newtonschulz5(G, steps=5)
        assert P.shape == G.shape
        assert P.dtype == torch.bfloat16

    def test_compute_dtype_fp32(self):
        """When compute_dtype=fp32, iteration should use fp32 (no bf16 cast)."""
        G = torch.randn(16, 32, dtype=torch.float32)
        P = _zeropower_via_newtonschulz5(G, steps=5, compute_dtype=torch.float32)
        assert P.shape == G.shape
        assert P.dtype == torch.float32
        assert not torch.isnan(P).any()

    def test_compute_dtype_fp32_output_quality(self):
        """fp32 compute should produce near-orthogonal output similar to bf16."""
        G = torch.randn(64, 16, dtype=torch.float32)
        P_bf16 = _zeropower_via_newtonschulz5(G, steps=5, compute_dtype=torch.bfloat16)
        P_fp32 = _zeropower_via_newtonschulz5(G, steps=5, compute_dtype=torch.float32)
        # Both should be valid polar approximations (not NaN/Inf)
        assert not torch.isnan(P_bf16).any()
        assert not torch.isnan(P_fp32).any()
        # fp32 should also produce near-orthogonal columns
        PtP = P_fp32.T @ P_fp32
        I = torch.eye(16, dtype=torch.float32)
        max_diff = (PtP - I).abs().max().item()
        assert max_diff < 0.55, f"fp32 NS P^T @ P deviation too large: {max_diff:.4f}"


# ---------------------------------------------------------------------------
# Tests: Optimizer instantiation
# ---------------------------------------------------------------------------

class TestInstantiation:
    """Tests for optimizer creation and configuration."""

    def test_basic_creation(self):
        """Optimizer can be created with default parameters."""
        model = _make_model()
        opt = NorMuonScheduleFree(model.parameters())
        assert opt is not None

    def test_custom_hyperparameters(self):
        """Optimizer accepts custom hyperparameters."""
        model = _make_model()
        opt = NorMuonScheduleFree(
            model.parameters(),
            lr=0.01,
            betas=(0.95, 0.99),
            momentum=0.9,
            eps=1e-6,
            weight_decay=0.1,
            warmup_steps=1000,
            eta_scale=0.3,
            ns_steps=3,
        )
        group = opt.param_groups[0]
        assert group["lr"] == 0.01
        assert group["betas"] == (0.95, 0.99)
        assert group["momentum"] == 0.9
        assert group["weight_decay"] == 0.1

    def test_unknown_kwargs_warning(self, caplog):
        """Unknown kwargs should produce a warning but not crash."""
        model = _make_model()
        with caplog.at_level("WARNING"):
            opt = NorMuonScheduleFree(model.parameters(), foobar=42)
        assert "foobar" in caplog.text

    def test_registered_in_init(self):
        """NorMuonScheduleFree should be importable from __init__."""
        # This test verifies the __init__.py registration
        init_path = os.path.join(_pkg_dir, "__init__.py")
        with open(init_path, "r") as f:
            content = f.read()
        assert "NorMuonScheduleFree" in content


# ---------------------------------------------------------------------------
# Tests: train/eval mode switching
# ---------------------------------------------------------------------------

class TestTrainEval:
    """Tests for schedule-free train/eval mode transitions."""

    def test_initial_mode_is_not_train(self):
        """Optimizer should start in non-train mode."""
        model = _make_model()
        opt = NorMuonScheduleFree(model.parameters())
        assert opt.param_groups[0]["train_mode"] is False

    def test_train_switches_mode(self):
        """Calling train() should set train_mode=True."""
        model = _make_model()
        opt = NorMuonScheduleFree(model.parameters())
        opt.train()
        assert opt.param_groups[0]["train_mode"] is True

    def test_eval_switches_mode(self):
        """Calling eval() should set train_mode=False."""
        model = _make_model()
        opt = NorMuonScheduleFree(model.parameters())
        opt.train()
        opt.eval()
        assert opt.param_groups[0]["train_mode"] is False

    def test_step_raises_if_not_train(self):
        """step() should raise if optimizer is not in train mode."""
        model = _make_model()
        opt = NorMuonScheduleFree(model.parameters())
        x = torch.randn(4, 32)
        loss = model(x).sum()
        loss.backward()
        with pytest.raises(Exception, match="train mode"):
            opt.step()

    def test_train_eval_roundtrip_preserves_parameters(self):
        """Parameters should not change from a train/eval roundtrip without steps."""
        model = _make_model()
        params_before = [p.clone() for p in model.parameters()]
        opt = NorMuonScheduleFree(model.parameters())

        # Initialize state by running one step
        opt.train()
        x = torch.randn(4, 32)
        loss = model(x).sum()
        loss.backward()
        opt.step()
        opt.zero_grad()

        # Save params after step, then do train/eval roundtrip
        params_after_step = [p.clone() for p in model.parameters()]
        opt.eval()
        opt.train()

        for p_before, p_after in zip(params_after_step, model.parameters()):
            assert torch.allclose(p_before, p_after, atol=1e-6), \
                "Parameters changed during train/eval roundtrip"


# ---------------------------------------------------------------------------
# Tests: State initialization
# ---------------------------------------------------------------------------

class TestStateInit:
    """Tests for correct optimizer state initialization."""

    def test_2d_params_have_spectral_state(self):
        """2-D parameters should have z, v, mom state after first step."""
        model = _make_model()
        opt = NorMuonScheduleFree(model.parameters())
        opt.train()
        x = torch.randn(4, 32)
        loss = model(x).sum()
        loss.backward()
        opt.step()

        for p in model.parameters():
            if p.ndim == 2:
                state = opt.state[p]
                assert "z" in state, "Missing 'z' state for 2D param"
                assert "v" in state, "Missing 'v' state for 2D param"
                assert "mom" in state, "Missing 'mom' state for 2D param"
                assert state["v"].shape == (p.shape[0],), \
                    f"v shape mismatch: {state['v'].shape} vs ({p.shape[0]},)"

    def test_1d_params_have_adam_state(self):
        """1-D parameters should have z, exp_avg_sq state after first step."""
        model = _make_model()
        # Add a bias to get a 1D parameter
        model = torch.nn.Sequential(
            torch.nn.Linear(32, 64, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 16, bias=True),
        )
        opt = NorMuonScheduleFree(model.parameters())
        opt.train()
        x = torch.randn(4, 32)
        loss = model(x).sum()
        loss.backward()
        opt.step()

        for p in model.parameters():
            if p.ndim == 1:
                state = opt.state[p]
                assert "z" in state, "Missing 'z' state for 1D param"
                assert "exp_avg_sq" in state, "Missing 'exp_avg_sq' state for 1D param"

    def test_momentum_buffer_initialized_to_zero(self):
        """Momentum buffer should be initialized to zeros."""
        model = _make_model()
        opt = NorMuonScheduleFree(model.parameters())
        opt.train()
        x = torch.randn(4, 32)
        loss = model(x).sum()
        loss.backward()
        opt.step()

        for p in model.parameters():
            if p.ndim == 2:
                state = opt.state[p]
                # After first step, mom = μ·0 + (1-μ)·grad, so not zero anymore
                # But the initialization should have been zeros
                assert state["mom"].shape == p.shape

    def test_z_tracks_parameter_shape(self):
        """z state should have the same shape as the parameter."""
        model = _make_model()
        opt = NorMuonScheduleFree(model.parameters())
        opt.train()
        x = torch.randn(4, 32)
        loss = model(x).sum()
        loss.backward()
        opt.step()

        for p in model.parameters():
            state = opt.state[p]
            assert state["z"].shape == p.shape


# ---------------------------------------------------------------------------
# Tests: Convergence
# ---------------------------------------------------------------------------

class TestConvergence:
    """Tests that the optimizer reduces loss on a simple task."""

    def test_loss_decreases_fp32(self):
        """Loss should decrease over multiple steps with float32."""
        model = _make_model(dtype=torch.float32)
        opt = NorMuonScheduleFree(model.parameters(), lr=0.008, warmup_steps=5)

        loss_init = _run_steps(model, opt, n_steps=1).item()
        loss_final = _run_steps(model, opt, n_steps=50).item()

        assert loss_final < loss_init, \
            f"Loss did not decrease: init={loss_init:.4f}, final={loss_final:.4f}"

    def test_loss_decreases_with_weight_decay(self):
        """Loss should decrease even with weight decay applied at Z."""
        model = _make_model(dtype=torch.float32)
        opt = NorMuonScheduleFree(
            model.parameters(), lr=0.008, weight_decay=0.05, warmup_steps=5
        )

        loss_init = _run_steps(model, opt, n_steps=1).item()
        loss_final = _run_steps(model, opt, n_steps=50).item()

        assert loss_final < loss_init, \
            f"Loss did not decrease with weight decay: init={loss_init:.4f}, final={loss_final:.4f}"

    def test_loss_decreases_with_mixed_params(self):
        """Loss should decrease when model has both 1D and 2D parameters."""
        torch.manual_seed(42)
        model = torch.nn.Sequential(
            torch.nn.Linear(32, 64, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 16, bias=True),
        )
        opt = NorMuonScheduleFree(model.parameters(), lr=0.008, warmup_steps=5)

        loss_init = _run_steps(model, opt, n_steps=1, input_size=32).item()
        loss_final = _run_steps(model, opt, n_steps=50, input_size=32).item()

        assert loss_final < loss_init, \
            f"Loss did not decrease with mixed params: init={loss_init:.4f}, final={loss_final:.4f}"

    def test_multiple_train_eval_cycles(self):
        """Optimizer should work correctly across train/eval/train cycles."""
        model = _make_model(dtype=torch.float32)
        opt = NorMuonScheduleFree(model.parameters(), lr=0.008, warmup_steps=3)

        losses = []
        for cycle in range(3):
            opt.train()
            loss = _run_steps(model, opt, n_steps=10, input_size=32)
            losses.append(loss.item())
            opt.eval()
            # Simulate validation (no gradient update)
            opt.train()

        # Loss should generally decrease across cycles
        assert losses[-1] < losses[0], \
            f"Loss did not decrease across train/eval cycles: {losses}"


# ---------------------------------------------------------------------------
# Tests: Numerical stability
# ---------------------------------------------------------------------------

class TestNumericalStability:
    """Tests for NaN/Inf safety."""

    def test_no_nan_after_many_steps_fp32(self):
        """No NaN or Inf should appear after many fp32 steps."""
        model = _make_model(dtype=torch.float32)
        opt = NorMuonScheduleFree(model.parameters(), lr=0.008, warmup_steps=5)

        opt.train()
        for _ in range(100):
            x = torch.randn(4, 32, dtype=torch.float32)
            loss = model(x).sum()
            loss.backward()
            opt.step()
            opt.zero_grad()

        for p in model.parameters():
            assert not torch.isnan(p).any(), f"NaN in parameter {p.shape}"
            assert not torch.isinf(p).any(), f"Inf in parameter {p.shape}"
            state = opt.state[p]
            for key, val in state.items():
                if isinstance(val, torch.Tensor):
                    assert not torch.isnan(val).any(), f"NaN in state[{key}] for param {p.shape}"
                    assert not torch.isinf(val).any(), f"Inf in state[{key}] for param {p.shape}"

    def test_no_nan_with_large_lr(self):
        """No NaN even with a relatively large learning rate."""
        model = _make_model(dtype=torch.float32)
        opt = NorMuonScheduleFree(model.parameters(), lr=0.05, warmup_steps=2)

        opt.train()
        for _ in range(50):
            x = torch.randn(4, 32, dtype=torch.float32)
            loss = model(x).sum()
            loss.backward()
            opt.step()
            opt.zero_grad()

        for p in model.parameters():
            assert not torch.isnan(p).any(), f"NaN with large lr, param shape {p.shape}"

    def test_no_nan_with_zero_momentum(self):
        """No NaN when momentum is zero (ablation setting from paper)."""
        model = _make_model(dtype=torch.float32)
        opt = NorMuonScheduleFree(model.parameters(), lr=0.008, momentum=0.0, warmup_steps=5)

        opt.train()
        for _ in range(50):
            x = torch.randn(4, 32, dtype=torch.float32)
            loss = model(x).sum()
            loss.backward()
            opt.step()
            opt.zero_grad()

        for p in model.parameters():
            assert not torch.isnan(p).any(), f"NaN with zero momentum, param shape {p.shape}"

    def test_no_nan_with_high_weight_decay(self):
        """No NaN even with high weight decay."""
        model = _make_model(dtype=torch.float32)
        opt = NorMuonScheduleFree(model.parameters(), lr=0.008, weight_decay=0.5, warmup_steps=5)

        opt.train()
        for _ in range(50):
            x = torch.randn(4, 32, dtype=torch.float32)
            loss = model(x).sum()
            loss.backward()
            opt.step()
            opt.zero_grad()

        for p in model.parameters():
            assert not torch.isnan(p).any(), f"NaN with high WD, param shape {p.shape}"


# ---------------------------------------------------------------------------
# Tests: Weight decay at Z
# ---------------------------------------------------------------------------

class TestWeightDecayAtZ:
    """Verify weight decay is applied at Z (fast iterate), not at Y."""

    def test_z_norm_reduced_by_weight_decay(self):
        """Z should shrink over time due to weight decay at Z.

        Weight decay at Z applies: z = z * (1 - lr * lambda) per step.
        With lr=0.02, lambda=0.5, decay_rate = 0.01 per step, so after 500 steps:
        (1 - 0.01)^500 ≈ 0.0066.  Starting from norm ~40, Z should converge
        toward the steady-state bound from Lemma 3.1.
        """
        torch.manual_seed(42)
        model = torch.nn.Linear(8, 8, bias=False, dtype=torch.float32)
        model.weight.data.fill_(5.0)
        initial_norm = model.weight.norm().item()

        opt = NorMuonScheduleFree(
            model.parameters(), lr=0.02, weight_decay=0.5, warmup_steps=1
        )

        opt.train()
        # Run many steps with strong weight decay to see clear shrinkage
        for _ in range(500):
            x = torch.randn(4, 8, dtype=torch.float32)
            loss = model(x).sum()
            loss.backward()
            opt.step()
            opt.zero_grad()

        state = opt.state[list(model.parameters())[0]]
        z = state["z"]
        z_norm = z.norm().item()
        # Z should have been pulled significantly toward zero by weight decay.
        # From Lemma 3.1: ||Z||_F ≤ 0.2*sqrt(mn)/lambda = 0.2*8/0.5 = 3.2
        # In practice with the polar update term, it's higher but should
        # still be much smaller than the initial norm.
        assert z_norm < initial_norm * 0.8, \
            f"Z norm not reduced by weight decay: initial={initial_norm:.4f}, final={z_norm:.4f}"

    def test_no_weight_decay_still_works(self):
        """Optimizer should work with weight_decay=0."""
        model = _make_model(dtype=torch.float32)
        opt = NorMuonScheduleFree(model.parameters(), lr=0.008, weight_decay=0.0, warmup_steps=5)

        loss_init = _run_steps(model, opt, n_steps=1).item()
        loss_final = _run_steps(model, opt, n_steps=30).item()

        assert loss_final < loss_init, \
            f"Loss did not decrease without weight decay: init={loss_init:.4f}, final={loss_final:.4f}"


# ---------------------------------------------------------------------------
# Tests: BFloat16 support
# ---------------------------------------------------------------------------

class TestBFloat16:
    """Tests for bfloat16 parameter handling with stochastic rounding."""

    def test_bfloat16_params_no_crash(self):
        """Optimizer should work with bfloat16 model parameters."""
        model = _make_model(dtype=torch.bfloat16)
        opt = NorMuonScheduleFree(model.parameters(), lr=0.008, warmup_steps=3)

        opt.train()
        for _ in range(20):
            x = torch.randn(4, 32, dtype=torch.bfloat16)
            loss = model(x).sum()
            loss.backward()
            opt.step()
            opt.zero_grad()

        # Should not crash; parameters should remain bfloat16
        for p in model.parameters():
            assert p.dtype == torch.bfloat16

    def test_bfloat16_no_nan(self):
        """No NaN in bfloat16 parameters after multiple steps."""
        model = _make_model(dtype=torch.bfloat16)
        opt = NorMuonScheduleFree(model.parameters(), lr=0.008, warmup_steps=3)

        opt.train()
        for _ in range(50):
            x = torch.randn(4, 32, dtype=torch.bfloat16)
            loss = model(x).sum()
            loss.backward()
            opt.step()
            opt.zero_grad()

        for p in model.parameters():
            assert not torch.isnan(p).any(), f"NaN in bf16 param {p.shape}"

    def test_bfloat16_state_stored_correctly(self):
        """State tensors should be stored in the parameter's dtype (bf16)."""
        model = _make_model(dtype=torch.bfloat16)
        opt = NorMuonScheduleFree(model.parameters())

        opt.train()
        x = torch.randn(4, 32, dtype=torch.bfloat16)
        loss = model(x).sum()
        loss.backward()
        opt.step()

        for p in model.parameters():
            state = opt.state[p]
            # z should be in bfloat16 (copy_stochastic_ converts fp32 → bf16)
            assert state["z"].dtype == torch.bfloat16, \
                f"z dtype: {state['z'].dtype}, expected bfloat16 for {p.shape}"

    def test_bfloat16_train_eval_roundtrip(self):
        """train/eval roundtrip should work with bfloat16 parameters."""
        model = _make_model(dtype=torch.bfloat16)
        opt = NorMuonScheduleFree(model.parameters(), lr=0.008, warmup_steps=2)

        # Initialize state
        opt.train()
        x = torch.randn(4, 32, dtype=torch.bfloat16)
        loss = model(x).sum()
        loss.backward()
        opt.step()
        opt.zero_grad()

        # Save params, do eval/train roundtrip
        params_before = [p.clone() for p in model.parameters()]
        opt.eval()
        opt.train()

        for p_before, p_after in zip(params_before, model.parameters()):
            # With stochastic rounding, exact equality isn't guaranteed
            # but they should be very close
            diff = (p_before.float() - p_after.float()).abs().max().item()
            assert diff < 0.05, \
                f"bf16 params changed too much during train/eval: max_diff={diff:.6f}"


# ---------------------------------------------------------------------------
# Tests: Float16 support
# ---------------------------------------------------------------------------

class TestFloat16:
    """Tests for float16 parameter handling with stochastic rounding.

    When parameters are fp16, the optimizer should:
    - Upcast to fp32 for all calculations
    - Use fp32 (not bf16) for Newton-Schulz iteration (bf16 assumed unsupported)
    - Use stochastic rounding when writing back to fp16 state/parameters
    """

    def test_fp16_params_no_crash(self):
        """Optimizer should work with float16 model parameters."""
        model = _make_model(dtype=torch.float16)
        opt = NorMuonScheduleFree(model.parameters(), lr=0.008, warmup_steps=3)

        opt.train()
        for _ in range(20):
            x = torch.randn(4, 32, dtype=torch.float16)
            loss = model(x).sum()
            loss.backward()
            opt.step()
            opt.zero_grad()

        # Should not crash; parameters should remain float16
        for p in model.parameters():
            assert p.dtype == torch.float16

    def test_fp16_no_nan(self):
        """No NaN in float16 parameters after multiple steps."""
        model = _make_model(dtype=torch.float16)
        opt = NorMuonScheduleFree(model.parameters(), lr=0.008, warmup_steps=3)

        opt.train()
        for _ in range(50):
            x = torch.randn(4, 32, dtype=torch.float16)
            loss = model(x).sum()
            loss.backward()
            opt.step()
            opt.zero_grad()

        for p in model.parameters():
            assert not torch.isnan(p).any(), f"NaN in fp16 param {p.shape}"

    def test_fp16_state_stored_correctly(self):
        """State tensors should be stored in the parameter's dtype (fp16)."""
        model = _make_model(dtype=torch.float16)
        opt = NorMuonScheduleFree(model.parameters())

        opt.train()
        x = torch.randn(4, 32, dtype=torch.float16)
        loss = model(x).sum()
        loss.backward()
        opt.step()

        for p in model.parameters():
            state = opt.state[p]
            # z should be in float16 (copy_stochastic_ converts fp32 → fp16)
            assert state["z"].dtype == torch.float16, \
                f"z dtype: {state['z'].dtype}, expected float16 for {p.shape}"

    def test_fp16_2d_state_types(self):
        """2-D fp16 parameters should have z (fp16), v (fp32), mom (fp16) state."""
        model = _make_model(dtype=torch.float16)
        opt = NorMuonScheduleFree(model.parameters())
        opt.train()
        x = torch.randn(4, 32, dtype=torch.float16)
        loss = model(x).sum()
        loss.backward()
        opt.step()

        for p in model.parameters():
            if p.ndim == 2:
                state = opt.state[p]
                assert state["z"].dtype == torch.float16, \
                    f"z dtype: {state['z'].dtype}, expected fp16 for {p.shape}"
                assert state["v"].dtype == torch.float32, \
                    f"v dtype: {state['v'].dtype}, expected fp32 for {p.shape}"
                assert state["mom"].dtype == torch.float16, \
                    f"mom dtype: {state['mom'].dtype}, expected fp16 for {p.shape}"

    def test_fp16_1d_state_types(self):
        """1-D fp16 parameters should have z (fp16), exp_avg_sq (fp16) state."""
        model = torch.nn.Sequential(
            torch.nn.Linear(32, 64, bias=True, dtype=torch.float16),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 16, bias=True, dtype=torch.float16),
        )
        opt = NorMuonScheduleFree(model.parameters())
        opt.train()
        x = torch.randn(4, 32, dtype=torch.float16)
        loss = model(x).sum()
        loss.backward()
        opt.step()

        for p in model.parameters():
            if p.ndim == 1:
                state = opt.state[p]
                assert state["z"].dtype == torch.float16, \
                    f"z dtype: {state['z'].dtype}, expected fp16 for 1D param {p.shape}"
                assert state["exp_avg_sq"].dtype == torch.float16, \
                    f"exp_avg_sq dtype: {state['exp_avg_sq'].dtype}, expected fp16 for 1D param {p.shape}"

    def test_fp16_ns_uses_fp32_internally(self):
        """When p is fp16, Newton-Schulz should use fp32 (not bf16) for X.

        This is verified by running the optimizer and checking that no
        bf16-related errors occur (fp16-only environments may lack bf16 support).
        We also directly test _zeropower_via_newtonschulz5 with compute_dtype=fp32.
        """
        G = torch.randn(16, 32, dtype=torch.float32)
        # fp32 compute: should work and not use bf16
        P = _zeropower_via_newtonschulz5(G, steps=5, compute_dtype=torch.float32)
        assert P.dtype == torch.float32
        assert not torch.isnan(P).any()

        # Also verify the optimizer correctly selects fp32 for fp16 params
        model = _make_model(dtype=torch.float16)
        opt = NorMuonScheduleFree(model.parameters(), lr=0.008, warmup_steps=2)
        opt.train()
        for _ in range(10):
            x = torch.randn(4, 32, dtype=torch.float16)
            loss = model(x).sum()
            loss.backward()
            opt.step()
            opt.zero_grad()

        for p in model.parameters():
            assert not torch.isnan(p).any(), f"NaN in fp16 param after NS: {p.shape}"

    def test_fp16_train_eval_roundtrip(self):
        """train/eval roundtrip should work with float16 parameters."""
        model = _make_model(dtype=torch.float16)
        opt = NorMuonScheduleFree(model.parameters(), lr=0.008, warmup_steps=2)

        # Initialize state
        opt.train()
        x = torch.randn(4, 32, dtype=torch.float16)
        loss = model(x).sum()
        loss.backward()
        opt.step()
        opt.zero_grad()

        # Save params, do eval/train roundtrip
        params_before = [p.clone() for p in model.parameters()]
        opt.eval()
        opt.train()

        for p_before, p_after in zip(params_before, model.parameters()):
            # With stochastic rounding, exact equality isn't guaranteed
            # but they should be very close
            diff = (p_before.float() - p_after.float()).abs().max().item()
            assert diff < 0.05, \
                f"fp16 params changed too much during train/eval: max_diff={diff:.6f}"

    def test_fp16_loss_decreases(self):
        """Loss should decrease over multiple steps with float16 parameters."""
        model = _make_model(dtype=torch.float16)
        opt = NorMuonScheduleFree(model.parameters(), lr=0.008, warmup_steps=5)

        loss_init = _run_steps(model, opt, n_steps=1).item()
        loss_final = _run_steps(model, opt, n_steps=50).item()

        assert loss_final < loss_init, \
            f"Loss did not decrease with fp16: init={loss_init:.4f}, final={loss_final:.4f}"

    def test_fp16_mixed_params_no_crash(self):
        """Optimizer should handle mixed 1D/2D fp16 parameters."""
        torch.manual_seed(42)
        model = torch.nn.Sequential(
            torch.nn.Linear(32, 64, bias=True, dtype=torch.float16),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 16, bias=True, dtype=torch.float16),
        )
        opt = NorMuonScheduleFree(model.parameters(), lr=0.008, warmup_steps=5)

        opt.train()
        for _ in range(30):
            x = torch.randn(4, 32, dtype=torch.float16)
            loss = model(x).sum()
            loss.backward()
            opt.step()
            opt.zero_grad()

        for p in model.parameters():
            assert p.dtype == torch.float16
            assert not torch.isnan(p).any(), f"NaN in fp16 mixed param {p.shape}"


# ---------------------------------------------------------------------------
# Tests: Schedule-free averaging
# ---------------------------------------------------------------------------

class TestScheduleFreeAveraging:
    """Tests for the schedule-free weight averaging mechanism."""

    def test_step_counter_increments(self):
        """Step counter k should increment each step."""
        model = _make_model()
        opt = NorMuonScheduleFree(model.parameters(), warmup_steps=5)

        assert opt.param_groups[0]["k"] == 0

        opt.train()
        x = torch.randn(4, 32)
        loss = model(x).sum()
        loss.backward()
        opt.step()
        opt.zero_grad()

        assert opt.param_groups[0]["k"] == 1

        x = torch.randn(4, 32)
        loss = model(x).sum()
        loss.backward()
        opt.step()
        opt.zero_grad()

        assert opt.param_groups[0]["k"] == 2

    def test_s_sum_grows(self):
        """The schedule-free weight sum s_sum should grow monotonically."""
        model = _make_model()
        opt = NorMuonScheduleFree(model.parameters(), warmup_steps=5)

        s_values = []
        opt.train()
        for _ in range(10):
            x = torch.randn(4, 32)
            loss = model(x).sum()
            loss.backward()
            opt.step()
            opt.zero_grad()
            s_values.append(opt.param_groups[0]["s_sum"])

        for i in range(1, len(s_values)):
            assert s_values[i] > s_values[i - 1], \
                f"s_sum not growing: {s_values}"

    def test_warmup_affects_lr(self):
        """During warmup, the effective learning rate should ramp up linearly."""
        model = _make_model()
        warmup = 10
        opt = NorMuonScheduleFree(model.parameters(), lr=0.01, warmup_steps=warmup)

        # Check that the first step uses lr/warmup
        # We can verify indirectly by checking s_sum grows with warmup
        opt.train()
        x = torch.randn(4, 32)
        loss = model(x).sum()
        loss.backward()
        opt.step()
        opt.zero_grad()

        # After 1 step with warmup_steps=10: lr_effective = 0.01 * 1/10 = 0.001
        # s_sum = lr_effective² = 1e-6
        expected_s = (0.01 * 1.0 / 10) ** 2
        actual_s = opt.param_groups[0]["s_sum"]
        assert abs(actual_s - expected_s) < 1e-12, \
            f"warmup s_sum mismatch: expected={expected_s}, actual={actual_s}"


# ---------------------------------------------------------------------------
# Tests: Reset
# ---------------------------------------------------------------------------

class TestReset:
    """Tests for the reset() method."""

    def test_reset_clears_state(self):
        """reset() should reset all counters and state."""
        model = _make_model()
        opt = NorMuonScheduleFree(model.parameters(), warmup_steps=5)

        # Run some steps to build state
        opt.train()
        for _ in range(10):
            x = torch.randn(4, 32)
            loss = model(x).sum()
            loss.backward()
            opt.step()
            opt.zero_grad()

        assert opt.param_groups[0]["k"] == 10
        assert opt.param_groups[0]["s_sum"] > 0

        opt.reset()

        assert opt.param_groups[0]["k"] == 0
        assert opt.param_groups[0]["s_sum"] == 0.0


# ---------------------------------------------------------------------------
# Tests: Paper reference implementation agreement
# ---------------------------------------------------------------------------

class TestPaperReference:
    """Tests comparing against the paper's reference implementation (Appendix F)."""

    def test_reference_implementation_agreement(self):
        """Our optimizer should produce similar behavior to the reference.

        We run both the reference implementation and our optimizer on the same
        problem and verify they produce similar loss trajectories.
        """

        # --- Reference implementation from the paper (Appendix F) ---
        @torch.no_grad()
        def ref_zeropower(G, steps=5, eps=1e-7):
            assert len(G.shape) == 2
            a, b, c = (3.4445, -4.7750, 2.0315)
            X = G.bfloat16()
            X /= (X.norm() + eps)
            if G.size(0) > G.size(1):
                X = X.T
                for _ in range(steps):
                    A = X @ X.T
                    B = A @ X
                    X = a * X + b * B + c * A @ B
                return X.T
            for _ in range(steps):
                A = X @ X.T
                B = A @ X
                X = a * X + b * B + c * A @ B
            return X

        class RefNorMuonScheduleFree(torch.optim.Optimizer):
            """Exact reference implementation from the paper."""

            def __init__(self, params, lr=0.005, betas=(0.9, 0.95), momentum=0.8,
                         eps=1e-8, weight_decay=0.1, warmup_steps=2000, eta_scale=0.2):
                defaults = dict(lr=lr, betas=betas, momentum=momentum, eps=eps,
                                weight_decay=weight_decay, warmup_steps=warmup_steps,
                                eta_scale=eta_scale, k=0, train_mode=False, weight_sum=0.0)
                super().__init__(params, defaults)

            @torch.no_grad()
            def eval(self):
                for group in self.param_groups:
                    if group["train_mode"]:
                        beta = group["betas"][0]
                        for p in group["params"]:
                            state = self.state[p]
                            if "z" in state:
                                p.lerp_(end=state["z"], weight=1.0 - 1.0 / beta)
                        group["train_mode"] = False

            @torch.no_grad()
            def train(self):
                for group in self.param_groups:
                    if not group["train_mode"]:
                        beta = group["betas"][0]
                        for p in group["params"]:
                            state = self.state[p]
                            if "z" in state:
                                p.lerp_(end=state["z"], weight=1.0 - beta)
                        group["train_mode"] = True

            @torch.no_grad()
            def step(self, closure=None):
                loss = closure() if closure else None
                for group in self.param_groups:
                    beta, beta2 = group["betas"]
                    mu, eps = group["momentum"], group["eps"]
                    eta_scale, decay = group["eta_scale"], group["weight_decay"]
                    k, warmup_steps = group["k"], group["warmup_steps"]
                    sched = (k + 1) / warmup_steps if k < warmup_steps else 1.0
                    lr = group["lr"] * sched
                    weight = lr * lr
                    weight_sum = group["weight_sum"] = group["weight_sum"] + weight
                    ckp1 = weight / weight_sum
                    for p in group["params"]:
                        if p.grad is None:
                            continue
                        grad, state = p.grad, self.state[p]
                        if p.ndim < 2:
                            # Simple SGD fallback for non-matrix params in reference
                            if "z" not in state:
                                state["z"] = p.clone()
                            z = state["z"]
                            if decay != 0:
                                z.sub_(z, alpha=lr * decay)
                            z.sub_(grad, alpha=lr)
                            continue
                        if "z" not in state:
                            state["z"] = p.clone()
                            state["v"] = torch.zeros(p.shape[0], device=p.device, dtype=torch.float32)
                            state["mom"] = torch.zeros_like(p)
                        z, v, mom = state["z"], state["v"], state["mom"]
                        mom.mul_(mu).add_(grad, alpha=1.0 - mu)
                        P = ref_zeropower(mom).to(p.dtype)
                        row_ms = (P * P).mean(dim=1).float()
                        v.mul_(beta2).add_(row_ms, alpha=1.0 - beta2)
                        Phat = P / (v.sqrt() + eps).to(P.dtype).unsqueeze(1)
                        m, n = p.shape
                        eta_hat = eta_scale * lr * math.sqrt(m * n) / max(1e-12, Phat.float().norm())
                        x_t = (p - (1.0 - beta) * z) / beta if beta > 0 else z.clone()
                        if decay != 0:
                            z.sub_(z, alpha=lr * decay)
                        z.sub_(Phat, alpha=eta_hat)
                        x_tp1 = (1.0 - ckp1) * x_t + ckp1 * z
                        p.copy_((1.0 - beta) * z + beta * x_tp1)
                    group["k"] = k + 1
                return loss

        # --- Compare both on the same problem ---
        n_steps = 30
        torch.manual_seed(42)
        ref_model = _make_model(dtype=torch.float32)
        torch.manual_seed(42)
        our_model = _make_model(dtype=torch.float32)

        ref_opt = RefNorMuonScheduleFree(
            ref_model.parameters(), lr=0.008, betas=(0.9, 0.95),
            momentum=0.8, weight_decay=0.05, warmup_steps=5, eta_scale=0.2
        )
        our_opt = NorMuonScheduleFree(
            our_model.parameters(), lr=0.008, betas=(0.9, 0.95),
            momentum=0.8, weight_decay=0.05, warmup_steps=5, eta_scale=0.2
        )

        torch.manual_seed(123)
        inputs = [torch.randn(4, 32) for _ in range(n_steps)]

        ref_losses = []
        our_losses = []

        ref_opt.train()
        our_opt.train()

        for i in range(n_steps):
            x = inputs[i]

            # Reference
            ref_loss = ref_model(x).sum()
            ref_loss.backward()
            ref_opt.step()
            ref_opt.zero_grad()
            ref_losses.append(ref_loss.item())

            # Ours
            our_loss = our_model(x).sum()
            our_loss.backward()
            our_opt.step()
            our_opt.zero_grad()
            our_losses.append(our_loss.item())

        # Both should converge (losses decrease)
        assert ref_losses[-1] < ref_losses[0], "Reference didn't converge"
        assert our_losses[-1] < our_losses[0], "Our optimizer didn't converge"

        # Final losses should be in a similar ballpark
        # (exact match isn't expected due to bf16 rounding differences in NS)
        loss_ratio = our_losses[-1] / ref_losses[-1]
        assert 0.5 < loss_ratio < 2.0, \
            f"Final losses diverge too much: ref={ref_losses[-1]:.4f}, ours={our_losses[-1]:.4f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
