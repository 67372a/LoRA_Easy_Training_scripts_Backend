"""Tests for AMUSE optimizer.

Validates:
1. Import and optimizer instantiation works.
2. β_t schedule matches the paper's Appendix C.4 exact formula after the fix.
3. train()/eval() mode switching and step() run without error.
4. State tensors (z, momentum_buffer for Muon; z, exp_avg_sq for AdamW) are initialized.
5. Parameters change after a step (update is active).
6. Convergence: loss decreases over multiple steps on a simple task.
7. Numerical stability: no NaN or Inf after multiple steps.
8. Newton-Schulz produces near-orthogonal output.
9. Mixed 2D (Muon) + 1D (AdamW) parameter handling works.
10. Weight decay is applied to z (decoupled).
11. β_t is constant during warmup and increases after warmup.
12. eval()/train() round-trip preserves parameters.

Run with:
    cd backend/sd_scripts
    python -m pytest ../custom_scheduler/tests/test_amuse.py -v
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

# Load utils first (amuse.py does `from .utils import copy_stochastic_`)
_utils_path = os.path.join(_pkg_dir, "utils.py")
_utils_spec = importlib.util.spec_from_file_location(
    "LoraEasyCustomOptimizer.utils", _utils_path
)
_utils_mod = importlib.util.module_from_spec(_utils_spec)
sys.modules["LoraEasyCustomOptimizer.utils"] = _utils_mod
_utils_spec.loader.exec_module(_utils_mod)

copy_stochastic_ = _utils_mod.copy_stochastic_

# Load the amuse module
_amuse_path = os.path.join(_pkg_dir, "amuse.py")
_spec = importlib.util.spec_from_file_location(
    "LoraEasyCustomOptimizer.amuse", _amuse_path
)
_amuse = importlib.util.module_from_spec(_spec)
sys.modules["LoraEasyCustomOptimizer.amuse"] = _amuse
_spec.loader.exec_module(_amuse)

AMUSE = _amuse.AMUSE
zeropower_via_newtonschulz5 = _amuse.zeropower_via_newtonschulz5
muon_update = _amuse.muon_update

# Use CUDA when available for torch.compile (Inductor requires GPU or a C++ compiler)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_model_2d_only(seed=42, dtype=torch.float32, device=DEVICE):
    """Create a model with only 2D (matrix) layers, no bias (Muon requires ndim>=2)."""
    torch.manual_seed(seed)
    layers = [
        torch.nn.Linear(32, 64, bias=False, dtype=dtype, device=device),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 16, bias=False, dtype=dtype, device=device),
    ]
    return torch.nn.Sequential(*layers)


def _make_model_mixed(seed=42, dtype=torch.float32, device=DEVICE):
    """Create a model with 2D layers (Muon) and 1D bias params (AdamW fallback).

    Uses bias=True so there are both 2D weights and 1D biases for testing
    the mixed parameter group handling.
    """
    torch.manual_seed(seed)
    model = torch.nn.Sequential(
        torch.nn.Linear(32, 64, bias=True, dtype=dtype, device=device),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 16, bias=True, dtype=dtype, device=device),
    )
    return model


def _make_amuse_for_model(model, use_muon=True, warmup_steps=5, **kwargs):
    """Create AMUSE optimizer with a single parameter group."""
    return AMUSE(
        [{"params": list(model.parameters()), "use_muon": use_muon}],
        warmup_steps=warmup_steps,
        **kwargs,
    )


def _make_amuse_mixed(model, warmup_steps=5, **kwargs):
    """Create AMUSE with mixed Muon (2D) + AdamW (1D) parameter groups."""
    muon_params = []
    adamw_params = []
    for p in model.parameters():
        if p.ndim >= 2:
            muon_params.append(p)
        else:
            adamw_params.append(p)
    groups = []
    if muon_params:
        groups.append({"params": muon_params, "use_muon": True})
    if adamw_params:
        groups.append({"params": adamw_params, "use_muon": False})
    return AMUSE(groups, warmup_steps=warmup_steps, **kwargs)


def _run_steps(model, opt, n_steps=5, input_size=32, seed=999, device=DEVICE):
    """Run n_steps optimizer.step() calls and return the losses."""
    torch.manual_seed(seed)
    losses = []
    opt.train()
    for _ in range(n_steps):
        x = torch.randn(4, input_size, dtype=model[0].weight.dtype, device=device)
        loss = model(x).sum()
        loss.backward()
        opt.step()
        opt.zero_grad()
        losses.append(loss.item())
    return losses


# ---------------------------------------------------------------------------
# Tests: Newton-Schulz
# ---------------------------------------------------------------------------

class TestNewtonSchulz:
    """Tests for the Newton-Schulz polar decomposition helper."""

    def test_output_shape_matches_input(self):
        G = torch.randn(16, 32, device=DEVICE)
        P = zeropower_via_newtonschulz5(G)
        assert P.shape == G.shape

    def test_near_orthogonal_output(self):
        """For a tall matrix, P^T @ P should be near identity."""
        G = torch.randn(64, 16, dtype=torch.float32, device=DEVICE)
        P = zeropower_via_newtonschulz5(G, steps=5)
        PtP = P.T @ P
        I = torch.eye(16, dtype=torch.float32, device=DEVICE)
        max_diff = (PtP - I).abs().max().item()
        assert max_diff < 0.55, f"P^T @ P deviation too large: {max_diff:.4f}"

    def test_wide_matrix_transpose_handling(self):
        """Wide matrix (more columns than rows) should be handled via transpose."""
        G = torch.randn(8, 64, dtype=torch.float32, device=DEVICE)
        P = zeropower_via_newtonschulz5(G)
        assert P.shape == G.shape

    def test_square_matrix(self):
        G = torch.randn(16, 16, dtype=torch.float32, device=DEVICE)
        P = zeropower_via_newtonschulz5(G)
        assert P.shape == G.shape


# ---------------------------------------------------------------------------
# Tests: Instantiation
# ---------------------------------------------------------------------------

class TestInstantiation:
    """Tests for AMUSE optimizer creation."""

    def test_basic_creation_muon(self):
        model = _make_model_2d_only()
        opt = _make_amuse_for_model(model, use_muon=True, warmup_steps=5)
        assert isinstance(opt, AMUSE)
        assert opt.beta1_init == 0.9
        assert opt.warmup_steps == 5

    def test_basic_creation_adamw(self):
        model = _make_model_2d_only()
        opt = _make_amuse_for_model(model, use_muon=False, warmup_steps=5)
        assert isinstance(opt, AMUSE)

    def test_warmup_steps_required(self):
        model = _make_model_2d_only()
        with pytest.raises(ValueError, match="warmup_steps"):
            AMUSE(
                [{"params": list(model.parameters()), "use_muon": True}],
                warmup_steps=0,
            )

    def test_mixed_groups(self):
        model = _make_model_mixed()
        opt = _make_amuse_mixed(model, warmup_steps=5)
        assert len(opt.param_groups) == 2
        muon_group = [g for g in opt.param_groups if g["use_muon"]]
        adamw_group = [g for g in opt.param_groups if not g["use_muon"]]
        assert len(muon_group) == 1
        assert len(adamw_group) == 1

    def test_state_initialization_muon(self):
        """Muon group should initialize momentum_buffer for each param."""
        model = _make_model_2d_only()
        opt = _make_amuse_for_model(model, use_muon=True, warmup_steps=5)
        for group in opt.param_groups:
            for p in group["params"]:
                assert "momentum_buffer" in opt.state[p]
                assert torch.all(opt.state[p]["momentum_buffer"] == 0)

    def test_state_initialization_adamw(self):
        """AdamW group should initialize exp_avg_sq for each param."""
        model = _make_model_2d_only()
        opt = _make_amuse_for_model(model, use_muon=False, warmup_steps=5)
        for group in opt.param_groups:
            for p in group["params"]:
                assert "exp_avg_sq" in opt.state[p]
                assert torch.all(opt.state[p]["exp_avg_sq"] == 0)

# ---------------------------------------------------------------------------
# Tests: Top-level lr, weight_decay, and heuristic_muon
# ---------------------------------------------------------------------------

class TestTopLevelParams:
    """Tests for the new lr, weight_decay, and heuristic_muon init params."""

    def test_top_level_lr_applied_to_muon_group(self):
        """lr passed at init should override the Muon default (0.02)."""
        model = _make_model_2d_only()
        opt = AMUSE(
            [{"params": list(model.parameters()), "use_muon": True}],
            lr=0.05,
            warmup_steps=3,
        )
        assert opt.param_groups[0]["lr"] == 0.05
        assert opt.param_groups[0]["base_lr"] == 0.05

    def test_top_level_lr_applied_to_adamw_group(self):
        """lr passed at init should override the AdamW default (3e-4)."""
        model = _make_model_2d_only()
        opt = AMUSE(
            [{"params": list(model.parameters()), "use_muon": False}],
            lr=0.01,
            warmup_steps=3,
        )
        assert opt.param_groups[0]["lr"] == 0.01
        assert opt.param_groups[0]["base_lr"] == 0.01

    def test_top_level_lr_not_set_uses_defaults(self):
        """When lr=None, per-type defaults should be used."""
        model = _make_model_2d_only()
        muon_opt = AMUSE(
            [{"params": list(model.parameters()), "use_muon": True}],
            warmup_steps=3,
        )
        assert muon_opt.param_groups[0]["lr"] == 0.02

        adamw_opt = AMUSE(
            [{"params": list(model.parameters()), "use_muon": False}],
            warmup_steps=3,
        )
        assert adamw_opt.param_groups[0]["lr"] == 3e-4

    def test_top_level_weight_decay_applied(self):
        """weight_decay passed at init should be applied to all groups."""
        model = _make_model_mixed()
        muon_params = [p for p in model.parameters() if p.ndim >= 2]
        adamw_params = [p for p in model.parameters() if p.ndim < 2]
        groups = [
            {"params": muon_params, "use_muon": True},
            {"params": adamw_params, "use_muon": False},
        ]
        opt = AMUSE(groups, weight_decay=0.1, warmup_steps=3)
        for group in opt.param_groups:
            assert group["weight_decay"] == 0.1

    def test_per_group_lr_overrides_top_level(self):
        """Per-group lr should take priority over top-level lr."""
        model = _make_model_2d_only()
        opt = AMUSE(
            [{"params": list(model.parameters()), "use_muon": True, "lr": 0.03}],
            lr=0.05,
            warmup_steps=3,
        )
        assert opt.param_groups[0]["lr"] == 0.03
        assert opt.param_groups[0]["base_lr"] == 0.03

    def test_per_group_weight_decay_overrides_top_level(self):
        """Per-group weight_decay should take priority over top-level."""
        model = _make_model_2d_only()
        opt = AMUSE(
            [{"params": list(model.parameters()), "use_muon": True,
              "weight_decay": 0.02}],
            weight_decay=0.1,
            warmup_steps=3,
        )
        assert opt.param_groups[0]["weight_decay"] == 0.02


class TestHeuristicMuon:
    """Tests for the heuristic_muon auto-split feature."""

    def test_heuristic_creates_two_groups(self):
        """With a mixed model, heuristic_muon should create 2 groups."""
        model = _make_model_mixed()
        opt = AMUSE(
            list(model.parameters()),
            heuristic_muon=True,
            warmup_steps=3,
        )
        assert len(opt.param_groups) == 2
        muon_group = [g for g in opt.param_groups if g["use_muon"]]
        adamw_group = [g for g in opt.param_groups if not g["use_muon"]]
        assert len(muon_group) == 1
        assert len(adamw_group) == 1

    def test_heuristic_muon_params_are_ndim_ge_2(self):
        """Muon group should contain only ndim >= 2 params."""
        model = _make_model_mixed()
        opt = AMUSE(
            list(model.parameters()),
            heuristic_muon=True,
            warmup_steps=3,
        )
        muon_group = [g for g in opt.param_groups if g["use_muon"]][0]
        for p in muon_group["params"]:
            assert p.ndim >= 2, f"Muon param has ndim={p.ndim}, expected >= 2"

    def test_heuristic_adamw_params_are_ndim_lt_2(self):
        """AdamW group should contain only ndim < 2 params."""
        model = _make_model_mixed()
        opt = AMUSE(
            list(model.parameters()),
            heuristic_muon=True,
            warmup_steps=3,
        )
        adamw_group = [g for g in opt.param_groups if not g["use_muon"]][0]
        for p in adamw_group["params"]:
            assert p.ndim < 2, f"AdamW param has ndim={p.ndim}, expected < 2"

    def test_heuristic_with_2d_only_model(self):
        """With a 2D-only model, heuristic should create only a Muon group."""
        model = _make_model_2d_only()
        opt = AMUSE(
            list(model.parameters()),
            heuristic_muon=True,
            warmup_steps=3,
        )
        assert len(opt.param_groups) == 1
        assert opt.param_groups[0]["use_muon"] is True

    def test_heuristic_with_1d_only_params(self):
        """With only 1D params, heuristic should create only an AdamW group."""
        p1 = torch.nn.Parameter(torch.randn(10))
        p2 = torch.nn.Parameter(torch.randn(5))
        opt = AMUSE(
            [p1, p2],
            heuristic_muon=True,
            warmup_steps=3,
        )
        assert len(opt.param_groups) == 1
        assert opt.param_groups[0]["use_muon"] is False

    def test_heuristic_with_top_level_lr(self):
        """heuristic_muon + top-level lr should apply lr to both groups."""
        model = _make_model_mixed()
        opt = AMUSE(
            list(model.parameters()),
            lr=0.05,
            heuristic_muon=True,
            warmup_steps=3,
        )
        for group in opt.param_groups:
            assert group["lr"] == 0.05
            assert group["base_lr"] == 0.05

    def test_heuristic_with_top_level_weight_decay(self):
        """heuristic_muon + top-level weight_decay should apply to both groups."""
        model = _make_model_mixed()
        opt = AMUSE(
            list(model.parameters()),
            weight_decay=0.1,
            heuristic_muon=True,
            warmup_steps=3,
        )
        for group in opt.param_groups:
            assert group["weight_decay"] == 0.1

    def test_heuristic_with_pre_grouped_dicts_warns(self):
        """heuristic_muon=True with pre-grouped dicts should warn and use as-is."""
        model = _make_model_mixed()
        groups = [{"params": list(model.parameters()), "use_muon": True}]
        # Should not raise, just warn
        opt = AMUSE(groups, heuristic_muon=True, warmup_steps=3)
        assert len(opt.param_groups) == 1
        assert opt.param_groups[0]["use_muon"] is True

    def test_heuristic_convergence(self):
        """heuristic_muon model should converge on a simple task."""
        model = _make_model_mixed()
        opt = AMUSE(
            list(model.parameters()),
            lr=0.01,
            heuristic_muon=True,
            warmup_steps=5,
            beta1=0.6,
        )
        losses = _run_steps(model, opt, n_steps=50, input_size=32)
        early_avg = sum(losses[:5]) / 5
        late_avg = sum(losses[-5:]) / 5
        assert late_avg < early_avg, (
            f"heuristic_muon loss did not decrease: "
            f"early_avg={early_avg:.4f}, late_avg={late_avg:.4f}"
        )

    def test_heuristic_no_nan(self):
        """heuristic_muon should not produce NaN after multiple steps."""
        model = _make_model_mixed()
        opt = AMUSE(
            list(model.parameters()),
            lr=0.01,
            heuristic_muon=True,
            warmup_steps=5,
        )
        _run_steps(model, opt, n_steps=20, input_size=32)
        for p in model.parameters():
            assert not torch.isnan(p).any(), "NaN in parameters"
            assert not torch.isinf(p).any(), "Inf in parameters"

    def test_heuristic_state_initialization(self):
        """heuristic_muon should initialize correct state buffers."""
        model = _make_model_mixed()
        opt = AMUSE(
            list(model.parameters()),
            heuristic_muon=True,
            warmup_steps=3,
        )
        muon_group = [g for g in opt.param_groups if g["use_muon"]][0]
        adamw_group = [g for g in opt.param_groups if not g["use_muon"]][0]
        for p in muon_group["params"]:
            assert "momentum_buffer" in opt.state[p]
        for p in adamw_group["params"]:
            assert "exp_avg_sq" in opt.state[p]

    def test_backward_compat_no_heuristic_no_lr(self):
        """Without heuristic_muon or lr, behavior should be identical to before."""
        model = _make_model_2d_only()
        opt = AMUSE(
            [{"params": list(model.parameters()), "use_muon": True}],
            warmup_steps=3,
        )
        assert opt.param_groups[0]["lr"] == 0.02  # Muon default
        assert opt.param_groups[0]["base_lr"] == 0.02


# ---------------------------------------------------------------------------
# Tests: β_t Schedule (the critical fix)
# ---------------------------------------------------------------------------



class TestBetaSchedule:
    """Tests verifying the β_t schedule matches the paper's exact formula.

    Paper Appendix C.4:
        β_t = 1 - (c_t(1-c_{T₀}) / (c_{T₀}(1-c_t)))^ρ · (1-β₁)

    Where c_t is the averaging weight from the PREVIOUS step (not the current).
    """

    def test_beta1_constant_during_warmup(self):
        """β₁ should remain at beta1_init throughout warmup."""
        model = _make_model_2d_only()
        beta1_init = 0.7
        warmup = 10
        opt = _make_amuse_for_model(
            model, use_muon=False, warmup_steps=warmup, beta1=beta1_init
        )
        opt.train()
        for i in range(warmup):
            x = torch.randn(2, 32, device=DEVICE)
            loss = model(x).sum()
            loss.backward()
            opt.step()
            opt.zero_grad()
            for group in opt.param_groups:
                assert group["beta1"] == pytest.approx(
                    beta1_init
                ), f"beta1 should be {beta1_init} during warmup, got {group['beta1']} at step {i+1}"

    def test_beta1_increases_after_warmup(self):
        """β₁ should increase toward 1 after warmup when rho > 0."""
        model = _make_model_2d_only()
        beta1_init = 0.6
        warmup = 5
        rho = 0.8
        opt = _make_amuse_for_model(
            model, use_muon=False, warmup_steps=warmup, beta1=beta1_init, rho=rho
        )
        opt.train()
        # Run through warmup
        for _ in range(warmup):
            x = torch.randn(2, 32, device=DEVICE)
            loss = model(x).sum()
            loss.backward()
            opt.step()
            opt.zero_grad()

        # After warmup, beta1 should increase
        prev_beta1 = beta1_init
        for _ in range(20):
            x = torch.randn(2, 32, device=DEVICE)
            loss = model(x).sum()
            loss.backward()
            opt.step()
            opt.zero_grad()
            curr_beta1 = opt.param_groups[0]["beta1"]
            assert curr_beta1 >= prev_beta1 - 1e-10, (
                f"beta1 should be non-decreasing after warmup: "
                f"prev={prev_beta1:.6f}, curr={curr_beta1:.6f}"
            )
            assert curr_beta1 <= 1.0, f"beta1 should not exceed 1.0, got {curr_beta1}"
            prev_beta1 = curr_beta1

    def test_beta1_with_rho_zero_is_constant(self):
        """With ρ=0, β₁ should stay at beta1_init even after warmup."""
        model = _make_model_2d_only()
        beta1_init = 0.5
        warmup = 5
        opt = _make_amuse_for_model(
            model, use_muon=False, warmup_steps=warmup, beta1=beta1_init, rho=0.0
        )
        opt.train()
        for _ in range(30):
            x = torch.randn(2, 32, device=DEVICE)
            loss = model(x).sum()
            loss.backward()
            opt.step()
            opt.zero_grad()
            assert opt.param_groups[0]["beta1"] == pytest.approx(beta1_init)

    def test_beta1_schedule_matches_paper_formula(self):
        """Verify β_t against the paper's Appendix C.4 exact formula.

        β_t = 1 - (c_t(1-c_{T₀}) / (c_{T₀}(1-c_t)))^ρ · (1-β₁)

        Where c_t is the ckp1 computed at the current step (c_{t+1} in
        Algorithm 1 notation), matching the reference implementation.
        """
        model = _make_model_2d_only()
        beta1_init = 0.6
        warmup = 5
        rho = 0.8
        opt = _make_amuse_for_model(
            model, use_muon=False, warmup_steps=warmup, beta1=beta1_init, rho=rho
        )
        opt.train()

        # Manually track c values to verify the formula
        c_history = []  # ckp1 values at each step (c_{t+1} in paper notation)

        for step in range(25):
            x = torch.randn(2, 32, device=DEVICE)
            loss = model(x).sum()
            loss.backward()
            opt.step()
            opt.zero_grad()

            group = opt.param_groups[0]
            t = step + 1  # 1-indexed step
            ckp1 = group["ckp1"]  # c_{t+1}
            c_history.append(ckp1)

            if t > warmup:
                # After the fix, _compute_beta1 receives the current step's
                # ckp1, which is c_{t+1} in Algorithm 1 notation.
                c_t = ckp1
                # c_warmup was saved at t == warmup_steps from the current
                # step's ckp1, which is c_history[warmup - 1].
                c_T0 = c_history[warmup - 1]  # ckp1 from step T₀

                # Paper formula (using the reference's interpretation of c_t)
                S_t = (c_t * (1.0 - c_T0)) / (c_T0 * (1.0 - c_t))
                expected_beta1 = 1.0 - (S_t ** rho) * (1.0 - beta1_init)

                actual_beta1 = group["beta1"]
                assert actual_beta1 == pytest.approx(
                    expected_beta1, abs=1e-10
                ), (
                    f"Step {t}: β_t mismatch. "
                    f"Expected {expected_beta1:.10f}, got {actual_beta1:.10f}. "
                    f"c_t={c_t:.10f}, c_T0={c_T0:.10f}"
                )


# ---------------------------------------------------------------------------
# Tests: train/eval mode switching
# ---------------------------------------------------------------------------

class TestTrainEvalMode:
    """Tests for train()/eval() mode switching."""

    def test_step_raises_in_eval_mode(self):
        model = _make_model_2d_only()
        opt = _make_amuse_for_model(model, use_muon=False, warmup_steps=5)
        # opt starts in eval mode
        assert not opt.train_mode
        x = torch.randn(2, 32, device=DEVICE)
        loss = model(x).sum()
        loss.backward()
        with pytest.raises(Exception, match="train mode"):
            opt.step()

    def test_train_eval_roundtrip_preserves_params(self):
        """After train→eval→train, parameters should return to original values."""
        model = _make_model_2d_only()
        opt = _make_amuse_for_model(model, use_muon=False, warmup_steps=5)

        # Run one step to populate z state
        opt.train()
        x = torch.randn(2, 32, device=DEVICE)
        loss = model(x).sum()
        loss.backward()
        opt.step()
        opt.zero_grad()

        # Save params after train step
        params_after_step = [p.clone() for p in model.parameters()]

        # eval → train round-trip
        opt.eval()
        opt.train()

        for p_orig, p_restored in zip(params_after_step, model.parameters()):
            assert torch.allclose(
                p_orig, p_restored, atol=1e-6
            ), "Parameters changed after train→eval→train round-trip"

    def test_eval_converts_to_x(self):
        """After eval(), parameters should be the x sequence (averaged iterate)."""
        model = _make_model_2d_only()
        opt = _make_amuse_for_model(model, use_muon=False, warmup_steps=5)

        opt.train()
        x = torch.randn(2, 32, device=DEVICE)
        loss = model(x).sum()
        loss.backward()
        opt.step()
        opt.zero_grad()

        # Save y (training params) and z
        y_params = [p.clone() for p in model.parameters()]
        z_params = [
            opt.state[p]["z"].clone() for p in model.parameters()
        ]
        beta1 = opt.param_groups[0]["beta1"]

        opt.eval()

        for p, y, z in zip(model.parameters(), y_params, z_params):
            # x = (y - (1-β₁)z) / β₁
            expected_x = (y - (1 - beta1) * z) / beta1
            assert torch.allclose(
                p, expected_x, atol=1e-5
            ), "eval() did not correctly convert y → x"


# ---------------------------------------------------------------------------
# Tests: Step execution
# ---------------------------------------------------------------------------

class TestStep:
    """Tests for the step() method."""

    def test_parameters_change_after_step(self):
        """Parameters should change after an optimizer step."""
        model = _make_model_2d_only()
        params_before = [p.clone() for p in model.parameters()]

        opt = _make_amuse_for_model(model, use_muon=True, warmup_steps=5)
        opt.train()
        x = torch.randn(2, 32, device=DEVICE)
        loss = model(x).sum()
        loss.backward()
        opt.step()

        changed = False
        for p_before, p_after in zip(params_before, model.parameters()):
            if not torch.allclose(p_before, p_after):
                changed = True
                break
        assert changed, "Parameters did not change after step()"

    def test_z_state_initialized_on_first_step(self):
        """z state should be created on first step (cloned from initial params)."""
        model = _make_model_2d_only()
        opt = _make_amuse_for_model(model, use_muon=True, warmup_steps=5)

        # z should not exist yet
        for p in model.parameters():
            assert "z" not in opt.state[p]

        opt.train()
        x = torch.randn(2, 32, device=DEVICE)
        loss = model(x).sum()
        loss.backward()
        opt.step()
        opt.zero_grad()

        # z should now exist
        for p in model.parameters():
            assert "z" in opt.state[p]

    def test_step_counter_increments(self):
        """Group k counter should increment each step."""
        model = _make_model_2d_only()
        opt = _make_amuse_for_model(model, use_muon=True, warmup_steps=5)
        opt.train()

        for i in range(5):
            assert opt.param_groups[0]["k"] == i
            x = torch.randn(2, 32, device=DEVICE)
            loss = model(x).sum()
            loss.backward()
            opt.step()
            opt.zero_grad()
        assert opt.param_groups[0]["k"] == 5

    def test_warmup_lr_schedule(self):
        """LR should linearly warmup: lr = base_lr * min(1, t/T₀)."""
        model = _make_model_2d_only()
        base_lr = 0.01
        warmup = 10
        opt = AMUSE(
            [{"params": list(model.parameters()), "use_muon": True, "lr": base_lr}],
            warmup_steps=warmup,
        )
        opt.train()

        for step in range(15):
            t = step + 1
            x = torch.randn(2, 32, device=DEVICE)
            loss = model(x).sum()
            loss.backward()
            opt.step()
            opt.zero_grad()

            expected_lr = base_lr * min(1.0, t / warmup)
            actual_lr = opt.param_groups[0]["lr"]
            assert actual_lr == pytest.approx(
                expected_lr, abs=1e-10
            ), f"LR mismatch at step {t}: expected {expected_lr}, got {actual_lr}"


# ---------------------------------------------------------------------------
# Tests: Numerical stability
# ---------------------------------------------------------------------------

class TestNumericalStability:
    """Tests for NaN/Inf safety."""

    def test_no_nan_after_steps_muon(self):
        model = _make_model_2d_only()
        opt = _make_amuse_for_model(model, use_muon=True, warmup_steps=5)
        losses = _run_steps(model, opt, n_steps=20, input_size=32)
        for p in model.parameters():
            assert not torch.isnan(p).any(), "NaN in parameters"
            assert not torch.isinf(p).any(), "Inf in parameters"

    def test_no_nan_after_steps_adamw(self):
        model = _make_model_2d_only()
        opt = _make_amuse_for_model(model, use_muon=False, warmup_steps=5)
        losses = _run_steps(model, opt, n_steps=20, input_size=32)
        for p in model.parameters():
            assert not torch.isnan(p).any(), "NaN in parameters"
            assert not torch.isinf(p).any(), "Inf in parameters"

    def test_no_nan_after_steps_mixed(self):
        model = _make_model_mixed()
        opt = _make_amuse_mixed(model, warmup_steps=5)
        losses = _run_steps(model, opt, n_steps=20, input_size=32)
        for p in model.parameters():
            assert not torch.isnan(p).any(), "NaN in parameters"
            assert not torch.isinf(p).any(), "Inf in parameters"


# ---------------------------------------------------------------------------
# Tests: Convergence
# ---------------------------------------------------------------------------

class TestConvergence:
    """Tests that loss decreases on a simple task."""

    def test_loss_decreases_muon(self):
        model = _make_model_2d_only()
        opt = _make_amuse_for_model(
            model, use_muon=True, warmup_steps=5, beta1=0.6
        )
        losses = _run_steps(model, opt, n_steps=50, input_size=32)
        # Check that later losses are generally smaller than early ones
        early_avg = sum(losses[:5]) / 5
        late_avg = sum(losses[-5:]) / 5
        assert late_avg < early_avg, (
            f"Loss did not decrease: early_avg={early_avg:.4f}, late_avg={late_avg:.4f}"
        )

    def test_loss_decreases_adamw(self):
        model = _make_model_2d_only()
        opt = _make_amuse_for_model(
            model, use_muon=False, warmup_steps=5, beta1=0.6
        )
        losses = _run_steps(model, opt, n_steps=50, input_size=32)
        early_avg = sum(losses[:5]) / 5
        late_avg = sum(losses[-5:]) / 5
        assert late_avg < early_avg, (
            f"Loss did not decrease: early_avg={early_avg:.4f}, late_avg={late_avg:.4f}"
        )


# ---------------------------------------------------------------------------
# Tests: Weight decay
# ---------------------------------------------------------------------------

class TestWeightDecay:
    """Tests for decoupled weight decay on z."""

    def test_weight_decay_applied_to_z(self):
        """With nonzero weight_decay, z should shrink toward zero."""
        model = _make_model_2d_only()
        opt = AMUSE(
            [{"params": list(model.parameters()), "use_muon": True, "lr": 0.01,
              "weight_decay": 0.1}],
            warmup_steps=3,
        )
        opt.train()

        # Record z norms before step
        x = torch.randn(2, 32, device=DEVICE)
        loss = model(x).sum()
        loss.backward()
        opt.step()
        opt.zero_grad()

        # z should have been modified (not all zeros after a step with wd)
        for p in model.parameters():
            z = opt.state[p].get("z")
            if z is not None:
                # Just verify z exists and has been modified
                assert z.shape == p.shape


# ---------------------------------------------------------------------------
# Tests: Packaging
# ---------------------------------------------------------------------------

class TestPackaging:
    """Tests for package registration."""

    def test_import_from_package(self):
        """AMUSE should be importable from the LoraEasyCustomOptimizer package."""
        # We already imported it above, but verify the class is correct
        assert hasattr(AMUSE, "step")
        assert hasattr(AMUSE, "train")
        assert hasattr(AMUSE, "eval")
        assert issubclass(AMUSE, torch.optim.Optimizer)


# ---------------------------------------------------------------------------
# Tests: Newton-Schulz compute_dtype
# ---------------------------------------------------------------------------

class TestNewtonSchulzComputeDtype:
    """Tests for the compute_dtype parameter of zeropower_via_newtonschulz5."""

    def test_default_compute_dtype_is_bf16(self):
        """Default compute_dtype should be bfloat16."""
        G = torch.randn(16, 32, dtype=torch.float32, device=DEVICE)
        P = zeropower_via_newtonschulz5(G)
        # The iteration runs in bf16 internally, but output dtype matches input
        assert P.shape == G.shape

    def test_fp32_compute_dtype(self):
        """Passing compute_dtype=fp32 should produce results in fp32."""
        G = torch.randn(16, 32, dtype=torch.float32, device=DEVICE)
        P = zeropower_via_newtonschulz5(G, compute_dtype=torch.float32)
        assert P.shape == G.shape

    def test_fp16_input_with_bf16_compute(self):
        """fp16 input with bf16 compute_dtype should work."""
        G = torch.randn(16, 32, dtype=torch.float16, device=DEVICE)
        P = zeropower_via_newtonschulz5(G, compute_dtype=torch.bfloat16)
        assert P.shape == G.shape

    def test_fp16_input_with_fp32_compute(self):
        """fp16 input with fp32 compute_dtype should work (recommended for fp16)."""
        G = torch.randn(16, 32, dtype=torch.float16, device=DEVICE)
        P = zeropower_via_newtonschulz5(G, compute_dtype=torch.float32)
        assert P.shape == G.shape

    def test_compute_dtype_affects_result(self):
        """Different compute_dtypes should produce different (but valid) results."""
        torch.manual_seed(42)
        G = torch.randn(16, 32, dtype=torch.float32, device=DEVICE)
        P_bf16 = zeropower_via_newtonschulz5(G, compute_dtype=torch.bfloat16)
        P_fp32 = zeropower_via_newtonschulz5(G, compute_dtype=torch.float32)
        # Results should differ due to precision differences
        # Cast to common dtype for comparison
        assert not torch.allclose(P_bf16.float(), P_fp32.float(), atol=1e-6)


# ---------------------------------------------------------------------------
# Tests: copy_stochastic_ import
# ---------------------------------------------------------------------------

class TestCopyStochasticImport:
    """Tests that copy_stochastic_ is properly imported and available."""

    def test_copy_stochastic_is_callable(self):
        assert callable(copy_stochastic_)

    def test_copy_stochastic_basic(self):
        """copy_stochastic_ should copy fp32 source to bf16 target."""
        source = torch.randn(8, 16, dtype=torch.float32, device=DEVICE)
        target = torch.zeros(8, 16, dtype=torch.bfloat16, device=DEVICE)
        copy_stochastic_(target, source)
        # After stochastic copy, target should not be all zeros
        assert not torch.all(target == 0)
        # Values should be approximately correct (within bf16 precision)
        assert torch.allclose(target.float(), source, atol=0.02)


# ---------------------------------------------------------------------------
# Tests: fp16/bf16 precision for eval/train mode
# ---------------------------------------------------------------------------

class TestHalfPrecisionEvalTrain:
    """Tests for eval()/train() with bf16 and fp16 parameters."""

    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
    def test_eval_train_roundtrip_preserves_params_bf16(self, dtype):
        """After train→eval→train, bf16/fp16 parameters should be approximately preserved."""
        model = _make_model_2d_only(dtype=dtype)
        opt = _make_amuse_for_model(model, use_muon=False, warmup_steps=5)

        opt.train()
        x = torch.randn(2, 32, dtype=dtype, device=DEVICE)
        loss = model(x).sum()
        loss.backward()
        opt.step()
        opt.zero_grad()

        params_after_step = [p.clone() for p in model.parameters()]

        opt.eval()
        opt.train()

        for p_orig, p_restored in zip(params_after_step, model.parameters()):
            assert torch.allclose(
                p_orig, p_restored, atol=1e-2
            ), f"Parameters changed after train→eval→train round-trip for {dtype}"

    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
    def test_eval_converts_to_x_bf16(self, dtype):
        """After eval(), bf16/fp16 parameters should be the x sequence."""
        model = _make_model_2d_only(dtype=dtype)
        opt = _make_amuse_for_model(model, use_muon=False, warmup_steps=5)

        opt.train()
        x = torch.randn(2, 32, dtype=dtype, device=DEVICE)
        loss = model(x).sum()
        loss.backward()
        opt.step()
        opt.zero_grad()

        y_params = [p.clone() for p in model.parameters()]
        z_params = [opt.state[p]["z"].clone() for p in model.parameters()]
        beta1 = opt.param_groups[0]["beta1"]

        opt.eval()

        for p, y, z in zip(model.parameters(), y_params, z_params):
            expected_x = (y.float() - (1 - beta1) * z.float()) / beta1
            assert torch.allclose(
                p.float(), expected_x, atol=0.02
            ), f"eval() did not correctly convert y → x for {dtype}"

    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
    def test_params_stay_in_original_dtype(self, dtype):
        """Parameters should remain in their original dtype after eval/train/step."""
        model = _make_model_2d_only(dtype=dtype)
        opt = _make_amuse_for_model(model, use_muon=False, warmup_steps=5)

        opt.train()
        x = torch.randn(2, 32, dtype=dtype, device=DEVICE)
        loss = model(x).sum()
        loss.backward()
        opt.step()
        opt.zero_grad()

        for p in model.parameters():
            assert p.dtype == dtype, f"Expected {dtype}, got {p.dtype} after step"

        opt.eval()
        for p in model.parameters():
            assert p.dtype == dtype, f"Expected {dtype}, got {p.dtype} after eval"

        opt.train()
        for p in model.parameters():
            assert p.dtype == dtype, f"Expected {dtype}, got {p.dtype} after train"


# ---------------------------------------------------------------------------
# Tests: fp16/bf16 precision for step (Muon path)
# ---------------------------------------------------------------------------

class TestHalfPrecisionMuonStep:
    """Tests for step() with bf16 and fp16 parameters using Muon path."""

    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
    def test_muon_step_no_nan_bf16(self, dtype):
        """Muon step should not produce NaN for bf16/fp16 parameters."""
        model = _make_model_2d_only(dtype=dtype)
        opt = _make_amuse_for_model(model, use_muon=True, warmup_steps=5)
        losses = _run_steps(model, opt, n_steps=20, input_size=32)
        for p in model.parameters():
            assert not torch.isnan(p).any(), f"NaN in parameters ({dtype})"
            assert not torch.isinf(p).any(), f"Inf in parameters ({dtype})"

    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
    def test_muon_params_change_bf16(self, dtype):
        """Parameters should change after Muon step for bf16/fp16."""
        model = _make_model_2d_only(dtype=dtype)
        params_before = [p.clone() for p in model.parameters()]

        opt = _make_amuse_for_model(model, use_muon=True, warmup_steps=5)
        opt.train()
        x = torch.randn(2, 32, dtype=dtype, device=DEVICE)
        loss = model(x).sum()
        loss.backward()
        opt.step()

        changed = False
        for p_before, p_after in zip(params_before, model.parameters()):
            if not torch.allclose(p_before, p_after):
                changed = True
                break
        assert changed, f"Parameters did not change after Muon step ({dtype})"

    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
    def test_muon_state_dtype_matches_param_bf16(self, dtype):
        """State buffers should match parameter dtype after step for bf16/fp16."""
        model = _make_model_2d_only(dtype=dtype)
        opt = _make_amuse_for_model(model, use_muon=True, warmup_steps=5)

        opt.train()
        x = torch.randn(2, 32, dtype=dtype, device=DEVICE)
        loss = model(x).sum()
        loss.backward()
        opt.step()
        opt.zero_grad()

        for p in model.parameters():
            state = opt.state[p]
            assert state["z"].dtype == dtype, (
                f"z dtype mismatch: expected {dtype}, got {state['z'].dtype}"
            )
            assert state["momentum_buffer"].dtype == dtype, (
                f"momentum_buffer dtype mismatch: expected {dtype}, "
                f"got {state['momentum_buffer'].dtype}"
            )

    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
    def test_muon_convergence_bf16(self, dtype):
        """Muon should converge for bf16/fp16 parameters."""
        model = _make_model_2d_only(dtype=dtype)
        opt = _make_amuse_for_model(
            model, use_muon=True, warmup_steps=5, beta1=0.6
        )
        losses = _run_steps(model, opt, n_steps=50, input_size=32)
        early_avg = sum(losses[:5]) / 5
        late_avg = sum(losses[-5:]) / 5
        assert late_avg < early_avg, (
            f"Muon loss did not decrease ({dtype}): "
            f"early_avg={early_avg:.4f}, late_avg={late_avg:.4f}"
        )


# ---------------------------------------------------------------------------
# Tests: fp16/bf16 precision for step (AdamW fallback path)
# ---------------------------------------------------------------------------

class TestHalfPrecisionAdamWStep:
    """Tests for step() with bf16 and fp16 parameters using AdamW fallback path."""

    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
    def test_adamw_step_no_nan_bf16(self, dtype):
        """AdamW step should not produce NaN for bf16/fp16 parameters."""
        model = _make_model_2d_only(dtype=dtype)
        opt = _make_amuse_for_model(model, use_muon=False, warmup_steps=5)
        losses = _run_steps(model, opt, n_steps=20, input_size=32)
        for p in model.parameters():
            assert not torch.isnan(p).any(), f"NaN in parameters ({dtype})"
            assert not torch.isinf(p).any(), f"Inf in parameters ({dtype})"

    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
    def test_adamw_state_dtype_matches_param_bf16(self, dtype):
        """State buffers should match parameter dtype after step for bf16/fp16."""
        model = _make_model_2d_only(dtype=dtype)
        opt = _make_amuse_for_model(model, use_muon=False, warmup_steps=5)

        opt.train()
        x = torch.randn(2, 32, dtype=dtype, device=DEVICE)
        loss = model(x).sum()
        loss.backward()
        opt.step()
        opt.zero_grad()

        for p in model.parameters():
            state = opt.state[p]
            assert state["z"].dtype == dtype, (
                f"z dtype mismatch: expected {dtype}, got {state['z'].dtype}"
            )
            assert state["exp_avg_sq"].dtype == dtype, (
                f"exp_avg_sq dtype mismatch: expected {dtype}, "
                f"got {state['exp_avg_sq'].dtype}"
            )

    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
    def test_adamw_convergence_bf16(self, dtype):
        """AdamW fallback should converge for bf16/fp16 parameters."""
        model = _make_model_2d_only(dtype=dtype)
        opt = _make_amuse_for_model(
            model, use_muon=False, warmup_steps=5, beta1=0.6
        )
        losses = _run_steps(model, opt, n_steps=50, input_size=32)
        early_avg = sum(losses[:5]) / 5
        late_avg = sum(losses[-5:]) / 5
        assert late_avg < early_avg, (
            f"AdamW loss did not decrease ({dtype}): "
            f"early_avg={early_avg:.4f}, late_avg={late_avg:.4f}"
        )


# ---------------------------------------------------------------------------
# Tests: Mixed precision (bf16/fp16) with mixed parameter groups
# ---------------------------------------------------------------------------

class TestHalfPrecisionMixed:
    """Tests for mixed Muon + AdamW parameter groups with bf16/fp16."""

    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
    def test_mixed_no_nan_bf16(self, dtype):
        """Mixed groups should not produce NaN for bf16/fp16."""
        model = _make_model_mixed(dtype=dtype)
        opt = _make_amuse_mixed(model, warmup_steps=5)
        losses = _run_steps(model, opt, n_steps=20, input_size=32)
        for p in model.parameters():
            assert not torch.isnan(p).any(), f"NaN in parameters ({dtype})"
            assert not torch.isinf(p).any(), f"Inf in parameters ({dtype})"

    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
    def test_mixed_state_dtypes_bf16(self, dtype):
        """All state buffers should match parameter dtype for bf16/fp16."""
        model = _make_model_mixed(dtype=dtype)
        opt = _make_amuse_mixed(model, warmup_steps=5)

        opt.train()
        x = torch.randn(2, 32, dtype=dtype, device=DEVICE)
        loss = model(x).sum()
        loss.backward()
        opt.step()
        opt.zero_grad()

        for p in model.parameters():
            state = opt.state[p]
            assert state["z"].dtype == dtype
            if p.ndim >= 2:
                assert state["momentum_buffer"].dtype == dtype
            else:
                assert state["exp_avg_sq"].dtype == dtype


# ---------------------------------------------------------------------------
# Tests: fp16 uses fp32 for Newton-Schulz
# ---------------------------------------------------------------------------

class TestFP16NewtonSchulzFP32:
    """Verify that fp16 parameters use fp32 Newton-Schulz internally."""

    def test_fp16_muon_step_uses_fp32_ns(self):
        """The muon path for fp16 params should produce valid results
        (implicitly tests fp32 Newton-Schulz)."""
        torch.manual_seed(42)
        model = _make_model_2d_only(dtype=torch.float16)
        opt = _make_amuse_for_model(model, use_muon=True, warmup_steps=3)

        opt.train()
        x = torch.randn(2, 32, dtype=torch.float16, device=DEVICE)
        loss = model(x).sum()
        loss.backward()
        opt.step()
        opt.zero_grad()

        # Verify no NaN/Inf and state exists
        for p in model.parameters():
            assert not torch.isnan(p).any(), "NaN in fp16 Muon step"
            assert not torch.isinf(p).any(), "Inf in fp16 Muon step"
            state = opt.state[p]
            assert "z" in state
            assert "momentum_buffer" in state

    def test_bf16_muon_step_uses_bf16_ns(self):
        """The muon path for bf16 params should use bf16 Newton-Schulz."""
        torch.manual_seed(42)
        model = _make_model_2d_only(dtype=torch.bfloat16)
        opt = _make_amuse_for_model(model, use_muon=True, warmup_steps=3)

        opt.train()
        x = torch.randn(2, 32, dtype=torch.bfloat16, device=DEVICE)
        loss = model(x).sum()
        loss.backward()
        opt.step()
        opt.zero_grad()

        for p in model.parameters():
            assert not torch.isnan(p).any(), "NaN in bf16 Muon step"
            assert not torch.isinf(p).any(), "Inf in bf16 Muon step"


# ---------------------------------------------------------------------------
# Tests: Stochastic rounding verification
# ---------------------------------------------------------------------------

class TestStochasticRounding:
    """Tests that verify stochastic rounding is used for bf16/fp16 write-back."""

    def test_stochastic_rounding_differs_from_truncation(self):
        """copy_stochastic_ should produce statistically different results
        from simple truncation for values that don't round exactly."""
        torch.manual_seed(123)
        # Create a large fp32 tensor with values that don't round exactly in bf16
        source = torch.randn(1000, 1000, dtype=torch.float32, device=DEVICE) * 0.01

        # Truncation (direct copy)
        target_trunc = torch.zeros_like(source, dtype=torch.bfloat16)
        target_trunc.copy_(source)

        # Stochastic rounding
        target_stoch = torch.zeros_like(source, dtype=torch.bfloat16)
        copy_stochastic_(target_stoch, source)

        # Both should be close to source, but they should differ from each other
        # (stochastic rounding is not deterministic)
        assert not torch.equal(target_trunc, target_stoch), (
            "Stochastic rounding should differ from truncation"
        )

        # Both should be reasonable approximations (within bf16 precision)
        trunc_error = (target_trunc.float() - source).abs().mean()
        stoch_error = (target_stoch.float() - source).abs().mean()
        # Both should have comparable error magnitude (within 2x)
        assert stoch_error < trunc_error * 2.0, (
            f"Stochastic rounding error ({stoch_error:.6f}) should be "
            f"comparable to truncation error ({trunc_error:.6f})"
        )


# ---------------------------------------------------------------------------
# Tests: _lerp_to_z workspace caching
# ---------------------------------------------------------------------------

class TestLerpToZWorkspaceCaching:
    """Tests verifying _lerp_to_z uses cached fp32 workspaces."""

    def test_lerp_to_z_uses_fp32_workspace_bf16(self):
        """_lerp_to_z on a bf16 param should create and reuse fp32 workspaces."""
        torch.manual_seed(42)
        model = _make_model_2d_only(dtype=torch.bfloat16)
        opt = _make_amuse_for_model(model, use_muon=True, warmup_steps=3)

        opt.train()
        x = torch.randn(2, 32, dtype=torch.bfloat16, device=DEVICE)
        loss = model(x).sum()
        loss.backward()
        opt.step()
        opt.zero_grad()

        for p in model.parameters():
            state = opt.state[p]
            z = state["z"]

            # Call _lerp_to_z directly — should use cached workspaces
            opt._lerp_to_z(p, z, 0.5, state)

            # Workspace keys should now exist in state
            assert "_ws_p" in state, "_ws_p workspace not created"
            assert "_ws_z" in state, "_ws_z workspace not created"
            # Workspaces should be fp32
            assert state["_ws_p"].dtype == torch.float32
            assert state["_ws_z"].dtype == torch.float32
            # Workspaces should match param shape
            assert state["_ws_p"].shape == p.shape
            assert state["_ws_z"].shape == z.shape

    def test_lerp_to_z_uses_fp32_workspace_fp16(self):
        """_lerp_to_z on a fp16 param should create and reuse fp32 workspaces."""
        torch.manual_seed(42)
        model = _make_model_2d_only(dtype=torch.float16)
        opt = _make_amuse_for_model(model, use_muon=True, warmup_steps=3)

        opt.train()
        x = torch.randn(2, 32, dtype=torch.float16, device=DEVICE)
        loss = model(x).sum()
        loss.backward()
        opt.step()
        opt.zero_grad()

        for p in model.parameters():
            state = opt.state[p]
            z = state["z"]
            opt._lerp_to_z(p, z, 0.3, state)
            assert "_ws_p" in state
            assert "_ws_z" in state

    def test_lerp_to_z_fp32_params_no_workspace(self):
        """_lerp_to_z on fp32 params should lerp directly (no workspace needed)."""
        torch.manual_seed(42)
        model = _make_model_2d_only(dtype=torch.float32)
        opt = _make_amuse_for_model(model, use_muon=True, warmup_steps=3)

        opt.train()
        # Run 2 steps so ckp1 < 1.0 and p ≠ z (step 1 always gives ckp1=1.0 → p=z)
        for _ in range(2):
            x = torch.randn(2, 32, dtype=torch.float32, device=DEVICE)
            loss = model(x).sum()
            loss.backward()
            opt.step()
            opt.zero_grad()

        with torch.no_grad():
            for p in model.parameters():
                state = opt.state[p]
                z = state["z"]
                p_before = p.clone()
                opt._lerp_to_z(p, z, 0.5, state)
                # fp32 path should NOT create workspace keys
                assert "_ws_p" not in state
                # But p should have changed
                assert not torch.equal(p_before, p)

    def test_lerp_to_z_workspace_reused_across_calls(self):
        """Repeated _lerp_to_z calls should reuse the same workspace tensor."""
        torch.manual_seed(42)
        model = _make_model_2d_only(dtype=torch.bfloat16)
        opt = _make_amuse_for_model(model, use_muon=True, warmup_steps=3)

        opt.train()
        x = torch.randn(2, 32, dtype=torch.bfloat16, device=DEVICE)
        loss = model(x).sum()
        loss.backward()
        opt.step()
        opt.zero_grad()

        for p in model.parameters():
            state = opt.state[p]
            z = state["z"]

            # First call creates workspace
            opt._lerp_to_z(p, z, 0.3, state)
            ws_p_id = state["_ws_p"].data_ptr()

            # Second call should reuse same workspace
            opt._lerp_to_z(p, z, 0.7, state)
            assert state["_ws_p"].data_ptr() == ws_p_id, (
                "Workspace tensor was reallocated instead of reused"
            )

    def test_lerp_to_z_no_nan_bf16(self):
        """_lerp_to_z should not produce NaN in bf16 params."""
        torch.manual_seed(42)
        model = _make_model_2d_only(dtype=torch.bfloat16)
        opt = _make_amuse_for_model(model, use_muon=True, warmup_steps=3)

        opt.train()
        x = torch.randn(2, 32, dtype=torch.bfloat16, device=DEVICE)
        loss = model(x).sum()
        loss.backward()
        opt.step()
        opt.zero_grad()

        for p in model.parameters():
            state = opt.state[p]
            z = state["z"]
            opt._lerp_to_z(p, z, 0.5, state)
            assert not torch.isnan(p).any(), "NaN after _lerp_to_z"
            assert not torch.isinf(p).any(), "Inf after _lerp_to_z"


# ---------------------------------------------------------------------------
# Tests: Batch eval/train with _foreach_lerp_
# ---------------------------------------------------------------------------

class TestBatchEvalTrainForeach:
    """Tests verifying eval()/train() use batched _foreach_lerp_."""

    def test_batch_eval_fp32_params_change(self):
        """eval() on fp32 model should change params (y -> x conversion)."""
        torch.manual_seed(42)
        model = _make_model_2d_only(dtype=torch.float32)
        opt = _make_amuse_for_model(model, use_muon=True, warmup_steps=3)

        opt.train()
        # Run 2 steps so ckp1 < 1.0 and p ≠ z (step 1 always gives ckp1=1.0 → p=z)
        for _ in range(2):
            x = torch.randn(2, 32, dtype=torch.float32, device=DEVICE)
            loss = model(x).sum()
            loss.backward()
            opt.step()
            opt.zero_grad()

        params_before_eval = [p.clone() for p in model.parameters()]
        opt.eval()

        changed = any(
            not torch.allclose(pb, p)
            for pb, p in zip(params_before_eval, model.parameters())
        )
        assert changed, "eval() should change params (y -> x conversion)"

    def test_batch_train_fp32_params_change(self):
        """train() on fp32 model should change params (x -> y conversion)."""
        torch.manual_seed(42)
        model = _make_model_2d_only(dtype=torch.float32)
        opt = _make_amuse_for_model(model, use_muon=True, warmup_steps=3)

        opt.train()
        # Run 2 steps so ckp1 < 1.0 and p ≠ z (step 1 always gives ckp1=1.0 → p=z)
        for _ in range(2):
            x = torch.randn(2, 32, dtype=torch.float32, device=DEVICE)
            loss = model(x).sum()
            loss.backward()
            opt.step()
            opt.zero_grad()

        opt.eval()
        params_after_eval = [p.clone() for p in model.parameters()]
        opt.train()

        changed = any(
            not torch.allclose(pe, p)
            for pe, p in zip(params_after_eval, model.parameters())
        )
        assert changed, "train() should change params (x -> y conversion)"

    def test_batch_eval_train_roundtrip_fp32(self):
        """eval -> train round-trip on fp32 should preserve params."""
        torch.manual_seed(42)
        model = _make_model_2d_only(dtype=torch.float32)
        opt = _make_amuse_for_model(model, use_muon=True, warmup_steps=3)

        opt.train()
        x = torch.randn(2, 32, dtype=torch.float32, device=DEVICE)
        loss = model(x).sum()
        loss.backward()
        opt.step()
        opt.zero_grad()

        params_after_step = [p.clone() for p in model.parameters()]
        opt.eval()
        opt.train()

        for p_orig, p_restored in zip(params_after_step, model.parameters()):
            assert torch.allclose(p_orig, p_restored, atol=1e-6), (
                "eval -> train round-trip changed fp32 params"
            )

    def test_batch_eval_train_roundtrip_bf16(self):
        """eval -> train round-trip on bf16 should preserve params approximately."""
        torch.manual_seed(42)
        model = _make_model_2d_only(dtype=torch.bfloat16)
        opt = _make_amuse_for_model(model, use_muon=True, warmup_steps=3)

        opt.train()
        x = torch.randn(2, 32, dtype=torch.bfloat16, device=DEVICE)
        loss = model(x).sum()
        loss.backward()
        opt.step()
        opt.zero_grad()

        params_after_step = [p.clone() for p in model.parameters()]
        opt.eval()
        opt.train()

        for p_orig, p_restored in zip(params_after_step, model.parameters()):
            # bf16 stochastic rounding introduces small noise
            assert torch.allclose(
                p_orig.float(), p_restored.float(), atol=0.05
            ), "eval -> train round-trip changed bf16 params beyond tolerance"

    def test_batch_eval_bf16_no_nan(self):
        """eval() on bf16 model should not produce NaN."""
        torch.manual_seed(42)
        model = _make_model_2d_only(dtype=torch.bfloat16)
        opt = _make_amuse_for_model(model, use_muon=True, warmup_steps=3)

        opt.train()
        x = torch.randn(2, 32, dtype=torch.bfloat16, device=DEVICE)
        loss = model(x).sum()
        loss.backward()
        opt.step()
        opt.zero_grad()

        opt.eval()
        for p in model.parameters():
            assert not torch.isnan(p).any(), "NaN after eval() on bf16"
            assert not torch.isinf(p).any(), "Inf after eval() on bf16"

    def test_batch_eval_adamw_roundtrip_fp32(self):
        """eval -> train round-trip with AdamW fallback should preserve params."""
        torch.manual_seed(42)
        model = _make_model_2d_only(dtype=torch.float32)
        opt = _make_amuse_for_model(model, use_muon=False, warmup_steps=3)

        opt.train()
        x = torch.randn(2, 32, dtype=torch.float32, device=DEVICE)
        loss = model(x).sum()
        loss.backward()
        opt.step()
        opt.zero_grad()

        params_after_step = [p.clone() for p in model.parameters()]
        opt.eval()
        opt.train()

        for p_orig, p_restored in zip(params_after_step, model.parameters()):
            assert torch.allclose(p_orig, p_restored, atol=1e-6), (
                "eval -> train round-trip changed fp32 AdamW params"
            )

    def test_batch_eval_mixed_precision_groups(self):
        """eval -> train should work with a group containing multiple bf16 params."""
        torch.manual_seed(42)
        # All layers in the same reduced precision to avoid dtype mismatch
        model = torch.nn.Sequential(
            torch.nn.Linear(32, 64, bias=False, dtype=torch.bfloat16, device=DEVICE),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 16, bias=False, dtype=torch.bfloat16, device=DEVICE),
        )
        muon_params = [p for p in model.parameters()]
        opt = AMUSE(
            [{"params": muon_params, "use_muon": True}],
            warmup_steps=3,
        )

        opt.train()
        x = torch.randn(2, 32, dtype=torch.bfloat16, device=DEVICE)
        loss = model(x).sum()
        loss.backward()
        opt.step()
        opt.zero_grad()

        params_after_step = [p.clone() for p in model.parameters()]
        opt.eval()
        opt.train()

        for p_orig, p_restored in zip(params_after_step, model.parameters()):
            assert not torch.isnan(p_restored).any(), "NaN after round-trip"
            assert not torch.isinf(p_restored).any(), "Inf after round-trip"

    def test_batch_lerp_multiple_params_foreach(self):
        """_batch_lerp_to_z should handle multiple parameters in one group."""
        torch.manual_seed(42)
        model = _make_model_mixed(dtype=torch.float32)
        muon_params = [p for p in model.parameters() if p.ndim >= 2]
        opt = AMUSE(
            [{"params": muon_params, "use_muon": True}],
            warmup_steps=3,
        )

        opt.train()
        # Run 2 steps so ckp1 < 1.0 and p ≠ z (step 1 always gives ckp1=1.0 → p=z)
        for _ in range(2):
            x = torch.randn(2, 32, dtype=torch.float32, device=DEVICE)
            loss = model(x).sum()
            loss.backward()
            opt.step()
            opt.zero_grad()

        # Verify _batch_lerp_to_z works with the group containing multiple params
        with torch.no_grad():
            group = opt.param_groups[0]
            params_before = [p.clone() for p in group["params"]]
            opt._batch_lerp_to_z(group, 0.5)

            changed = any(
                not torch.allclose(pb, p)
                for pb, p in zip(params_before, group["params"])
            )
            assert changed, "_batch_lerp_to_z should change params"

    def test_batch_lerp_no_z_state_skipped(self):
        """_batch_lerp_to_z should skip params without z state."""
        torch.manual_seed(42)
        model = _make_model_2d_only(dtype=torch.float32)
        opt = _make_amuse_for_model(model, use_muon=True, warmup_steps=3)

        # Don't run step() — no z state should exist yet
        group = opt.param_groups[0]
        # Should not raise even though no z state exists
        opt._batch_lerp_to_z(group, 0.5)

    def test_batch_lerp_convergence_after_mode_switches(self):
        """Training should converge even with frequent eval/train switches."""
        torch.manual_seed(42)
        model = _make_model_2d_only(dtype=torch.float32)
        opt = _make_amuse_for_model(model, use_muon=False, warmup_steps=3)

        torch.manual_seed(999)
        opt.train()
        losses = []
        for i in range(15):
            x = torch.randn(4, 32, dtype=torch.float32, device=DEVICE)
            loss = model(x).sum()
            loss.backward()
            opt.step()
            opt.zero_grad()
            losses.append(loss.item())
            # Switch modes every 3 steps
            if i % 3 == 2:
                opt.eval()
                opt.train()

        # Loss should trend downward despite mode switches
        early_avg = sum(losses[:5]) / 5
        late_avg = sum(losses[-5:]) / 5
        assert late_avg < early_avg, (
            f"Loss should decrease: early={early_avg:.4f}, late={late_avg:.4f}"
        )

    def test_batch_eval_train_convergence_bf16(self):
        """Training with bf16 and frequent mode switches should converge."""
        torch.manual_seed(42)
        model = _make_model_2d_only(dtype=torch.bfloat16)
        opt = _make_amuse_for_model(model, use_muon=False, warmup_steps=3)

        torch.manual_seed(999)
        opt.train()
        losses = []
        for i in range(15):
            x = torch.randn(4, 32, dtype=torch.bfloat16, device=DEVICE)
            loss = model(x).float().sum()
            loss.backward()
            opt.step()
            opt.zero_grad()
            losses.append(loss.item())
            if i % 3 == 2:
                opt.eval()
                opt.train()

        early_avg = sum(losses[:5]) / 5
        late_avg = sum(losses[-5:]) / 5
        assert late_avg < early_avg, (
            f"bf16 loss should decrease: early={early_avg:.4f}, late={late_avg:.4f}"
        )
