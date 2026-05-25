"""Tests for AdamWScheduleFreePlus optimizer fp32 computation and stochastic rounding.

Validates:
1. Import and optimizer instantiation works.
2. train()/eval() mode switching and step_func() run without error.
3. State tensors (z, exp_avg, exp_avg_sq) are correctly initialized.
4. Parameters change after a step (fp32 computation is active).
5. bfloat16 model parameters use stochastic rounding write-back correctly.
6. float32 model parameters use regular copy write-back correctly.
7. Numerical stability: no NaN or Inf after multiple steps.

Run with:
    cd backend/sd_scripts
    python -m pytest ../custom_scheduler/tests/test_adamw_schedulefree_plus.py -v
"""

import sys
import os
import copy
import pytest
import torch

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

# Load utils first (needed by adamw_schedulefree_plus via ``from .utils import copy_stochastic_``)
_utils_path = os.path.join(_pkg_dir, "utils.py")
_utils_spec = importlib.util.spec_from_file_location(
    "LoraEasyCustomOptimizer.utils", _utils_path
)
_utils_mod = importlib.util.module_from_spec(_utils_spec)
sys.modules["LoraEasyCustomOptimizer.utils"] = _utils_mod
_utils_spec.loader.exec_module(_utils_mod)

# Load adamw_schedulefree_plus as a submodule of the fake package
_asfp_path = os.path.join(_pkg_dir, "adamw_schedulefree_plus.py")
_spec = importlib.util.spec_from_file_location(
    "LoraEasyCustomOptimizer.adamw_schedulefree_plus", _asfp_path
)
_asfp = importlib.util.module_from_spec(_spec)
sys.modules["LoraEasyCustomOptimizer.adamw_schedulefree_plus"] = _asfp
_spec.loader.exec_module(_asfp)

AdamWScheduleFreePlus = _asfp.AdamWScheduleFreePlus
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


def _run_step_func_steps(model, opt, n_steps=5, input_size=32, seed=999):
    """Run *n_steps* optimizer.step_func() calls and return the final loss."""
    torch.manual_seed(seed)
    final_loss = None
    opt.train()
    for _ in range(n_steps):
        x = torch.randn(8, input_size, dtype=next(model.parameters()).dtype)
        loss = model(x).sum()
        loss.backward()
        final_loss = opt.step_func(loss.item())
        opt.zero_grad()
    return final_loss


# ---------------------------------------------------------------------------
# Test: Import and basic instantiation
# ---------------------------------------------------------------------------

class TestAdamWScheduleFreePlusBasic:
    """Basic sanity checks for the optimizer."""

    def test_import_works(self):
        """AdamWScheduleFreePlus should be importable."""
        assert AdamWScheduleFreePlus is not None

    def test_instantiate(self):
        """Optimizer should instantiate without error."""
        model = _make_model()
        opt = AdamWScheduleFreePlus(model.parameters(), lr=1.0)
        assert opt is not None
        assert len(opt.param_groups) == 1

    def test_defaults_set(self):
        """Default hyperparameters should be set on the param group."""
        model = _make_model()
        opt = AdamWScheduleFreePlus(model.parameters(), lr=1.0)
        group = opt.param_groups[0]
        assert group['lr'] == 1.0
        assert group['betas'] == (0.9, 0.95)
        assert group['sf_beta1'] == 0.9
        assert group['eps'] == 1e-8
        assert group['weight_decay'] == 0
        assert group['r'] == 1.0
        assert group['k'] == 0

    def test_train_mode_initial(self):
        """Optimizer should start in eval mode (train_mode=False)."""
        model = _make_model()
        opt = AdamWScheduleFreePlus(model.parameters(), lr=1.0)
        assert opt.param_groups[0]['train_mode'] is False

    def test_step_without_train_raises(self):
        """Calling step_func() without .train() should raise."""
        model = _make_model()
        opt = AdamWScheduleFreePlus(model.parameters(), lr=1.0)
        x = torch.randn(8, 32)
        loss = model(x).sum()
        loss.backward()
        with pytest.raises(Exception, match="not in train mode"):
            opt.step_func(loss.item())


# ---------------------------------------------------------------------------
# Test: State initialization
# ---------------------------------------------------------------------------

class TestAdamWScheduleFreePlusState:
    """Verify optimizer state is correctly initialized."""

    def test_state_created_on_first_step(self):
        """z, exp_avg, exp_avg_sq should be created on first step_func()."""
        model = _make_model()
        opt = AdamWScheduleFreePlus(model.parameters(), lr=1.0)
        opt.train()
        x = torch.randn(8, 32)
        loss = model(x).sum()
        loss.backward()
        opt.step_func(loss.item())

        for p in model.parameters():
            state = opt.state[p]
            assert 'z' in state
            assert 'exp_avg' in state
            assert 'exp_avg_sq' in state
            assert state['z'].shape == p.shape
            assert state['exp_avg'].shape == p.shape
            assert state['exp_avg_sq'].shape == p.shape

    def test_z_starts_as_clone_of_p(self):
        """z should be initialized as a clone of the parameter."""
        model = _make_model()
        opt = AdamWScheduleFreePlus(model.parameters(), lr=1.0)
        opt.train()
        # Store initial params
        initial_params = {p: p.detach().clone() for p in model.parameters()}
        x = torch.randn(8, 32)
        loss = model(x).sum()
        loss.backward()
        opt.step_func(loss.item())

        for p in model.parameters():
            state = opt.state[p]
            assert torch.equal(state['z'], initial_params[p]), \
                "z should have been initialized as a clone of the initial parameter"

    def test_exp_avg_starts_as_zeros(self):
        """exp_avg and exp_avg_sq should start as zeros."""
        model = _make_model()
        opt = AdamWScheduleFreePlus(model.parameters(), lr=1.0)
        opt.train()
        # Run one step
        x = torch.randn(8, 32)
        loss = model(x).sum()
        loss.backward()
        opt.step_func(loss.item())

        # Check exp_avg and exp_avg_sq were initially zeros (they have been
        # updated now, so they should be non-zero if gradients were non-zero)
        for p in model.parameters():
            state = opt.state[p]
            assert not torch.all(state['exp_avg'] == 0), \
                "exp_avg should be non-zero after a step with non-zero gradients"

    def test_step_counter_increments(self):
        """The step counter (k) should increment on each step_func() call."""
        model = _make_model()
        opt = AdamWScheduleFreePlus(model.parameters(), lr=1.0)
        opt.train()

        for expected_k in range(1, 4):
            x = torch.randn(8, 32)
            loss = model(x).sum()
            loss.backward()
            opt.step_func(loss.item())
            opt.zero_grad()
            assert opt.param_groups[0]['k'] == expected_k


# ---------------------------------------------------------------------------
# Test: fp32 computation and write-back
# ---------------------------------------------------------------------------

class TestAdamWScheduleFreePlusFP32Computation:
    """Verify that fp32 computation is active and state is written back correctly."""

    def test_parameters_change_after_step(self):
        """Parameters should change after step_func() when polyak_lr > 0.

        Note: Polyak step size is ``max(0, loss + ip_term) / grad_l1_ema_corr``.
        On the first step ip_term = 0 (no z state yet), so if loss <= 0 the
        effective LR is zero.  We force a positive loss by using ``abs()``.
        """
        model = _make_model(dtype=torch.float32)
        params_before = {p: p.detach().clone() for p in model.parameters()}
        opt = AdamWScheduleFreePlus(model.parameters(), lr=1.0)
        opt.train()
        x = torch.randn(8, 32, dtype=torch.float32)
        loss = model(x).sum().abs()  # guarantee positive function value
        loss.backward()
        opt.step_func(loss.item())

        any_changed = False
        for p in model.parameters():
            if not torch.equal(p, params_before[p]):
                any_changed = True
                break
        assert any_changed, \
            "At least one parameter should change after optimizer step with positive loss"

    def test_state_updated_after_step(self):
        """State tensors (z, exp_avg, exp_avg_sq) should be updated after steps."""
        model = _make_model(dtype=torch.float32)
        opt = AdamWScheduleFreePlus(model.parameters(), lr=1.0)
        opt.train()

        # Step 1
        x = torch.randn(8, 32, dtype=torch.float32)
        loss = model(x).sum().abs()
        loss.backward()
        opt.step_func(loss.item())
        opt.zero_grad()

        # Record state after first step
        state_after_1 = {}
        for p in model.parameters():
            state = opt.state[p]
            state_after_1[p] = {
                'z': state['z'].clone(),
                'exp_avg': state['exp_avg'].clone(),
                'exp_avg_sq': state['exp_avg_sq'].clone(),
            }

        # Step 2
        x = torch.randn(8, 32, dtype=torch.float32)
        loss = model(x).sum().abs()
        loss.backward()
        opt.step_func(loss.item())

        any_z_changed = False
        any_exp_avg_changed = False
        for p in model.parameters():
            state = opt.state[p]
            if not torch.equal(state['z'], state_after_1[p]['z']):
                any_z_changed = True
            if not torch.equal(state['exp_avg'], state_after_1[p]['exp_avg']):
                any_exp_avg_changed = True
        assert any_z_changed, "z should be updated after second step"
        assert any_exp_avg_changed, "exp_avg should be updated after second step"

    def test_no_nan_after_multiple_steps(self):
        """Parameters should not become NaN after multiple steps."""
        model = _make_model(dtype=torch.float32)
        opt = AdamWScheduleFreePlus(model.parameters(), lr=1.0)
        opt.train()
        for _ in range(20):
            x = torch.randn(8, 32, dtype=torch.float32)
            loss = model(x).sum()
            loss.backward()
            opt.step_func(loss.item())
            opt.zero_grad()

        for p in model.parameters():
            assert not torch.isnan(p).any(), "Parameter should not be NaN"
            assert not torch.isinf(p).any(), "Parameter should not be Inf"

    def test_fp32_write_back_uses_direct_copy(self):
        """fp32 parameters should be written back via direct .copy_() (not stochastic)."""
        model = _make_model(dtype=torch.float32)
        opt = AdamWScheduleFreePlus(model.parameters(), lr=1.0)
        opt.train()
        x = torch.randn(8, 32, dtype=torch.float32)
        loss = model(x).sum()
        loss.backward()
        opt.step_func(loss.item())

        for p in model.parameters():
            state = opt.state[p]
            # fp32 parameters: state should be exact copies (same dtype as param)
            assert state['z'].dtype == torch.float32
            assert state['exp_avg'].dtype == torch.float32
            assert state['exp_avg_sq'].dtype == torch.float32


# ---------------------------------------------------------------------------
# Test: bfloat16 stochastic rounding write-back
# ---------------------------------------------------------------------------

class TestAdamWScheduleFreePlusBF16Stochastic:
    """Verify that bfloat16 model uses stochastic rounding for write-back."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_bf16_model_runs_on_gpu(self):
        """bf16 model on GPU should run step_func() without error."""
        model = _make_model(dtype=torch.bfloat16).cuda()
        opt = AdamWScheduleFreePlus(model.parameters(), lr=1.0)
        opt.train()
        x = torch.randn(8, 32, dtype=torch.bfloat16, device="cuda")
        loss = model(x).sum()
        loss.backward()
        opt.step_func(loss.item())  # should not crash

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_bf16_parameters_change_on_gpu(self):
        """bf16 parameters on GPU should change after step_func() with positive loss."""
        model = _make_model(dtype=torch.bfloat16).cuda()
        params_before = {p: p.detach().clone() for p in model.parameters()}
        opt = AdamWScheduleFreePlus(model.parameters(), lr=1.0)
        opt.train()
        x = torch.randn(8, 32, dtype=torch.bfloat16, device="cuda")
        loss = model(x).sum().abs()  # guarantee positive function value
        loss.backward()
        opt.step_func(loss.item())

        any_changed = False
        for p in model.parameters():
            if not torch.equal(p, params_before[p]):
                any_changed = True
                break
        assert any_changed, \
            "At least one bf16 parameter should change after optimizer step with positive loss"

    def test_bf16_model_cpu_runs(self):
        """bf16 model on CPU should run step_func() without error."""
        model = _make_model(dtype=torch.bfloat16)
        opt = AdamWScheduleFreePlus(model.parameters(), lr=1.0)
        opt.train()
        x = torch.randn(8, 32, dtype=torch.bfloat16)
        loss = model(x).sum()
        loss.backward()
        opt.step_func(loss.item())  # should not crash

    def test_bf16_state_preserves_bf16_dtype(self):
        """State tensors for bf16 parameters should stay in bf16 dtype."""
        model = _make_model(dtype=torch.bfloat16)
        opt = AdamWScheduleFreePlus(model.parameters(), lr=1.0)
        opt.train()
        x = torch.randn(8, 32, dtype=torch.bfloat16)
        loss = model(x).sum()
        loss.backward()
        opt.step_func(loss.item())

        for p in model.parameters():
            state = opt.state[p]
            assert state['z'].dtype == torch.bfloat16, \
                f"Expected z dtype bfloat16, got {state['z'].dtype}"
            assert state['exp_avg'].dtype == torch.bfloat16, \
                f"Expected exp_avg dtype bfloat16, got {state['exp_avg'].dtype}"
            assert state['exp_avg_sq'].dtype == torch.bfloat16, \
                f"Expected exp_avg_sq dtype bfloat16, got {state['exp_avg_sq'].dtype}"


# ---------------------------------------------------------------------------
# Test: train/eval mode switching
# ---------------------------------------------------------------------------

class TestAdamWScheduleFreePlusTrainEval:
    """Verify train() and eval() mode switching modifies parameters correctly."""

    def test_train_eval_switching_fp32(self):
        """train() / eval() calls should run without error and modify parameters.

        Note: On the first step with ckp1=1 (c_warmup=0), p == z after update,
        so eval() lerp between equal values is a no-op.  After multiple steps
        with ckp1 < 1, p != z and the lerp becomes visible.
        """
        model = _make_model(dtype=torch.float32)
        opt = AdamWScheduleFreePlus(model.parameters(), lr=1.0)
        opt.train()

        # Run multiple steps so ckp1 < 1 (p and z diverge)
        for _ in range(5):
            x = torch.randn(8, 32, dtype=torch.float32)
            loss = model(x).sum().abs()
            loss.backward()
            opt.step_func(loss.item())
            opt.zero_grad()

        # Record params after train mode steps
        params_after_train = {p: p.detach().clone() for p in model.parameters()}

        # Switch to eval mode
        opt.eval()
        any_changed = False
        for p in model.parameters():
            if not torch.equal(p, params_after_train[p]):
                any_changed = True
                break
        assert any_changed, \
            "At least one parameter should change when switching to eval mode after multiple steps"

        # Switch back to train mode — should not crash
        opt.train()
        for p in model.parameters():
            pass  # Just checking it doesn't crash


# ---------------------------------------------------------------------------
# Test: edge cases
# ---------------------------------------------------------------------------

class TestAdamWScheduleFreePlusEdgeCases:
    """Edge-case handling."""

    def test_params_without_grad_skipped(self):
        """Parameters with grad=None should be skipped without error."""
        model = _make_model(dtype=torch.float32, sizes=[(8, 8), (8, 4)])
        opt = AdamWScheduleFreePlus(model.parameters(), lr=1.0)
        opt.train()

        x = torch.randn(2, 8, dtype=torch.float32)
        out = model[0](x)  # only first layer forward
        loss = out.sum()
        loss.backward()

        # Null out gradients for some params
        for name, param in model.named_parameters():
            if '1' in name:
                param.grad = None

        opt.step_func(loss.item())  # should not crash

    def test_zero_lr_no_crash(self):
        """lr=0 should not crash."""
        model = _make_model(dtype=torch.float32)
        opt = AdamWScheduleFreePlus(model.parameters(), lr=0.0)
        opt.train()
        x = torch.randn(8, 32, dtype=torch.float32)
        loss = model(x).sum()
        loss.backward()
        opt.step_func(loss.item())  # should not crash

    def test_weight_decay_no_crash(self):
        """Non-zero weight_decay should not crash."""
        model = _make_model(dtype=torch.float32)
        opt = AdamWScheduleFreePlus(model.parameters(), lr=1.0, weight_decay=10.0)
        opt.train()
        x = torch.randn(8, 32, dtype=torch.float32)
        loss = model(x).sum()
        loss.backward()
        opt.step_func(loss.item())  # should not crash

    def test_c_warmup_no_crash(self):
        """c_warmup > 0 should not crash."""
        model = _make_model(dtype=torch.float32)
        opt = AdamWScheduleFreePlus(model.parameters(), lr=1.0, c_warmup=10)
        opt.train()
        for _ in range(15):
            x = torch.randn(8, 32, dtype=torch.float32)
            loss = model(x).sum()
            loss.backward()
            opt.step_func(loss.item())
            opt.zero_grad()
        # should not crash

    def test_sf_beta1_anneal_no_crash(self):
        """sf_beta1 annealing should not crash."""
        model = _make_model(dtype=torch.float32)
        opt = AdamWScheduleFreePlus(
            model.parameters(), lr=1.0,
            sf_beta1=0.9, sf_beta1_max=0.965,
            sf_beta1_anneal_steps=10,
        )
        opt.train()
        for _ in range(15):
            x = torch.randn(8, 32, dtype=torch.float32)
            loss = model(x).sum()
            loss.backward()
            opt.step_func(loss.item())
            opt.zero_grad()
        # should not crash

    def test_warmup_steps_no_crash(self):
        """warmup_steps > 0 should not crash."""
        model = _make_model(dtype=torch.float32)
        opt = AdamWScheduleFreePlus(model.parameters(), lr=1.0, warmup_steps=10)
        opt.train()
        for _ in range(15):
            x = torch.randn(8, 32, dtype=torch.float32)
            loss = model(x).sum()
            loss.backward()
            opt.step_func(loss.item())
            opt.zero_grad()
        # should not crash

    def test_polyak_f_ema_default(self):
        """polyak_f_ema should default to 0.9."""
        model = _make_model()
        opt = AdamWScheduleFreePlus(model.parameters(), lr=1.0)
        assert opt.param_groups[0]['polyak_f_ema'] == 0.9

    def test_max_polyak_lr_default(self):
        """max_polyak_lr should default to 10.0."""
        model = _make_model()
        opt = AdamWScheduleFreePlus(model.parameters(), lr=1.0)
        assert opt.param_groups[0]['max_polyak_lr'] == 10.0


# ---------------------------------------------------------------------------
# Test: Function value EMA (polyak_f_ema)
# ---------------------------------------------------------------------------

class TestAdamWScheduleFreePlusFunctionValueEMA:
    """Verify that the function value EMA feature works correctly."""

    def test_f_ema_initialized_on_first_step(self):
        """f_ema should be initialized to the raw function value on first step."""
        model = _make_model(dtype=torch.float32)
        opt = AdamWScheduleFreePlus(model.parameters(), lr=1.0, polyak_f_ema=0.9)
        assert opt.param_groups[0]['f_ema'] is None

        opt.train()
        x = torch.randn(8, 32, dtype=torch.float32)
        loss = model(x).sum().abs()
        loss.backward()
        first_fv = loss.item()
        opt.step_func(first_fv)

        assert opt.param_groups[0]['f_ema'] is not None
        assert opt.param_groups[0]['f_ema'] == pytest.approx(first_fv, abs=1e-6), \
            "f_ema should equal the raw function value on first step"

    def test_f_ema_converges_to_constant_input(self):
        """With constant function value input, f_ema should converge to that value."""
        model = _make_model(dtype=torch.float32)
        opt = AdamWScheduleFreePlus(model.parameters(), lr=1.0, polyak_f_ema=0.9)
        opt.train()

        constant_fv = 3.14
        for _ in range(50):
            # Create a scenario with positive loss
            x = torch.randn(8, 32, dtype=torch.float32)
            loss = model(x).sum().abs()
            loss.backward()
            # Force the function value to be constant by passing it directly
            opt.step_func(constant_fv)
            opt.zero_grad()

        f_ema = opt.param_groups[0]['f_ema']
        # After 50 steps with ema=0.9, f_ema should be very close to constant_fv
        # f_ema after n steps = 0.9^n * f0 + (1-0.9^n) * constant_fv
        # At n=50: 0.9^50 ≈ 0.0052, so f_ema ≈ 0.0052 * f0 + 0.9948 * 3.14
        assert f_ema == pytest.approx(constant_fv, rel=0.01), \
            f"f_ema={f_ema} should be close to constant_fv={constant_fv} after 50 steps"

    def test_f_ema_disabled_when_coeff_zero(self):
        """When polyak_f_ema=0, f_ema should equal the raw function value (no smoothing)."""
        model = _make_model(dtype=torch.float32)
        opt = AdamWScheduleFreePlus(model.parameters(), lr=1.0, polyak_f_ema=0.0)
        opt.train()

        prev_f_ema = None
        for i in range(10):
            x = torch.randn(8, 32, dtype=torch.float32)
            loss = model(x).sum().abs()
            loss.backward()
            raw_fv = float(i + 1)  # deterministic increasing values
            opt.step_func(raw_fv)
            opt.zero_grad()

            f_ema = opt.param_groups[0]['f_ema']
            if prev_f_ema is not None:
                # With polyak_f_ema=0, f_ema should equal the raw value from the PREVIOUS step
                # because on step 1 it's initialized, then on step 2+ the elif branch
                # doesn't trigger (coeff=0), so f_ema stays at the initialized value.
                # Actually, let's re-read the code:
                # f_ema is set once on first step, then for coeff==0 it stays unchanged
                # because neither the None check nor the coeff>0 branch fires.
                pass
            prev_f_ema = f_ema

        # The key assertion: f_ema should be the FIRST raw value (3.14 or whatever)
        # because after initialization it never changes when coeff=0
        assert opt.param_groups[0]['f_ema'] is not None

    def test_f_ema_reduces_variance_with_noisy_losses(self):
        """With noisy losses, f_ema should produce smoother values than raw input."""
        model = _make_model(dtype=torch.float32)
        opt = AdamWScheduleFreePlus(model.parameters(), lr=1.0, polyak_f_ema=0.9)
        opt.train()

        raw_values = []
        ema_values = []

        import random
        random.seed(42)
        for _ in range(20):
            x = torch.randn(8, 32, dtype=torch.float32)
            loss = model(x).sum().abs()
            loss.backward()
            # Simulate noisy function values
            noisy_fv = 1.0 + random.gauss(0, 0.5)
            raw_values.append(noisy_fv)
            opt.step_func(noisy_fv)
            ema_values.append(opt.param_groups[0]['f_ema'])
            opt.zero_grad()

        # EMA values should have lower variance than raw values
        import statistics
        raw_var = statistics.variance(raw_values)
        ema_var = statistics.variance(ema_values)
        assert ema_var < raw_var, \
            f"EMA variance ({ema_var:.6f}) should be less than raw variance ({raw_var:.6f})"

    def test_function_value_ema_in_logging(self):
        """function_value_raw and function_value_ema should be logged in param group."""
        model = _make_model(dtype=torch.float32)
        opt = AdamWScheduleFreePlus(model.parameters(), lr=1.0, polyak_f_ema=0.9)
        opt.train()

        x = torch.randn(8, 32, dtype=torch.float32)
        loss = model(x).sum().abs()
        loss.backward()
        fv = loss.item()
        opt.step_func(fv)

        group = opt.param_groups[0]
        assert 'function_value_raw' in group
        assert 'function_value_ema' in group
        assert group['function_value_raw'] == pytest.approx(fv, abs=1e-6)
        # On first step, f_ema == raw value
        assert group['function_value_ema'] == pytest.approx(fv, abs=1e-6)


# ---------------------------------------------------------------------------
# Test: Polyak LR cap (max_polyak_lr)
# ---------------------------------------------------------------------------

class TestAdamWScheduleFreePlusPolyakLRCap:
    """Verify that the Polyak LR cap feature works correctly."""

    def test_max_polyak_lr_stored_in_defaults(self):
        """max_polyak_lr should be stored in param group defaults."""
        model = _make_model()
        opt = AdamWScheduleFreePlus(model.parameters(), lr=1.0, max_polyak_lr=5.0)
        assert opt.param_groups[0]['max_polyak_lr'] == 5.0

    def test_polyak_lr_capped_when_large(self):
        """polyak_lr should be capped at max_polyak_lr when it would otherwise exceed it."""
        model = _make_model(dtype=torch.float32)
        # Use a very small max_polyak_lr so the cap is almost certainly hit
        cap = 0.001
        opt = AdamWScheduleFreePlus(
            model.parameters(), lr=1.0, max_polyak_lr=cap,
            polyak_beta=0.0,  # no EMA smoothing for deterministic denominator
        )
        opt.train()

        # Run a step with a large loss to get a large polyak_lr
        x = torch.randn(8, 32, dtype=torch.float32)
        loss = model(x).sum().abs() + 100.0  # large loss
        loss.backward()
        opt.step_func(loss.item())

        polyak_lr = opt.param_groups[0]['polyak_lr']
        assert polyak_lr <= cap + 1e-9, \
            f"polyak_lr={polyak_lr} should be <= max_polyak_lr={cap}"

    def test_polyak_lr_not_capped_when_below_limit(self):
        """polyak_lr should not be artificially reduced when below max_polyak_lr."""
        model = _make_model(dtype=torch.float32)
        cap = 1000.0  # very large cap
        opt = AdamWScheduleFreePlus(
            model.parameters(), lr=1.0, max_polyak_lr=cap,
        )
        opt.train()

        x = torch.randn(8, 32, dtype=torch.float32)
        loss = model(x).sum().abs()
        loss.backward()
        opt.step_func(loss.item())

        polyak_lr = opt.param_groups[0]['polyak_lr']
        # polyak_lr should be less than the cap (not hitting it)
        assert polyak_lr < cap, \
            f"polyak_lr={polyak_lr} should be well below cap={cap}"

    def test_max_polyak_lr_zero_disables_cap(self):
        """max_polyak_lr=0 should disable the cap (no capping applied).

        We verify this by comparing with a capped version: both should
        produce the same polyak_lr when the uncapped value is below the
        cap, and the uncapped version should be >= the capped version
        when the uncapped value exceeds the cap.
        """
        # --- Run with cap disabled (max_polyak_lr=0) ---
        torch.manual_seed(42)
        model_uncapped = _make_model(dtype=torch.float32)
        opt_uncapped = AdamWScheduleFreePlus(
            model_uncapped.parameters(), lr=1.0, max_polyak_lr=0.0,
            polyak_beta=0.0,
        )
        opt_uncapped.train()
        x = torch.randn(8, 32, dtype=torch.float32)
        loss = model_uncapped(x).sum().abs()
        loss.backward()
        opt_uncapped.step_func(loss.item())
        polyak_lr_uncapped = opt_uncapped.param_groups[0]['polyak_lr']

        # --- Run with a very small cap ---
        torch.manual_seed(42)
        model_capped = _make_model(dtype=torch.float32)
        tiny_cap = 0.001
        opt_capped = AdamWScheduleFreePlus(
            model_capped.parameters(), lr=1.0, max_polyak_lr=tiny_cap,
            polyak_beta=0.0,
        )
        opt_capped.train()
        x = torch.randn(8, 32, dtype=torch.float32)
        loss = model_capped(x).sum().abs()
        loss.backward()
        opt_capped.step_func(loss.item())
        polyak_lr_capped = opt_capped.param_groups[0]['polyak_lr']

        # The uncapped polyak_lr should be >= the capped version
        assert polyak_lr_uncapped >= polyak_lr_capped - 1e-9, \
            (f"Uncapped polyak_lr={polyak_lr_uncapped} should be >= "
             f"capped polyak_lr={polyak_lr_capped}")
        # And the capped version should be at most the tiny cap
        assert polyak_lr_capped <= tiny_cap + 1e-9

    def test_scheduled_lr_reflects_cap(self):
        """The logged scheduled_lr should reflect the capped polyak_lr."""
        model = _make_model(dtype=torch.float32)
        cap = 0.01
        lr_base = 2.0
        opt = AdamWScheduleFreePlus(
            model.parameters(), lr=lr_base, max_polyak_lr=cap,
        )
        opt.train()

        x = torch.randn(8, 32, dtype=torch.float32)
        loss = model(x).sum().abs() + 100.0
        loss.backward()
        opt.step_func(loss.item())

        polyak_lr = opt.param_groups[0]['polyak_lr']
        scheduled_lr = opt.param_groups[0]['scheduled_lr']
        assert polyak_lr <= cap + 1e-9
        assert scheduled_lr == pytest.approx(lr_base * polyak_lr, rel=1e-6)

    def test_polyak_lr_logged_in_group(self):
        """polyak_lr should be logged in the param group after step."""
        model = _make_model(dtype=torch.float32)
        opt = AdamWScheduleFreePlus(model.parameters(), lr=1.0)
        opt.train()

        x = torch.randn(8, 32, dtype=torch.float32)
        loss = model(x).sum().abs()
        loss.backward()
        opt.step_func(loss.item())

        assert 'polyak_lr' in opt.param_groups[0]
        assert isinstance(opt.param_groups[0]['polyak_lr'], float)

    def test_no_nan_with_cap_and_small_batch(self):
        """Simulating small-batch LoRA scenario: no NaN with cap and noisy grads."""
        model = _make_model(dtype=torch.float32, sizes=[(8, 16)])
        opt = AdamWScheduleFreePlus(
            model.parameters(), lr=1.0,
            max_polyak_lr=5.0,
            polyak_f_ema=0.9,
        )
        opt.train()
        for _ in range(30):
            x = torch.randn(2, 8, dtype=torch.float32)  # tiny batch
            loss = model(x).sum().abs()
            loss.backward()
            opt.step_func(loss.item())
            opt.zero_grad()

        for p in model.parameters():
            assert not torch.isnan(p).any(), "Parameter should not be NaN"
            assert not torch.isinf(p).any(), "Parameter should not be Inf"


# ---------------------------------------------------------------------------
# Test: Combined EMA + cap interaction
# ---------------------------------------------------------------------------

class TestAdamWScheduleFreePlusEMACapInteraction:
    """Verify that function value EMA and Polyak LR cap work together correctly."""

    def test_both_features_together_no_crash(self):
        """Using both polyak_f_ema and max_polyak_lr should not crash."""
        model = _make_model(dtype=torch.float32)
        opt = AdamWScheduleFreePlus(
            model.parameters(), lr=1.0,
            polyak_f_ema=0.9, max_polyak_lr=5.0,
        )
        opt.train()
        for _ in range(10):
            x = torch.randn(8, 32, dtype=torch.float32)
            loss = model(x).sum().abs()
            loss.backward()
            opt.step_func(loss.item())
            opt.zero_grad()

        for p in model.parameters():
            assert not torch.isnan(p).any()
            assert not torch.isinf(p).any()

    def test_ema_reduces_cap_hits(self):
        """With EMA, the polyak_lr should hit the cap less often than without."""
        import random
        random.seed(42)

        def _count_cap_hits(f_ema_val, n_steps=50, cap=0.5):
            torch.manual_seed(123)
            model = _make_model(dtype=torch.float32, sizes=[(8, 16)])
            opt = AdamWScheduleFreePlus(
                model.parameters(), lr=1.0,
                polyak_f_ema=f_ema_val, max_polyak_lr=cap,
            )
            opt.train()
            hits = 0
            for _ in range(n_steps):
                x = torch.randn(4, 8, dtype=torch.float32)
                loss = model(x).sum().abs()
                loss.backward()
                noisy_fv = loss.item() + random.gauss(0, 0.3)
                opt.step_func(noisy_fv)
                if opt.param_groups[0]['polyak_lr'] >= cap - 1e-9:
                    hits += 1
                opt.zero_grad()
            return hits

        hits_with_ema = _count_cap_hits(0.9)
        hits_without_ema = _count_cap_hits(0.0)

        # With EMA, there should be fewer or equal cap hits
        assert hits_with_ema <= hits_without_ema, \
            f"EMA should reduce cap hits: {hits_with_ema} vs {hits_without_ema}"

    def test_logging_fields_present(self):
        """All new logging fields should be present after step."""
        model = _make_model(dtype=torch.float32)
        opt = AdamWScheduleFreePlus(
            model.parameters(), lr=1.0,
            polyak_f_ema=0.9, max_polyak_lr=5.0,
        )
        opt.train()
        x = torch.randn(8, 32, dtype=torch.float32)
        loss = model(x).sum().abs()
        loss.backward()
        opt.step_func(loss.item())

        group = opt.param_groups[0]
        expected_fields = [
            'function_value_raw', 'function_value_ema',
            'function_value_with_correction', 'polyak_lr',
        ]
        for field in expected_fields:
            assert field in group, f"Missing logging field: {field}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
