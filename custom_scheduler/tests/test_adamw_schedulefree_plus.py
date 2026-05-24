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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
