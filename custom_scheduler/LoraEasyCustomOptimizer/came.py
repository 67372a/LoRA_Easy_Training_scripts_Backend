# Authored originally by: https://github.com/kozistr
# Source: https://github.com/kozistr/pytorch_optimizer/blob/main/pytorch_optimizer/optimizer/came.py
# With stochastic rounding added per https://github.com/neggles/neurosis/blob/main/src/neurosis/optimizers/came.py

import math
from typing import Tuple

import torch

from pytorch_optimizer.base.exception import NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import Betas, Closure, Defaults, Loss, ParamGroup
from .utils import apply_weight_decay, copy_stochastic_, UPDATE_STRATEGY, apply_cautious
import logging

logger = logging.getLogger(__name__)


class CAME(BaseOptimizer):
    r"""Confidence-guided Adaptive Memory Efficient Optimization.

    :param params: ParamGroup. iterable of parameters to optimize or dicts defining parameter groups.
    :param lr: float. learning rate.
    :param betas: Betas. coefficients used for computing running averages of gradient and the squared hessian trace.
    :param weight_decay: float. weight decay (L2 penalty).
    :param weight_decouple: bool. the optimizer uses decoupled weight decay as in AdamW.
    :param fixed_decay: bool. fix weight decay.
    :param clip_threshold: float. threshold of root-mean-square of final gradient update.
    :param ams_bound: bool. whether to use the AMSBound variant.
    :param eps1: float. term added to the denominator to improve numerical stability.
    :param eps2: float. term added to the denominator to improve numerical stability.
    :param cautious: bool: (deprecated, use update strategy)
        Use cautious mask on parameter update - https://arxiv.org/abs/2411.16085 (default: False)
    :param update_strategy: str: (NOTE: for backwards compatibility, cautious parameter being set to true will override to cautious)
        Determine the update strategy to use, valid values are 'unmodified', 'cautious' (https://arxiv.org/abs/2411.16085),
        'grams' (https://arxiv.org/abs/2412.17107), and 'both' (cautious then grams sequentially) (default: unmodified)
    :param sync_chunk_size: int: Size of chunks to sync between devices (default: 128)
    :param state_storage_dtype: str|torch.dtype: Data type for storing optimizer state (default: bfloat16)
    :param state_storage_device: str|torch.device: Device for storing optimizer state (default: cpu)
    :param cautious_weight_decay: bool: Applies weight decay only to parameter coordinates whose signs align with the optimizer update. (default: False)
    :param compile_step: bool: Use torch.compile on the core per-parameter step (default: False)
    :param foreach: bool: Use torch._foreach_* operations for unfactored (1D/0D) parameters (default: False)
    :param non_factored_confidence: bool: Apply confidence/residual mechanism to non-factored (1D/0D) parameters (default: False)
    """

    def __init__(
        self,
        params: ParamGroup,
        lr: float = 5e-5,
        betas: Betas = (0.9, 0.999, 0.9999),
        weight_decay: float = 0.0,
        weight_decouple: bool = True,
        fixed_decay: bool = False,
        clip_threshold: float = 1.0,
        ams_bound: bool = False,
        eps1: float = 1e-30,
        eps2: float = 1e-16,
        cautious: bool = False,
        update_strategy: UPDATE_STRATEGY = 'unmodified',
        sync_chunk_size: int = 128,
        state_storage_dtype: str|torch.dtype = torch.bfloat16,
        state_storage_device: str|torch.device = "cpu",
        cautious_weight_decay: bool = False,
        compile_step: bool = False,
        foreach: bool = False,
        non_factored_confidence: bool = True,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_betas(betas)
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_non_negative(eps1, 'eps1')
        self.validate_non_negative(eps2, 'eps2')

        # Loop over the keys in the kwargs dictionary
        for key in kwargs:
            logging.warning(
                f"Unrecognized optimizer argument '{key}'. It will be ignored."
            )

        if isinstance(state_storage_dtype, str):
            normalized_str_dtype = state_storage_dtype.strip().lower()
            if normalized_str_dtype == "float32":
                final_dtype = torch.float32
            elif normalized_str_dtype == "float16":
                final_dtype = torch.float16
            elif normalized_str_dtype == "bfloat16":
                final_dtype = torch.bfloat16
            else:
                final_dtype = torch.bfloat16
        else:
            final_dtype = state_storage_dtype

        self.sync_chunk_size = sync_chunk_size
        self.state_storage_dtype = final_dtype
        self.state_storage_device = state_storage_device

        # Caches to avoid per-parameter tensor allocations in compiled step
        self._scalar_cache: dict = {}       # (device, group_idx) -> dict of scalar tensors
        self._empty_tensor_cache: dict = {}  # device -> empty(0) tensor for AMSBound placeholder

        if update_strategy is not None and update_strategy not in {'unmodified','cautious','grams','both'}:
            raise ValueError("Invalid update strategy: {}".format(update_strategy))

        # If cautious true, override update strategy to cautious
        if cautious:
            update_strategy = 'cautious'

        self.clip_threshold = clip_threshold
        self.eps1 = eps1
        self.eps2 = eps2

        # Compiled step callables (lazily compiled on first step() call)
        self._compiled_factored = None
        self._compiled_unfactored = None

        defaults: Defaults = {
            'lr': lr,
            'betas': betas,
            'weight_decay': weight_decay,
            'weight_decouple': weight_decouple,
            'fixed_decay': fixed_decay,
            'ams_bound': ams_bound,
            'eps1': eps1,
            'eps2': eps2,
            'cautious':cautious,
            'update_strategy':update_strategy,
            'sync_chunk_size': sync_chunk_size,
            'state_storage_dtype': final_dtype,
            'state_storage_device': state_storage_device,
            'clip_threshold': clip_threshold,
            'cautious_weight_decay': cautious_weight_decay,
            'compile_step': compile_step,
            'foreach': foreach,
            'non_factored_confidence': non_factored_confidence,
        }
        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'CAME'

    def init_group(self, group, **kwargs) -> None:
        pass

    @torch.no_grad()
    def reset(self):
        for group in self.param_groups:
            group['step'] = 0
            for p in group['params']:
                state = self.state[p]

                grad = p.grad

                grad_shape: Tuple[int, ...] = grad.shape
                factored: bool = self.get_options(grad_shape)

                state["exp_avg"] = torch.zeros_like(p,
                                                dtype=self.state_storage_dtype,
                                                device=self.state_storage_device)
                if factored:
                    state['exp_avg_sq_row'] = torch.zeros(
                        grad_shape[:-1],
                        dtype=torch.float32,
                        device=self.state_storage_device
                    )
                    state['exp_avg_sq_col'] = torch.zeros(
                        grad_shape[:-2] + grad_shape[-1:],
                        dtype=torch.float32,
                        device=self.state_storage_device
                    )
                    state['exp_avg_res_row'] = torch.zeros(
                        grad_shape[:-1],
                        dtype=torch.float32,
                        device=self.state_storage_device
                    )
                    state['exp_avg_res_col'] = torch.zeros(
                        grad_shape[:-2] + grad_shape[-1:],
                        dtype=torch.float32,
                        device=self.state_storage_device
                    )
                else:
                    state['exp_avg_sq'] = torch.zeros_like(grad,
                                                dtype=self.state_storage_dtype,
                                                device=self.state_storage_device)

                if group['ams_bound']:
                    state['exp_avg_sq_hat'] = torch.zeros_like(grad,
                                                dtype=self.state_storage_dtype,
                                                device=self.state_storage_device)

                # Non-factored confidence residual state
                if not factored and group['non_factored_confidence']:
                    state['exp_avg_res'] = torch.zeros_like(
                        grad,
                        dtype=torch.float32,
                        device=self.state_storage_device
                    )

                if self.state_storage_device == "cpu":
                    state["exp_avg"] = state["exp_avg"].pin_memory()

                    if factored:
                        state['exp_avg_sq_row'] = state["exp_avg_sq_row"].pin_memory()
                        state['exp_avg_sq_col'] = state["exp_avg_sq_col"].pin_memory()
                        state['exp_avg_res_row'] = state["exp_avg_res_row"].pin_memory()
                        state['exp_avg_res_col'] = state["exp_avg_res_col"].pin_memory()
                    else:
                        state['exp_avg_sq'] = state['exp_avg_sq'].pin_memory()

                    if group['ams_bound']:
                        state['exp_avg_sq_hat'] = state['exp_avg_sq_hat'].pin_memory()

                    if not factored and group['non_factored_confidence']:
                        state['exp_avg_res'] = state['exp_avg_res'].pin_memory()

    @staticmethod
    def get_options(shape: Tuple[int, ...]) -> bool:
        r"""Get `factored`."""
        return len(shape) >= 2

    @staticmethod
    def get_rms(x: torch.Tensor) -> float:
        r"""Get RMS."""
        return x.norm(2) / math.sqrt(x.numel())

    @staticmethod
    def approximate_sq_grad(
        exp_avg_sq_row: torch.Tensor,
        exp_avg_sq_col: torch.Tensor,
        output: torch.Tensor,
    ):
        r"""Get approximation of EMA of squared gradient."""
        r_factor: torch.Tensor = (exp_avg_sq_row / exp_avg_sq_row.mean(dim=-1, keepdim=True)).rsqrt_().unsqueeze(-1)
        c_factor: torch.Tensor = exp_avg_sq_col.unsqueeze(-2).rsqrt()
        torch.mul(r_factor, c_factor, out=output)

    # --- Compiled Core Functions ---

    @staticmethod
    @torch.no_grad()
    def _core_unfactored_fp32(
        grad: torch.Tensor,
        exp_avg: torch.Tensor,
        exp_avg_sq: torch.Tensor,
        exp_avg_sq_hat: torch.Tensor,
        beta1: torch.Tensor,
        beta2: torch.Tensor,
        eps1: torch.Tensor,
        clip_threshold: torch.Tensor,
        use_amsbound: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Core unfactored per-parameter step. All inputs are FP32 tensors on the compute device.

        Modifies exp_avg and exp_avg_sq in-place.
        Returns ``(exp_avg_post_momentum, pre_momentum_update)`` as a tuple.
        The pre-momentum update is needed for non-factored confidence computation.
        """
        # update = grad^2 + eps1
        update = torch.mul(grad, grad).add_(eps1)

        # EMA of squared gradient
        exp_avg_sq.mul_(beta2).add_(update, alpha=1.0 - beta2)
        torch.rsqrt(exp_avg_sq, out=update)

        # AMSBound
        if use_amsbound:
            torch.max(exp_avg_sq_hat, 1.0 / update, out=exp_avg_sq_hat)
            torch.rsqrt(exp_avg_sq_hat / beta2, out=update)

        # Precondition gradient
        update.mul_(grad)

        # RMS clip
        rms = update.norm(2) / torch.sqrt(
            torch.tensor(update.numel(), device=update.device, dtype=torch.float32)
        )
        clip_factor = (rms / clip_threshold).clamp_(min=1.0)
        update.div_(clip_factor)

        # Save pre-momentum update for confidence computation
        pre_momentum_update = update.clone()

        # Momentum
        exp_avg.mul_(beta1).add_(update, alpha=1.0 - beta1)

        # Return (post-momentum exp_avg, pre-momentum update)
        return exp_avg, pre_momentum_update

    @staticmethod
    @torch.no_grad()
    def _core_factored_fp32(
        grad: torch.Tensor,
        exp_avg: torch.Tensor,
        exp_avg_sq_row: torch.Tensor,
        exp_avg_sq_col: torch.Tensor,
        exp_avg_res_row: torch.Tensor,
        exp_avg_res_col: torch.Tensor,
        exp_avg_sq_hat: torch.Tensor,
        beta1: torch.Tensor,
        beta2: torch.Tensor,
        beta3: torch.Tensor,
        eps1: torch.Tensor,
        eps2: torch.Tensor,
        clip_threshold: torch.Tensor,
        use_amsbound: bool,
    ) -> torch.Tensor:
        r"""Core factored per-parameter step. All inputs are FP32 tensors on the compute device.

        Modifies all state tensors in-place. Returns the ``update`` tensor
        with confidence modulation applied.
        """
        # update = grad^2 + eps1
        update = torch.mul(grad, grad).add_(eps1)

        # Factored second moment EMA
        exp_avg_sq_row.mul_(beta2).add_(update.mean(dim=-1), alpha=1.0 - beta2)
        exp_avg_sq_col.mul_(beta2).add_(update.mean(dim=-2), alpha=1.0 - beta2)

        # Approximate sq grad as denominator
        r_factor = (exp_avg_sq_row / exp_avg_sq_row.mean(dim=-1, keepdim=True)).rsqrt_().unsqueeze(-1)
        c_factor = exp_avg_sq_col.unsqueeze(-2).rsqrt()
        torch.mul(r_factor, c_factor, out=update)

        # AMSBound
        if use_amsbound:
            torch.max(exp_avg_sq_hat, 1.0 / update, out=exp_avg_sq_hat)
            torch.rsqrt(exp_avg_sq_hat / beta2, out=update)

        # Precondition gradient
        update.mul_(grad)

        # RMS clip
        rms = update.norm(2) / torch.sqrt(
            torch.tensor(update.numel(), device=update.device, dtype=torch.float32)
        )
        clip_factor = (rms / clip_threshold).clamp_(min=1.0)
        update.div_(clip_factor)

        # Momentum
        exp_avg.mul_(beta1).add_(update, alpha=1.0 - beta1)

        # Confidence (residual)
        res = update - exp_avg
        res.pow_(2).add_(eps2)

        exp_avg_res_row.mul_(beta3).add_(res.mean(dim=-1), alpha=1.0 - beta3)
        exp_avg_res_col.mul_(beta3).add_(res.mean(dim=-2), alpha=1.0 - beta3)

        # Approximate sq grad for confidence modulation
        r_factor_res = (exp_avg_res_row / exp_avg_res_row.mean(dim=-1, keepdim=True)).rsqrt_().unsqueeze(-1)
        c_factor_res = exp_avg_res_col.unsqueeze(-2).rsqrt()
        torch.mul(r_factor_res, c_factor_res, out=update)
        update.mul_(exp_avg)

        return update

    @staticmethod
    @torch.no_grad()
    def _core_factored_full_fp32(
        grad: torch.Tensor,
        exp_avg: torch.Tensor,
        exp_avg_sq_row: torch.Tensor,
        exp_avg_sq_col: torch.Tensor,
        exp_avg_res_row: torch.Tensor,
        exp_avg_res_col: torch.Tensor,
        exp_avg_sq_hat: torch.Tensor,
        beta1: torch.Tensor,
        beta2: torch.Tensor,
        beta3: torch.Tensor,
        eps1: torch.Tensor,
        eps2: torch.Tensor,
        clip_threshold: torch.Tensor,
        use_amsbound: bool,
        lr: torch.Tensor,
        weight_decay: torch.Tensor,
        weight_decouple: bool,
        fixed_decay: bool,
        cautious_weight_decay: bool,
        use_cautious: bool,
        use_grams: bool,
        p_fp32: torch.Tensor,
    ) -> None:
        r"""Core factored per-parameter step INCLUDING weight decay, LR scale, update strategy, and param update.

        All inputs are FP32 tensors on the compute device.
        Modifies all state tensors and p_fp32 in-place.
        Branch booleans (use_amsbound, weight_decouple, fixed_decay, cautious_weight_decay,
        use_cautious, use_grams) are compile-time constants resolved during tracing.
        """
        # update = grad^2 + eps1
        update = torch.mul(grad, grad).add_(eps1)

        # Factored second moment EMA
        exp_avg_sq_row.mul_(beta2).add_(update.mean(dim=-1), alpha=1.0 - beta2)
        exp_avg_sq_col.mul_(beta2).add_(update.mean(dim=-2), alpha=1.0 - beta2)

        # Approximate sq grad as denominator
        r_factor = (exp_avg_sq_row / exp_avg_sq_row.mean(dim=-1, keepdim=True)).rsqrt_().unsqueeze(-1)
        c_factor = exp_avg_sq_col.unsqueeze(-2).rsqrt()
        torch.mul(r_factor, c_factor, out=update)

        # AMSBound
        if use_amsbound:
            torch.max(exp_avg_sq_hat, 1.0 / update, out=exp_avg_sq_hat)
            torch.rsqrt(exp_avg_sq_hat / beta2, out=update)

        # Precondition gradient
        update.mul_(grad)

        # RMS clip — numel is static with dynamic=False, so math.sqrt is trace-time constant
        rms = update.norm(2) / math.sqrt(update.numel())
        clip_factor = (rms / clip_threshold).clamp_(min=1.0)
        update.div_(clip_factor)

        # Momentum
        exp_avg.mul_(beta1).add_(update, alpha=1.0 - beta1)

        # Confidence (residual)
        res = update - exp_avg
        res.pow_(2).add_(eps2)

        exp_avg_res_row.mul_(beta3).add_(res.mean(dim=-1), alpha=1.0 - beta3)
        exp_avg_res_col.mul_(beta3).add_(res.mean(dim=-2), alpha=1.0 - beta3)

        # Approximate sq grad for confidence modulation
        r_factor_res = (exp_avg_res_row / exp_avg_res_row.mean(dim=-1, keepdim=True)).rsqrt_().unsqueeze(-1)
        c_factor_res = exp_avg_res_col.unsqueeze(-2).rsqrt()
        torch.mul(r_factor_res, c_factor_res, out=update)
        update.mul_(exp_avg)

        # === Weight decay (inlined for compilation; branch resolved at trace time) ===
        if cautious_weight_decay:
            # Cautious weight decay: apply WD only where gradient and param agree in sign
            cwd_mask = (grad * p_fp32 >= 0).to(p_fp32.dtype)
            p_fp32.mul_(1.0 - weight_decay * lr * cwd_mask)
        elif weight_decouple:
            wd_factor = 1.0 if fixed_decay else lr
            p_fp32.mul_(1.0 - weight_decay * wd_factor)
        else:
            # Standard (non-decoupled) weight decay: add scaled parameter to gradient
            grad.add_(p_fp32, alpha=weight_decay)

        # === LR scale ===
        update.mul_(lr)

        # === Update strategy (resolved at compile time) ===
        if use_cautious:
            mask = (update * grad > 0).to(grad.dtype)
            mask.div_(mask.mean().clamp_(min=1e-3))
            update.mul_(mask)
        if use_grams:
            update.copy_(torch.sign(grad) * update.abs())

        # === Parameter update ===
        p_fp32.add_(-update)

    def _compile_core_fns(self) -> None:
        r"""Lazily compile the core step functions with torch.compile."""
        if self.defaults.get('compile_step', False):
            try:
                with torch._dynamo.utils.disable_cache_limit():
                    self._compiled_unfactored = torch.compile(
                        self._core_unfactored_fp32, fullgraph=True, dynamic=False
                    )
                    self._compiled_factored = torch.compile(
                        self._core_factored_full_fp32, fullgraph=True, dynamic=False
                    )
                logger.info("CAME core functions compiled with torch.compile(fullgraph=True, dynamic=False).")
            except Exception as e:
                logger.warning(f"torch.compile(fullgraph=True, dynamic=False) failed: {e}. Falling back to uncompiled step.")
                self._compiled_unfactored = self._core_unfactored_fp32
                self._compiled_factored = self._core_factored_full_fp32
        else:
            self._compiled_unfactored = self._core_unfactored_fp32
            self._compiled_factored = self._core_factored_full_fp32

    # --- Scalar Tensor Caching (avoids per-parameter allocation) ---

    def _get_scalar_tensors(
        self, device: torch.device, group_idx: int, group: dict
    ):
        r"""Get or create cached scalar tensors for a given (device, group)."""
        key = (device, group_idx)
        if key not in self._scalar_cache:
            self._scalar_cache[key] = {
                'beta1_t': torch.tensor(0.0, device=device, dtype=torch.float32),
                'beta2_t': torch.tensor(0.0, device=device, dtype=torch.float32),
                'beta3_t': torch.tensor(0.0, device=device, dtype=torch.float32),
                'eps1_t': torch.tensor(0.0, device=device, dtype=torch.float32),
                'eps2_t': torch.tensor(0.0, device=device, dtype=torch.float32),
                'clip_t': torch.tensor(0.0, device=device, dtype=torch.float32),
                'lr_t': torch.tensor(0.0, device=device, dtype=torch.float32),
                'wd_t': torch.tensor(0.0, device=device, dtype=torch.float32),
            }
        scalars = self._scalar_cache[key]
        betas = group['betas']
        scalars['beta1_t'].fill_(betas[0])
        scalars['beta2_t'].fill_(betas[1])
        scalars['beta3_t'].fill_(betas[2])
        scalars['eps1_t'].fill_(group['eps1'])
        scalars['eps2_t'].fill_(group['eps2'])
        scalars['clip_t'].fill_(group['clip_threshold'])
        scalars['lr_t'].fill_(group['lr'])
        scalars['wd_t'].fill_(group['weight_decay'])
        return scalars

    def _get_empty_tensor(self, device: torch.device) -> torch.Tensor:
        r"""Get or create cached empty tensor for AMSBound placeholder."""
        if device not in self._empty_tensor_cache:
            self._empty_tensor_cache[device] = torch.empty(0, device=device)
        return self._empty_tensor_cache[device]

    # --- Foreach Support (Unfactored params only) ---

    @torch.no_grad()
    def _foreach_unfactored_step(
        self,
        group,
        active_params: list,
        beta1: float,
        beta2: float,
        beta3: float,
        compute_device: torch.device,
    ) -> None:
        r"""Foreach step for unfactored (1D/0D) parameters.

        Batches operations using ``torch._foreach_*`` for better GPU utilization.
        Handles AMSBound, update strategy, weight decay, and non-factored confidence.
        """
        use_amsbound = group['ams_bound']
        update_strategy = group['update_strategy']
        lr = group['lr']
        wd = group['weight_decay']
        wd_decouple = group['weight_decouple']
        fixed_decay = group['fixed_decay']
        cwd = group['cautious_weight_decay']
        nfc = group['non_factored_confidence']

        # Collect phase: build lists of FP32 tensors on compute device
        p_fp32_list = []
        grad_list = []
        exp_avg_list = []
        exp_avg_sq_list = []
        exp_avg_sq_hat_list = [] if use_amsbound else None
        exp_avg_res_list = [] if nfc else None
        state_list = []

        for p in active_params:
            if p.grad is None:
                continue

            state = self.state[p]
            state_list.append(state)

            # Transfer to compute device
            p_fp32 = p.to(compute_device, dtype=torch.float32, non_blocking=True)
            grad = p.grad.data.to(torch.float32).to(compute_device, non_blocking=True)
            exp_avg = state["exp_avg"].to(compute_device, non_blocking=True, dtype=torch.float32)
            exp_avg_sq = state["exp_avg_sq"].to(compute_device, non_blocking=True, dtype=torch.float32)

            p_fp32_list.append(p_fp32)
            grad_list.append(grad)
            exp_avg_list.append(exp_avg)
            exp_avg_sq_list.append(exp_avg_sq)

            if use_amsbound:
                eash = state["exp_avg_sq_hat"].to(compute_device, non_blocking=True, dtype=torch.float32)
                exp_avg_sq_hat_list.append(eash)

            if nfc:
                ear = state['exp_avg_res'].to(compute_device, non_blocking=True, dtype=torch.float32)
                exp_avg_res_list.append(ear)

        if not p_fp32_list:
            return

        # ---- Batch compute phase ----

        # 1. Denominator: update = grad^2 + eps1, then EMA, then rsqrt
        #    We reuse exp_avg_sq_list as 'update' after setting it to grad^2 + eps1
        update_list = [g.mul(g) for g in grad_list]  # grad^2 (per-tensor, no foreach variant for mul)
        _ = [u.add_(group['eps1']) for u in update_list]  # + eps1

        # EMA of squared gradient: exp_avg_sq = beta2 * exp_avg_sq + (1-beta2) * update
        torch._foreach_mul_(exp_avg_sq_list, beta2)
        torch._foreach_add_(exp_avg_sq_list, update_list, alpha=1.0 - beta2)

        # rsqrt: denom = 1/sqrt(exp_avg_sq)
        torch._foreach_rsqrt_(exp_avg_sq_list)

        # AMSBound
        if use_amsbound:
            torch._foreach_max_(exp_avg_sq_hat_list, [1.0 / d for d in exp_avg_sq_list])
            torch._foreach_div_(exp_avg_sq_hat_list, beta2)
            torch._foreach_rsqrt_(exp_avg_sq_hat_list)
            denom_list = exp_avg_sq_hat_list
        else:
            denom_list = exp_avg_sq_list

        # 2. Precondition: update = denom * grad
        torch._foreach_mul_(denom_list, grad_list)

        # 3. RMS clip (per-tensor since norm/numel differ)
        for upd in denom_list:
            rms = upd.norm(2) / math.sqrt(upd.numel())
            clip_factor = max(rms / group['clip_threshold'], 1.0)
            upd.div_(clip_factor)

        # Save pre-momentum update values for optional confidence computation
        if nfc:
            pre_momentum_updates = [d.clone() for d in denom_list]

        # 4. Momentum: exp_avg = beta1 * exp_avg + (1-beta1) * update
        torch._foreach_mul_(exp_avg_list, beta1)
        torch._foreach_add_(exp_avg_list, denom_list, alpha=1.0 - beta1)

        # 5. Confidence residual modulation (non-factored)
        if nfc:
            # For each param: res = (pre_momentum_update - post_momentum_exp_avg)^2 + eps2
            # Then: exp_avg_res = beta3 * exp_avg_res + (1-beta3) * res
            # Then: final_update = exp_avg / sqrt(exp_avg_res)
            final_update_list = []
            for i in range(len(exp_avg_list)):
                pre_upd = pre_momentum_updates[i]
                post_exp_avg = exp_avg_list[i]
                ear = exp_avg_res_list[i]

                res = pre_upd.sub(post_exp_avg).pow_(2).add_(group['eps2'])
                ear.mul_(beta3).add_(res, alpha=1.0 - beta3)
                final_update = post_exp_avg.div_(ear.sqrt_().add_(group['eps2']))
                final_update_list.append(final_update)
        else:
            # update = exp_avg (no confidence)
            final_update_list = exp_avg_list

        # 6. Weight decay
        for i, p_fp32 in enumerate(p_fp32_list):
            apply_weight_decay(
                p=p_fp32,
                grad=grad_list[i],
                lr=lr,
                weight_decay=wd,
                weight_decouple=wd_decouple,
                fixed_decay=fixed_decay,
                cautious_weight_decay=cwd,
            )

        # 7. LR scale
        torch._foreach_mul_(final_update_list, lr)

        # 8. Update strategy
        if update_strategy == 'cautious':
            for i in range(len(final_update_list)):
                apply_cautious(final_update_list[i], grad_list[i])
                # apply_cautious modifies in-place, no mask needed
        elif update_strategy == 'grams':
            for i in range(len(final_update_list)):
                final_update_list[i].copy_(torch.sign(grad_list[i]) * final_update_list[i].abs())
        elif update_strategy == 'both':
            for i in range(len(final_update_list)):
                apply_cautious(final_update_list[i], grad_list[i])
                final_update_list[i].copy_(torch.sign(grad_list[i]) * final_update_list[i].abs())

        # 9. Apply: p -= update
        torch._foreach_add_(p_fp32_list, final_update_list, alpha=-1.0)

        # ---- Write-back phase with sync chunking ----
        for i, state in enumerate(state_list):
            p = active_params[i]
            p_fp32 = p_fp32_list[i]
            device = p.device

            # Parameter write-back
            if device.type == "cpu":
                if p.dtype == torch.bfloat16:
                    copy_stochastic_(p.data, p_fp32)
                else:
                    p.data.copy_(p_fp32)
            else:
                if p.dtype == torch.bfloat16:
                    copy_stochastic_(p, p_fp32)
                else:
                    p.data.copy_(p_fp32, non_blocking=True)

            # State write-back
            exp_avg = exp_avg_list[i]
            exp_avg_sq = exp_avg_sq_list[i]

            if self.state_storage_dtype == torch.bfloat16:
                copy_stochastic_(state["exp_avg"], exp_avg)
                copy_stochastic_(state["exp_avg_sq"], exp_avg_sq)
                if use_amsbound:
                    copy_stochastic_(state["exp_avg_sq_hat"], exp_avg_sq_hat_list[i])
            else:
                state["exp_avg"].copy_(exp_avg, non_blocking=True)
                state["exp_avg_sq"].copy_(exp_avg_sq, non_blocking=True)
                if use_amsbound:
                    state["exp_avg_sq_hat"].copy_(exp_avg_sq_hat_list[i], non_blocking=True)

            if nfc:
                ear = exp_avg_res_list[i]
                if self.state_storage_dtype == torch.bfloat16:
                    copy_stochastic_(state['exp_avg_res'], ear)
                else:
                    state['exp_avg_res'].copy_(ear, non_blocking=True)

            # Sync chunking
            if (i + 1) % group.get('sync_chunk_size', 128) == 0:
                torch.cuda.synchronize()

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:
        loss: Loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if 'step' in group:
                group['step'] += 1
            else:
                group['step'] = 1

            beta1, beta2, beta3 = group['betas']
            use_foreach = group.get('foreach', False)
            nfc = group.get('non_factored_confidence', False)
            update_strategy = group['update_strategy']
            lr = group['lr']

            # Lazily compile core functions on first step
            if self._compiled_unfactored is None:
                self._compile_core_fns()

            # Select compiled or uncompiled callables
            core_unfactored_fn = self._compiled_unfactored
            core_factored_fn = self._compiled_factored

            # Bucket params for foreach (unfactored only)
            unfactored_foreach_params = []
            if use_foreach:
                compute_device_for_foreach = None
                for p in group['params']:
                    if p.grad is None:
                        continue
                    grad_shape = p.grad.shape
                    if not self.get_options(grad_shape):
                        unfactored_foreach_params.append(p)
                        if compute_device_for_foreach is None:
                            first_device = p.device
                            compute_device_for_foreach = (
                                torch.cuda.current_device() if first_device.type == "cpu" else first_device
                            )

                if unfactored_foreach_params and compute_device_for_foreach is not None:
                    self._foreach_unfactored_step(
                        group, unfactored_foreach_params, beta1, beta2, beta3, compute_device_for_foreach
                    )

            # Per-parameter loop for factored params or when foreach is disabled
            for i, p in enumerate(group["params"]):
                if p.grad is None:
                    continue

                grad = p.grad.data
                if grad.is_sparse:
                    raise NoSparseGradientError(str(self))

                # Skip unfactored params that were already handled by foreach
                if use_foreach and not self.get_options(grad.shape):
                    continue

                state = self.state[p]
                device = p.device

                grad_shape: Tuple[int, ...] = grad.shape
                factored: bool = self.get_options(grad_shape)

                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(p,
                                                    dtype=self.state_storage_dtype,
                                                    device=self.state_storage_device)
                    if factored:
                        state['exp_avg_sq_row'] = torch.zeros(
                            grad_shape[:-1],
                            dtype=torch.float32,
                            device=self.state_storage_device
                        )
                        state['exp_avg_sq_col'] = torch.zeros(
                            grad_shape[:-2] + grad_shape[-1:],
                            dtype=torch.float32,
                            device=self.state_storage_device
                        )
                        state['exp_avg_res_row'] = torch.zeros(
                            grad_shape[:-1],
                            dtype=torch.float32,
                            device=self.state_storage_device
                        )
                        state['exp_avg_res_col'] = torch.zeros(
                            grad_shape[:-2] + grad_shape[-1:],
                            dtype=torch.float32,
                            device=self.state_storage_device
                        )
                    else:
                        state['exp_avg_sq'] = torch.zeros_like(grad,
                                                    dtype=self.state_storage_dtype,
                                                    device=self.state_storage_device)

                    if group['ams_bound']:
                        state['exp_avg_sq_hat'] = torch.zeros_like(grad,
                                                    dtype=self.state_storage_dtype,
                                                    device=self.state_storage_device)

                    # Non-factored confidence residual state
                    if not factored and nfc:
                        state['exp_avg_res'] = torch.zeros_like(
                            grad,
                            dtype=torch.float32,
                            device=self.state_storage_device
                        )

                    if self.state_storage_device == "cpu":
                        state["exp_avg"] = state["exp_avg"].pin_memory()

                        if factored:
                            state['exp_avg_sq_row'] = state["exp_avg_sq_row"].pin_memory()
                            state['exp_avg_sq_col'] = state["exp_avg_sq_col"].pin_memory()
                            state['exp_avg_res_row'] = state["exp_avg_res_row"].pin_memory()
                            state['exp_avg_res_col'] = state["exp_avg_res_col"].pin_memory()
                        else:
                            state['exp_avg_sq'] = state['exp_avg_sq'].pin_memory()

                        if group['ams_bound']:
                            state['exp_avg_sq_hat'] = state['exp_avg_sq_hat'].pin_memory()

                        if not factored and nfc:
                            state['exp_avg_res'] = state['exp_avg_res'].pin_memory()

                # ========= Determine compute device =========
                if device.type == "cpu":
                    compute_device = torch.cuda.current_device()
                else:
                    compute_device = device

                # ========= Asynchronously queue state to compute device =========
                exp_avg = state["exp_avg"].to(
                    compute_device,
                    non_blocking=True,
                    dtype=torch.float32
                )
                if factored:
                    exp_avg_sq_row = state["exp_avg_sq_row"].to(
                        compute_device,
                        non_blocking=True,
                        dtype=torch.float32
                    )
                    exp_avg_sq_col = state["exp_avg_sq_col"].to(
                        compute_device,
                        non_blocking=True,
                        dtype=torch.float32
                    )
                    exp_avg_res_row = state["exp_avg_res_row"].to(
                        compute_device,
                        non_blocking=True,
                        dtype=torch.float32
                    )
                    exp_avg_res_col = state["exp_avg_res_col"].to(
                        compute_device,
                        non_blocking=True,
                        dtype=torch.float32
                    )
                else:
                    exp_avg_sq = state["exp_avg_sq"].to(
                        compute_device,
                        non_blocking=True,
                        dtype=torch.float32
                    )

                if group['ams_bound']:
                    exp_avg_sq_hat = state["exp_avg_sq_hat"].to(
                        compute_device,
                        non_blocking=True,
                        dtype=torch.float32
                    )

                grad = grad.to(torch.float32).to(compute_device, non_blocking=True)
                p_fp32 = p.to(compute_device, dtype=torch.float32, non_blocking=True)

                # ========= Core computation (compiled or uncompiled) =========
                if factored:
                    # Get cached scalar tensors (avoids per-parameter allocation)
                    group_idx = self.param_groups.index(group)
                    scalars = self._get_scalar_tensors(compute_device, group_idx, group)

                    ams_hat = exp_avg_sq_hat if group['ams_bound'] else self._get_empty_tensor(compute_device)

                    core_factored_fn(
                        grad, exp_avg,
                        exp_avg_sq_row, exp_avg_sq_col,
                        exp_avg_res_row, exp_avg_res_col,
                        ams_hat,
                        scalars['beta1_t'], scalars['beta2_t'], scalars['beta3_t'],
                        scalars['eps1_t'], scalars['eps2_t'], scalars['clip_t'],
                        group['ams_bound'],
                        scalars['lr_t'], scalars['wd_t'],
                        group['weight_decouple'],
                        group['fixed_decay'],
                        group['cautious_weight_decay'],
                        update_strategy in {'cautious', 'both'},
                        update_strategy in {'grams', 'both'},
                        p_fp32,
                    )
                else:
                    # Unfactored path (not handled by foreach)
                    beta1_t = torch.tensor(beta1, device=compute_device, dtype=torch.float32)
                    beta2_t = torch.tensor(beta2, device=compute_device, dtype=torch.float32)
                    beta3_t = torch.tensor(beta3, device=compute_device, dtype=torch.float32)
                    eps1_t = torch.tensor(group['eps1'], device=compute_device, dtype=torch.float32)
                    eps2_t = torch.tensor(group['eps2'], device=compute_device, dtype=torch.float32)
                    clip_t = torch.tensor(group['clip_threshold'], device=compute_device, dtype=torch.float32)

                    exp_avg_result, pre_momentum_update = core_unfactored_fn(
                        grad, exp_avg, exp_avg_sq,
                        exp_avg_sq_hat if group['ams_bound'] else torch.empty(0, device=compute_device),
                        beta1_t, beta2_t, eps1_t, clip_t,
                        group['ams_bound'],
                    )

                    # Non-factored confidence: apply residual modulation
                    if nfc:
                        if 'exp_avg_res' not in state:
                            # Lazy creation if state was already initialized without nfc
                            state['exp_avg_res'] = torch.zeros_like(
                                grad, dtype=torch.float32, device=self.state_storage_device
                            )
                            if self.state_storage_device == "cpu":
                                state['exp_avg_res'] = state['exp_avg_res'].pin_memory()
                        exp_avg_res_nonfac = state['exp_avg_res'].to(
                            compute_device, non_blocking=True, dtype=torch.float32
                        )

                        # Residual: (pre_momentum_update - post_momentum_exp_avg)^2 + eps2
                        res = pre_momentum_update.sub(exp_avg_result).pow_(2).add_(eps2_t)

                        # EMA of residual
                        exp_avg_res_nonfac.mul_(beta3_t).add_(res, alpha=1.0 - beta3_t)

                        # Confidence modulation: update = exp_avg / sqrt(exp_avg_res)
                        # Use division (not div_) to avoid corrupting exp_avg_result which IS exp_avg
                        update = exp_avg_result / (exp_avg_res_nonfac.sqrt() + eps2_t)
                    else:
                        # Standard unfactored: update = exp_avg
                        update = exp_avg_result

                # For unfactored (non-foreach) path: apply weight decay, LR scale,
                # update strategy, and parameter update.
                # Factored path is fully handled by _core_factored_full_fp32().
                if not factored:
                    apply_weight_decay(
                        p=p_fp32,
                        grad=grad,
                        lr=lr,
                        weight_decay=group['weight_decay'],
                        weight_decouple=group['weight_decouple'],
                        fixed_decay=group['fixed_decay'],
                        cautious_weight_decay=group['cautious_weight_decay'],
                    )
                    update.mul_(lr)
                    if update_strategy in {'cautious', 'grams', 'both'}:
                        if update_strategy in {'cautious', 'both'}:
                            apply_cautious(update, grad)
                        if update_strategy in {'grams', 'both'}:
                            update.copy_(torch.sign(grad) * update.abs())
                    p_fp32.add_(-update)

                # ========= Write-back =========
                if device.type == "cpu":
                    if p.dtype == torch.bfloat16:
                        copy_stochastic_(p.data, p_fp32)
                    else:
                        p.data.copy_(p_fp32)
                else:
                    if p.dtype == torch.bfloat16:
                        copy_stochastic_(p, p_fp32)
                    else:
                        p.data.copy_(p_fp32, non_blocking=True)

                if self.state_storage_dtype == torch.bfloat16:
                    copy_stochastic_(state["exp_avg"], exp_avg)
                    if not factored:
                        copy_stochastic_(state["exp_avg_sq"], exp_avg_sq)
                    if group['ams_bound']:
                        copy_stochastic_(state["exp_avg_sq_hat"], exp_avg_sq_hat)
                else:
                    state["exp_avg"].copy_(exp_avg, non_blocking=True)
                    if not factored:
                        state["exp_avg_sq"].copy_(exp_avg_sq, non_blocking=True)
                    if group['ams_bound']:
                        state["exp_avg_sq_hat"].copy_(exp_avg_sq_hat, non_blocking=True)

                if factored:
                    state["exp_avg_sq_row"].copy_(exp_avg_sq_row, non_blocking=True)
                    state["exp_avg_sq_col"].copy_(exp_avg_sq_col, non_blocking=True)
                    state["exp_avg_res_row"].copy_(exp_avg_res_row, non_blocking=True)
                    state["exp_avg_res_col"].copy_(exp_avg_res_col, non_blocking=True)

                # Non-factored confidence write-back
                if not factored and nfc and 'exp_avg_res' in state:
                    if self.state_storage_dtype == torch.bfloat16:
                        copy_stochastic_(state['exp_avg_res'], exp_avg_res_nonfac)
                    else:
                        state['exp_avg_res'].copy_(exp_avg_res_nonfac, non_blocking=True)

                # ========= Sync chunking =========
                if (i + 1) % self.sync_chunk_size == 0:
                    torch.cuda.synchronize()

            # Final synchronization
            torch.cuda.synchronize()

        return loss
