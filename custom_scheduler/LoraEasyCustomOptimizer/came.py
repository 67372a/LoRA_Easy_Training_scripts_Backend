# Authored originally by: https://github.com/kozistr
# Source: https://github.com/kozistr/pytorch_optimizer/blob/main/pytorch_optimizer/optimizer/came.py
# With stochastic rounding added per https://github.com/neggles/neurosis/blob/main/src/neurosis/optimizers/came.py

import math
from typing import Tuple

import torch

from pytorch_optimizer.base.exception import NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import Betas, Closure, Defaults, Loss, ParamGroup
import logging

logger = logging.getLogger(__name__)

from .utils import UPDATE_STRATEGY

def copy_stochastic_(target: torch.Tensor, source: torch.Tensor):
    # thanks to Nerogar for fast stochastic pytorch implementation
    # https://github.com/pytorch/pytorch/issues/120376#issuecomment-1974828905
    with torch.no_grad():
        # create a random 16 bit integer
        result = torch.randint_like(
            source,
            dtype=torch.int32,
            low=0,
            high=(1 << 16),
        )

        # add the random number to the lower 16 bit of the mantissa
        result.add_(source.view(dtype=torch.int32))

        # mask off the lower 16 bit of the mantissa
        result.bitwise_and_(-65536)  # -65536 = FFFF0000 as a signed int32

        # copy the higher 16 bit into the target tensor
        target.copy_(result.view(dtype=torch.float32), non_blocking=True)

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
    :param cautious: bool: Use cautious mask on parameter update - https://arxiv.org/abs/2411.16085
    cautious (bool) (deprecated, use update strategy)
        Use cautious mask on parameter update - https://arxiv.org/abs/2411.16085 (default: False)
    update_strategy (str) (NOTE: for backwards compatibility, cautious parameter being set to true will override to cautious)
        Determine the update strategy to use, valid values are 'unmodified', 'cautious' (https://arxiv.org/abs/2411.16085), 
        and 'grams' (https://arxiv.org/abs/2412.17107) (default: unmodified)
    """

    def __init__(
        self,
        params: ParamGroup,
        lr: float = 2e-4,
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

        if update_strategy is not None and update_strategy not in {'unmodified','cautious','grams'}:
            raise ValueError("Invalid update strategy: {}".format(update_strategy))
        
        # If cautious true, override update strategy to cautious
        if cautious:
            update_strategy = 'cautious'

        self.clip_threshold = clip_threshold
        self.eps1 = eps1
        self.eps2 = eps2

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

            for i, p in enumerate(group["params"]):
                if p.grad is None:
                    continue

                grad = p.grad.data
                if grad.is_sparse:
                    raise NoSparseGradientError(str(self))

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

                # ========= Asynchronously queue all operations for this parameter =========
                # Determine target GPU device for computation
                if device.type == "cpu":
                    # If param is on CPU, use default GPU for computation
                    compute_device = torch.cuda.current_device()
                else:
                    # If param is on GPU, use its device
                    compute_device = device

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
                    exp_avg_sq  = state["exp_avg_sq"].to(
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
                p_fp32 = (
                    p.to(compute_device, dtype=torch.float32, non_blocking=True)
                )

                update = torch.mul(grad, grad).add_(self.eps1)

                if factored:
                    exp_avg_sq_row.mul_(beta2).add_(update.mean(dim=-1), alpha=1.0 - beta2)
                    exp_avg_sq_col.mul_(beta2).add_(update.mean(dim=-2), alpha=1.0 - beta2)

                    self.approximate_sq_grad(exp_avg_sq_row, exp_avg_sq_col, update)
                else:
                    exp_avg_sq.mul_(beta2).add_(update, alpha=1.0 - beta2)
                    torch.rsqrt(exp_avg_sq, out=update)

                if group['ams_bound']:
                    torch.max(exp_avg_sq_hat, 1 / update, out=exp_avg_sq_hat)
                    torch.rsqrt(exp_avg_sq_hat / beta2, out=update)

                update.mul_(grad)

                update.div_((self.get_rms(update) / self.clip_threshold).clamp_(min=1.0))

                exp_avg.mul_(beta1).add_(update, alpha=1.0 - beta1)

                res = update - exp_avg
                res.pow_(2).add_(self.eps2)

                if factored:
                    exp_avg_res_row.mul_(beta3).add_(res.mean(dim=-1), alpha=1.0 - beta3)
                    exp_avg_res_col.mul_(beta3).add_(res.mean(dim=-2), alpha=1.0 - beta3)

                    self.approximate_sq_grad(exp_avg_res_row, exp_avg_res_col, update)
                    update.mul_(exp_avg)
                else:
                    update = exp_avg

                self.apply_weight_decay(
                    p=p_fp32,
                    grad=grad,
                    lr=group['lr'],
                    weight_decay=group['weight_decay'],
                    weight_decouple=group['weight_decouple'],
                    fixed_decay=group['fixed_decay'],
                )

                update.mul_(group['lr'])

                if group['update_strategy'] in {'cautious','grams'}:
                    if group['update_strategy'] == 'cautious':
                        mask = (update * grad > 0).to(grad.dtype)
                        mask.div_(mask.mean().clamp_(min=1e-3))
                    elif group['update_strategy'] == 'grams':
                        update.copy_(torch.sign(grad) * update.abs())
                        mask = 1.0
                else:
                    mask = 1.0

                p_fp32.add_(-(update * mask))

                # 3. Queue Device-to-Host copy
                # only use stochastic rounding if using bf16
                if device.type == "cpu":
                    if p.dtype == torch.bfloat16:
                        copy_stochastic_(p.data, p_fp32)
                    else:
                        p.data.copy_(p_fp32)
                else:
                    # Original GPU path
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
                    
                # ========= Check if we need to synchronize =========
                # We synchronize after processing a chunk of parameters.
                # The (i + 1) ensures we sync after the 1st, 2nd, ... chunk.
                if (i + 1) % self.sync_chunk_size == 0:
                    torch.cuda.synchronize()

            # Final synchronization to handle the last partial chunk
            # This ensures all operations for the group are complete before exiting.
            torch.cuda.synchronize()

        return loss
