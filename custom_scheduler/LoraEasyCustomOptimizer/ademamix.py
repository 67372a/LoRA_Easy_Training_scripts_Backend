# Authored by: https://github.com/kozistr
# Source: https://github.com/kozistr/pytorch_optimizer/blob/main/pytorch_optimizer/optimizer/ademamix.py

import math
from typing import Callable, Dict, Optional, Tuple, Union, List, Literal

import torch

from pytorch_optimizer.base.exception import NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import Betas, Closure, Defaults, Loss, ParamGroup
from .utils import UPDATE_STRATEGY, NORM_TYPE, agc, _paper_orthograd, adaptive_eps, _stable_spam_clipping_compile_wrapper, _stable_spam_clipping_impl
import logging

logger = logging.getLogger(__name__)


def copy_stochastic_(target: torch.Tensor, source: torch.Tensor):
    # thanks to Nerogar for fast stochastic pytorch implementation
    # https://github.com/pytorch/pytorch/issues/120376#issuecomment-1974828905
    with torch.no_grad():
        # create a random 16 bit integer using torch.randint with explicit shape
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

# https://github.com/kozistr/pytorch_optimizer/blob/6397d56279ad80b26c4bba7fb4b04852b517fdeb/pytorch_optimizer/optimizer/shampoo_utils.py#L533
@torch.no_grad()
def zero_power_via_newton_schulz_6(grad: torch.Tensor) -> torch.Tensor:
    r"""Compute the zeroth power / orthogonalization of G.

    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a quintic iteration
    whose coefficients are selected to maximize the slope at zero. For the purpose of minimizing steps, it turns out
    to be empirically effective to keep increasing the slope at zero even beyond the point where the iteration no
    longer converges all the way to one everywhere on the interval. This iteration therefore does not produce UV^T but
    rather something like US'V^T where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt
    model performance at all relative to UV^T, where USV^T = G is the SVD.

    :param grad: torch.Tensor. matrix.
    """
    # Inline reshaping step within the method itself.
    G_shape = grad.shape
    grad = grad.view(grad.size(0), -1)

    abc_list = [
      (3955/1024, -8306/1024, 5008/1024),
      (3735/1024, -6681/1024, 3463/1024),
      (3799/1024, -6499/1024, 3211/1024),
      (4019/1024, -6385/1024, 2906/1024),
      (2677/1024, -3029/1024, 1162/1024),
      (2172/1024, -1833/1024,  682/1024)
   ]

    X = grad.float()
    if grad.size(0) > grad.size(1):
        X = X.T

    X = X.div(X.norm().add(1e-16))# ensure top singular value <= 1
    #for _ in range(num_steps):
    for a,b,c in abc_list:
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X

    if grad.size(0) > grad.size(1):
        X = X.T

    # Gradient scaling adaptation from: https://github.com/leloykun/adaptive-muon
    X = torch.einsum('ij,ij->', grad.type_as(X), X).clamp(-1.0, 1.0) * X

    return X.view(G_shape)

@torch._dynamo.utils.disable_cache_limit()
@torch.compile(fullgraph=True, mode="reduce-overhead")
def zero_power_via_newton_schulz_6_compile(grad: torch.Tensor) -> torch.Tensor:
    return zero_power_via_newton_schulz_6(grad)

@torch.no_grad()

def bias_rms(grad: torch.Tensor) -> torch.Tensor:
    rms_value = torch.sqrt(torch.sum(grad.pow(2), dim=0, keepdim=True))
    grad = grad.div(rms_value.add_(1e-16))
    return grad

@torch._dynamo.utils.disable_cache_limit()
@torch.compile(fullgraph=True, mode="reduce-overhead")
def bias_rms_compile(grad: torch.Tensor) -> torch.Tensor:
    return bias_rms(grad)


class AdEMAMix(BaseOptimizer):
    r"""Better, Faster, Older.

    :param params: ParamGroup. iterable of parameters to optimize or dicts defining parameter groups.
    :param lr: float. learning rate.
    :param betas: Betas. coefficients used for computing running averages of gradient and the squared hessian trace.
    :param weight_decay: float. weight decay (L2 penalty).
    :param weight_decouple: bool. the optimizer uses decoupled weight decay as in AdamW.
    :param fixed_decay: bool. fix weight decay.
    :param clip: float. threshold of root-mean-square of gradient update.
    :param alpha: float. usually between 4 and 10 would work well.
    :param t_alpha_beta3: Optional[float]. total number of iterations is preferred when needed.
    :param eps: float. term added to the denominator to improve numerical stability.
    :param centralization: float. center model grad 
    cautious (bool) (deprecated, use update strategy)
        Use cautious mask on parameter update - https://arxiv.org/abs/2411.16085 (default: False)
    update_strategy (str) (NOTE: for backwards compatibility, cautious parameter being set to true will override to cautious)
        Determine the update strategy to use, valid values are 'unmodified', 'cautious' (https://arxiv.org/abs/2411.16085), 
        and 'grams' (https://arxiv.org/abs/2412.17107) (default: unmodified)
    """

    def __init__(
        self,
        params: ParamGroup,
        lr: float = 1e-3,
        betas: Betas = (0.9, 0.999, 0.9999),
        weight_decay: float = 0.0,
        weight_decouple: bool = False,
        fixed_decay: bool = False,
        clip: float = 0.0,
        alpha: float = 5.0,
        t_alpha_beta3: Optional[float] = None,
        eps: float = 1e-8,
        centralization: float = 0.0,
        cautious: bool = False,
        update_strategy: UPDATE_STRATEGY = 'unmodified',
        adopt: bool = False,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_betas(betas)
        self.validate_non_negative(alpha, 'alpha')
        self.validate_non_negative(t_alpha_beta3, 't_alpha_beta3')
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_non_negative(eps, 'eps')
        self.validate_non_negative(clip, 'clip')
        self.validate_non_negative(centralization, 'centralization')

        if update_strategy is not None and update_strategy not in {'unmodified','cautious','grams'}:
            raise ValueError("Invalid update strategy: {}".format(update_strategy))
        
        # If cautious true, override update strategy to cautious
        if cautious:
            update_strategy = 'cautious'

        defaults: Defaults = {
            'lr': lr,
            'betas': betas,
            'weight_decay': weight_decay,
            'weight_decouple': weight_decouple,
            'clip': clip,
            'fixed_decay': fixed_decay,
            'alpha': alpha,
            't_alpha_beta3': t_alpha_beta3,
            'eps': eps,
            'centralization': centralization,
            'cautious': cautious,
            'update_strategy': update_strategy,
            'adopt': adopt
        }

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'AdEMAMix'
    
    def init_group(self, group, **kwargs) -> None:
        pass

    @torch.no_grad()
    def reset(self):
        for group in self.param_groups:
            group['step'] = 0
            beta1, beta2, beta3 = group['betas']

            for p in group['params']:
                state = self.state[p]

                if beta1 > 0.0: # save memory in case beta1 is 0.0
                    state['exp_avg'] = torch.zeros_like(p)
                else: 
                    state['exp_avg'] = None
                state['exp_avg_sq'] = torch.zeros_like(p)
                state['exp_avg_slow'] = torch.zeros_like(p)

    @staticmethod
    def schedule_alpha(t_alpha_beta3: Optional[float], step: int, alpha: float) -> float:
        if t_alpha_beta3 is None:
            return alpha
        return min(step * alpha / t_alpha_beta3, alpha)

    @staticmethod
    def schedule_beta3(t_alpha_beta3: Optional[float], step: int, beta1: float, beta3: float, eps: float) -> float:
        if t_alpha_beta3 is None:
            return beta3

        # Add eps to prevent log 0
        log_beta1, log_beta3 = math.log(beta1 + eps), math.log(beta3)

        return min(
            math.exp(
                log_beta1 * log_beta3 / ((1.0 - step / t_alpha_beta3) * log_beta3 + (step / t_alpha_beta3) * log_beta1)
            ),
            beta3,
        )
    
    @staticmethod
    def get_rms(x: torch.Tensor) -> float:
        r"""Get RMS."""
        return x.norm(2) / math.sqrt(x.numel())

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

            step = group['step']

            bias_correction1: float = self.debias(beta1, step)
            bias_correction2_sq: float = math.sqrt(self.debias(beta2, step))

            eps = group['eps']
            clip = group['clip']
            centralization = group['centralization']
            adopt = group['adopt']

            alpha_t: float = self.schedule_alpha(group['t_alpha_beta3'], step, group['alpha'])
            beta3_t: float = self.schedule_beta3(group['t_alpha_beta3'], step, beta1, beta3, eps)

            for p in group['params']:
                if p.grad is None:
                    continue
                    
                if p.grad.is_sparse:
                    raise NoSparseGradientError(str(self))
                
                p_fp32 = p
                grad = p.grad

                if p.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.to(torch.float32)
                    p_fp32 = p.to(torch.float32)
                
                state = self.state[p]

                if len(state) == 0:
                    if beta1 > 0.0: # save memory in case beta1 is 0.0
                        state['exp_avg'] = torch.zeros_like(p)
                    else: 
                        state['exp_avg'] = None
                    state['exp_avg_sq'] = torch.zeros_like(p)
                    state['exp_avg_slow'] = torch.zeros_like(p)

                # center the gradient vector
                if centralization > 0.0 and grad.dim() > 1:
                    grad.sub_(
                        grad.mean(dim=tuple(range(1, grad.dim())), keepdim=True).mul_(centralization)
                    )

                # Clip the gradient 
                if clip > 0.0:
                    grad.div_(((self.get_rms(grad) + eps) / clip).clamp_(min=1.0))

                exp_avg, exp_avg_sq, exp_avg_slow = state['exp_avg'], state['exp_avg_sq'], state['exp_avg_slow']

                if p.dtype in {torch.float16, torch.bfloat16}:
                    if beta1 > 0.0:
                        exp_avg = exp_avg.to(torch.float32)
                    exp_avg_sq, exp_avg_slow = exp_avg_sq.to(torch.float32), exp_avg_slow.to(torch.float32)

                if adopt and step == 0:
                    exp_avg_sq.add_(grad)
                else:
                    og_grad = grad
                    if not adopt:
                        exp_avg_sq.mul_(beta2).addcmul_(og_grad, og_grad, value=1.0 - beta2)
                        de_nom = (exp_avg_sq.sqrt() / bias_correction2_sq).add_(eps)
                    else:
                        de_nom = (exp_avg_sq.sqrt()).add_(eps)
                        exp_avg_sq.mul_(beta2).addcmul_(og_grad, og_grad, value=1.0 - beta2)
                        adopt_clip: float = (step-1)**0.25
                        scaled_adopt_clip = adopt_clip * de_nom
                        grad = grad.clamp(-scaled_adopt_clip, scaled_adopt_clip)

                    if beta1 > 0.0:
                        exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                    else:
                        exp_avg = grad

                    exp_avg_slow.mul_(beta3_t).add_(grad, alpha=1.0 - beta3_t)

                    update = (exp_avg.div(bias_correction1) + alpha_t * exp_avg_slow)

                    if group['update_strategy'] in {'cautious','grams'}:
                        if group['update_strategy'] == 'cautious':
                            mask = (update * grad > 0).to(grad.dtype)
                            mask.div_(mask.mean().clamp_(min=1e-3))
                            update = update * mask
                        elif group['update_strategy'] == 'grams':
                            update.copy_(torch.sign(grad) * update.abs())

                    update = update / de_nom

                    self.apply_weight_decay(
                        p=p_fp32,
                        grad=update,
                        lr=group['lr'],
                        weight_decay=group['weight_decay'],
                        weight_decouple=group['weight_decouple'],
                        fixed_decay=group['fixed_decay'],
                    )

                    p_fp32.add_(-group['lr'] * update)
                    
                if p.dtype in {torch.float16, torch.bfloat16}:
                    if beta1 > 0.0:
                        copy_stochastic_(state["exp_avg"], exp_avg)
                    copy_stochastic_(state["exp_avg_sq"], exp_avg_sq)
                    copy_stochastic_(state["exp_avg_slow"], exp_avg_slow)
                    copy_stochastic_(p, p_fp32)

        return loss

class SimplifiedAdEMAMix(BaseOptimizer):
    r"""Connections between Schedule-Free Optimizers, AdEMAMix, and Accelerated SGD Variants.

    :param params: ParamGroup. iterable of parameters to optimize or dicts defining parameter groups.
    :param lr: float. learning rate.
    :param betas: Betas. coefficients used for computing running averages of gradient and the squared hessian trace.
    :param alpha: float. coefficient for mixing the current gradient and EMA.
    :param beta1_warmup: Optional[int]. number of warmup steps used to increase beta1.
    :param min_beta1: float. minimum value of beta1 to start from.
    :param weight_decay: float. weight decay (L2 penalty).
    :param weight_decouple: bool. the optimizer uses decoupled weight decay as in AdamW.
    :param fixed_decay: bool. fix weight decay.
    :param eps: float. term added to the denominator to improve numerical stability.
    :param bias_correction1: bool. whether to use bias_correction in numerator
    :param bias_correction2: bool. whether to use bias_correction in denominator
    """

    def __init__(
        self,
        params: ParamGroup,
        lr: float|torch.Tensor = 1e-4,
        betas: Betas = (0.99, 0.95),
        weight_decay: float = 0.0,
        weight_decouple: bool = True,
        fixed_decay: bool = False,
        alpha: float = 1.0,
        beta1_warmup: Optional[int] = None,
        min_beta1: float = 0.9,
        eps: float = 1e-8,
        eps2: float = 1e-2,
        eps_floor: Optional[float] = None,
        use_orthograd: bool = False,
        adaptive_clip: Optional[float] = None,
        adaptive_clip_eps: float = 1e-3,
        adaptive_clip_type: NORM_TYPE = 'layer',
        update_strategy: UPDATE_STRATEGY = 'unmodified',
        bias_correction1: bool = False, 
        bias_correction2: bool = True,
        use_stable_spam_clipping:bool = False,
        use_adopt: bool = False,
        torch_compile: bool = False,
        sync_chunk_size: int = 128,
        state_storage_dtype: str|torch.dtype = torch.bfloat16,
        state_storage_device: str|torch.device = "cpu",
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_betas(betas)
        self.validate_non_negative(alpha, 'alpha')
        self.validate_non_negative(min_beta1, 'min_beta1')
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_non_negative(eps, 'eps')

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

        # Override zero to tiny
        if eps_floor is not None and eps_floor < eps and eps_floor <= 0:
            eps_floor = torch.finfo(torch.float32).tiny

        if update_strategy is not None and update_strategy not in {'unmodified','cautious','grams', 'both'}:
            raise ValueError("Invalid update strategy: {}".format(update_strategy))

        defaults: Defaults = {
            'lr': lr,
            'betas': betas,
            'alpha': alpha,
            'beta1_warmup': beta1_warmup,
            'min_beta1': min_beta1,
            'weight_decay': weight_decay,
            'weight_decouple': weight_decouple,
            'fixed_decay': fixed_decay,
            'eps': eps,
            'eps2': eps2,
            'eps_floor': eps_floor,
            'use_orthograd': use_orthograd,
            'adaptive_clip': adaptive_clip,
            'adaptive_clip_eps': adaptive_clip_eps,
            'adaptive_clip_type': adaptive_clip_type,
            'update_strategy': update_strategy,
            'bias_correction1': bias_correction1,
            'bias_correction2': bias_correction2,
            'use_stable_spam_clipping':use_stable_spam_clipping,
            'use_adopt':use_adopt,
            'torch_compile': torch_compile,
            'sync_chunk_size': sync_chunk_size,
            'state_storage_dtype': final_dtype,
            'state_storage_device': state_storage_device,
        }

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'SimplifiedAdEMAMix'
    
    def init_group(self, group, **kwargs) -> None:
        pass

    @torch.no_grad()
    def reset(self):
        pass

    @staticmethod
    def linear_hl_warmup_scheduler(step: int, beta_end: float, beta_start: float = 0.0, warmup: int = 1) -> float:

        def f(beta: float, eps: float = 1e-8) -> float:
            return math.log(0.5) / math.log(beta + eps) - 1.0

        def f_inv(t: float) -> float:
            return math.pow(0.5, 1.0 / (t + 1))

        if step < warmup:
            a: float = step / float(warmup)
            return f_inv((1.0 - a) * f(beta_start) + a * f(beta_end))

        return beta_end

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

            adopt_clip: float = (group['step']-1)**0.25

            beta1, beta2 = group['betas']

            use_orthograd = group['use_orthograd']
            adaptive_clip = group['adaptive_clip']
            adaptive_clip_eps = group['adaptive_clip_eps']
            adaptive_clip_type = group['adaptive_clip_type']
            update_strategy  = group['update_strategy']
            use_adopt  = group['use_adopt']

            use_stable_spam_clipping = group["use_stable_spam_clipping"]
            apply_ortho_to_group = group.get('is_ortho_group', False) # Default to False if key missing

            if group['beta1_warmup']:
                beta1 = self.linear_hl_warmup_scheduler(
                    group['step'], beta_end=beta1, beta_start=group['min_beta1'], warmup=group['beta1_warmup']
                )

            for i, p in enumerate(group["params"]):
                if p.grad is None:
                    continue

                p_fp32 = p
                grad = p.grad
                if grad.is_sparse:
                    raise NoSparseGradientError(str(self))

                state = self.state[p]
                device = p.device

                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(
                        p.data, 
                        dtype=self.state_storage_dtype, 
                        device=self.state_storage_device
                    )
                    state["exp_avg_sq"] = torch.zeros_like(
                        p.data, 
                        dtype=self.state_storage_dtype, 
                        device=self.state_storage_device
                    )

                    if self.optim_state_device == "cpu":
                        state["exp_avg"] = state["exp_avg"].pin_memory()
                        state["exp_avg_sq"] = state["exp_avg_sq"].pin_memory()

                    state['num_sum'] = 0.0
                    state['den_sum'] = 0.0

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
                exp_avg_sq = state["exp_avg_sq"].to(
                    compute_device, 
                    non_blocking=True, 
                    dtype=torch.float32
                )
                grad = grad.to(torch.float32).to(compute_device, non_blocking=True)
                p_fp32 = (
                    p.to(compute_device, dtype=torch.float32, non_blocking=True)
                )

                if apply_ortho_to_group and use_orthograd:
                    _paper_orthograd(param=p_fp32, grad=grad)

                if adaptive_clip is not None and adaptive_clip > 0:
                    grad = agc(p=p_fp32, grad=grad, agc_clip_val=adaptive_clip, agc_eps=adaptive_clip_eps, norm_type=adaptive_clip_type)

                if use_stable_spam_clipping:
                    if group['torch_compile']:
                        grad = _stable_spam_clipping_compile_wrapper(state, 
                                            grad, 
                                            step=group['step'])
                    else:
                        grad = _stable_spam_clipping_impl(state, 
                                            grad, 
                                            step=group['step'])


                curr_eps = adaptive_eps(grad, group)

                if use_adopt and group['step'] == 1:
                    exp_avg_sq.addcmul_(grad, grad)
                else:
                    exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)

                    state['num_sum'] = beta1 * state['num_sum'] + 1.0
                    state['den_sum'] = beta2 * state['den_sum'] + (1.0 - beta2)

                    if use_adopt:
                        de_nom = exp_avg_sq.sqrt().add_(math.sqrt(state['den_sum']) * curr_eps)
                        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                    else:   
                        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                        de_nom = exp_avg_sq.sqrt().add_(math.sqrt(state['den_sum']) * curr_eps)

                    update = (group['alpha'] * grad + exp_avg)

                    if update_strategy in {'cautious','grams','both'}:
                        if update_strategy in {'cautious','both'}:
                            mask = (update * grad > 0).to(grad.dtype)
                            mask.div_(mask.mean().clamp_(min=1e-3))
                            update = update * mask
                        if update_strategy in {'grams','both'}:
                            update.copy_(torch.sign(grad) * update.abs())

                    update.div_(de_nom)

                    if group['bias_correction1']:
                        update.div_(state['num_sum'])
                    if group['bias_correction2']:
                        update.mul_(math.sqrt(state['den_sum']))

                    if use_adopt:
                        update.clamp_(-adopt_clip, adopt_clip)

                    self.apply_weight_decay(
                        p=p_fp32,
                        grad=grad,
                        lr=group['lr'],
                        weight_decay=group['weight_decay'],
                        weight_decouple=group['weight_decouple'],
                        fixed_decay=group['fixed_decay'],
                    )

                    p_fp32.add_(update, alpha=-group['lr'])

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
                        copy_stochastic_(state["exp_avg_sq"], exp_avg_sq)
                    else:
                        state["exp_avg"].copy_(exp_avg, non_blocking=True)
                        state["exp_avg_sq"].copy_(exp_avg_sq, non_blocking=True)

                # ========= Check if we need to synchronize =========
                # We synchronize after processing a chunk of parameters.
                # The (i + 1) ensures we sync after the 1st, 2nd, ... chunk.
                if (i + 1) % self.sync_chunk_size == 0:
                    torch.cuda.synchronize()

            # Final synchronization to handle the last partial chunk
            # This ensures all operations for the group are complete before exiting.
            torch.cuda.synchronize()

        return loss
    
class SimplifiedAdEMAMixExM(BaseOptimizer):
    r"""Connections between Schedule-Free Optimizers, AdEMAMix, and Accelerated SGD Variants.

    :param params: ParamGroup. iterable of parameters to optimize or dicts defining parameter groups.
    :param lr: float. learning rate.
    :param betas: Betas. coefficients used for computing running averages of gradient and the squared hessian trace.
    :param alpha: float. coefficient for mixing the current gradient and EMA.
    :param beta1_warmup: Optional[int]. number of warmup steps used to increase beta1. Recommend setting to iteration/step count.
    :param min_beta1: float. minimum value of beta1 to start from.
    :param weight_decay: float. weight decay (L2 penalty).
    :param weight_decouple: bool. the optimizer uses decoupled weight decay as in AdamW.
    :param eps: float. term added to the denominator to improve numerical stability.
    """

    def __init__(
        self,
        params: ParamGroup,
        lr: float|torch.Tensor = 2e-4,
        betas: Betas = (0.95, 0.997),
        min_beta1: float = 0.95,
        beta1_warmup: Optional[int] = None,
        weight_decay: float = 0.0,
        weight_decouple: bool = True,
        alpha: float = 1.0,
        eps: float = 1e-8,
        eps_floor: Optional[float] = 1e-12,
        use_orthograd: bool = True,
        update_strategy: UPDATE_STRATEGY = 'unmodified',
        update_strategy_scale: float = 1.0,
        use_stable_spam_clipping:bool = True,
        use_compass: bool = False,
        use_adabelief: bool = True,
        use_newton_schulz: bool = True,
        amsgrad_min_decay_rate: float = 0.98,
        amsgrad_max_decay_rate: float = 0.98,
        torch_compile: bool = True,
        sync_chunk_size: int = 128,
        state_storage_dtype: str|torch.dtype = torch.bfloat16,
        state_storage_device: str|torch.device = "cpu",
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_betas(betas)
        self.validate_non_negative(alpha, 'alpha')
        self.validate_non_negative(min_beta1, 'min_beta1')
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_non_negative(eps, 'eps')

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

        self.chunk_size = sync_chunk_size
        self.optim_state_dtype = final_dtype
        self.optim_state_device = state_storage_device

        if not (0.0 <= update_strategy_scale <= 1.0):
            raise ValueError(f"update_strategy_scale ({update_strategy_scale}) must lie in [0.0, 1.0].")
        
        # Override zero to tiny
        if eps_floor is not None and eps_floor < eps and eps_floor <= 0:
            eps_floor = torch.finfo(torch.float32).tiny

        if update_strategy is not None and update_strategy not in {'unmodified','cautious','grams', 'both'}:
            raise ValueError("Invalid update strategy: {}".format(update_strategy))

        defaults: Defaults = {
            'lr': lr,
            'betas': betas,
            'alpha': alpha,
            'beta1_warmup': beta1_warmup,
            'min_beta1': min_beta1,
            'weight_decay': weight_decay,
            'weight_decouple': weight_decouple,
            'eps': eps,
            'eps2': 1e-2,
            'eps_floor': eps_floor,
            'use_orthograd': use_orthograd,
            'update_strategy': update_strategy,
            'update_strategy_scale': update_strategy_scale,
            'use_stable_spam_clipping':use_stable_spam_clipping,
            'use_compass': use_compass,
            'use_adabelief': use_adabelief,
            'torch_compile': torch_compile,
            'amsgrad_max_decay_rate': amsgrad_max_decay_rate,
            'amsgrad_min_decay_rate': amsgrad_min_decay_rate,
            'use_newton_schulz':use_newton_schulz,
            'chunk_size': sync_chunk_size,
            'dtype': final_dtype,
            'storage_device':state_storage_device,
        }

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'SimplifiedAdEMAMixExM'
    
    def init_group(self, group, **kwargs) -> None:
        pass

    @torch.no_grad()
    def reset(self):
        pass

    @staticmethod
    def linear_hl_warmup_scheduler(step: int, beta_end: float, beta_start: float = 0.0, warmup: int = 1) -> float:

        def f(beta: float, eps: float = 1e-8) -> float:
            return math.log(0.5) / math.log(beta + eps) - 1.0

        def f_inv(t: float) -> float:
            return math.pow(0.5, 1.0 / (t + 1))

        if step < warmup:
            a: float = step / float(warmup)
            return f_inv((1.0 - a) * f(beta_start) + a * f(beta_end))

        return beta_end

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

            step = group['step']

            adopt_clip: float = (step-1)**0.25

            beta1, beta2 = group['betas']

            use_orthograd = group['use_orthograd']
            use_compass = group['use_compass']
            use_adabelief = group['use_adabelief']
            use_newton_schulz = group['use_newton_schulz']
            update_strategy  = group['update_strategy']
            update_strategy_scale  = group['update_strategy_scale']
            amsgrad_min_decay_rate  = group['amsgrad_min_decay_rate']
            amsgrad_max_decay_rate  = group['amsgrad_max_decay_rate']
            torch_compile = group['torch_compile']

            use_stable_spam_clipping = group["use_stable_spam_clipping"]
            apply_ortho_to_group = group.get('is_ortho_group', False) # Default to False if key missing

            eps_floor = group['eps_floor']

            if group['beta1_warmup']:
                beta1 = self.linear_hl_warmup_scheduler(
                    step, beta_end=beta1, beta_start=group['min_beta1'], warmup=group['beta1_warmup']
                )

            beta2 = ((beta2 ** step - beta2) / (beta2 ** step - 1.0))

            bias_correction1 = 1 - beta1 ** step
            bias_correction2_sqrt = (1 - beta2 ** step) ** (1/2)

            for i, p in enumerate(group["params"]):
                if p.grad is None:
                    continue

                p_fp32 = p
                grad = p.grad
                device = p.device
                if grad.is_sparse:
                    raise NoSparseGradientError(str(self))

                state = self.state[p]

                if len(state) == 0:
                    if self.optim_state_device == "cpu":
                        state["exp_avg"] = torch.zeros_like(
                            p.data, dtype=self.optim_state_dtype, device=self.optim_state_device
                        ).pin_memory()
                        state["exp_avg_sq"] = torch.zeros_like(
                            p.data, dtype=self.optim_state_dtype, device=self.optim_state_device
                        ).pin_memory()
                    else:
                        state["exp_avg"] = torch.zeros_like(
                            p.data, dtype=self.optim_state_dtype, device=self.optim_state_device
                        )
                        state["exp_avg_sq"] = torch.zeros_like(
                            p.data, dtype=self.optim_state_dtype, device=self.optim_state_device
                        )

                # ========= Asynchronously queue all operations for this parameter =========
                # Determine target GPU device for computation
                if device.type == "cpu":
                    # If param is on CPU, use default GPU for computation
                    compute_device = torch.cuda.current_device()
                else:
                    # If param is on GPU, use its device
                    compute_device = device

                # 1. Queue Host-to-Device copy
                exp_avg = state["exp_avg"].to(
                    compute_device, non_blocking=True, dtype=torch.float32
                )
                exp_avg_sq = state["exp_avg_sq"].to(
                    compute_device, non_blocking=True, dtype=torch.float32
                )

                grad = grad.to(torch.float32).to(compute_device, non_blocking=True)
                p_fp32 = (
                    p.to(compute_device, dtype=torch.float32, non_blocking=True)
                )

                if apply_ortho_to_group and use_orthograd:
                    _paper_orthograd(param=p_fp32, grad=grad)

                if use_stable_spam_clipping:
                    if torch_compile:
                        grad = _stable_spam_clipping_compile_wrapper(state, 
                                            grad, 
                                            step=step,
                                            eps=eps_floor)
                    else:
                        grad = _stable_spam_clipping_impl(state, 
                                            grad, 
                                            step=step,
                                            eps=eps_floor)

                # Calculate RMS of grad once
                rms_grad = torch.sqrt(torch.mean(grad.pow(2)))
                curr_eps = adaptive_eps(grad, group, rms_grad=rms_grad)

                # RMS Norm
                grad_normed = grad.div(rms_grad.clamp_min_(1))

                if use_newton_schulz:
                    if grad_normed.ndim > 0:
                        if torch_compile:
                            grad_normed = zero_power_via_newton_schulz_6_compile(grad_normed)
                        else:
                            grad_normed = zero_power_via_newton_schulz_6(grad_normed)
                    elif grad_normed.numel() > 1:
                        if torch_compile:
                            grad_normed = bias_rms_compile(grad_normed)
                        else:
                            grad_normed = bias_rms(grad_normed)
                            
                # Adaptive ema
                mask = (grad_normed * exp_avg > 0).to(grad_normed.dtype)
                mask.clamp_min_(beta1)
                mask.div_(mask.mean().clamp_(min=1e-3)) # Divide by mean (0.001-1.0)
                exp_avg.mul_(mask)

                exp_avg.mul_(beta1).add_(grad_normed, alpha=1.0 - beta1)

                # Compass amplification + beta1 Bias correction
                if use_compass:
                    bias_corrected_axp_avg = exp_avg.div(bias_correction1)
                    c_t = grad_normed.add(bias_corrected_axp_avg, alpha=group['alpha'])
                else:
                    c_t = grad_normed

                if step == 1:
                    if use_compass:
                        # Try adding residual to c_t
                        grad_residual = c_t.add(grad_normed.add(bias_corrected_axp_avg, alpha=-1))
                    else:
                        grad_residual = grad_normed - exp_avg
                    exp_avg_sq.addcmul_(grad_residual, grad_residual)
                else:
                    de_nom = exp_avg_sq.sqrt().div_(bias_correction2_sqrt).add_(curr_eps)

                    if use_adabelief:
                        if use_compass:
                            # Try adding residual to c_t
                            grad_residual = c_t.add(grad_normed.add(bias_corrected_axp_avg, alpha=-1))
                        else:
                            grad_residual = grad_normed - exp_avg
                        new_exp_avg_sq = exp_avg_sq.mul(beta2).addcmul_(grad_residual, grad_residual, value=1.0 - beta2)
                    else:
                        new_exp_avg_sq = exp_avg_sq.mul(beta2).addcmul_(c_t, c_t, value=1.0 - beta2)

                    # Decaying amsgrad
                    torch.maximum(exp_avg_sq.mul(max(min(beta2, amsgrad_max_decay_rate), amsgrad_min_decay_rate)), new_exp_avg_sq, out=exp_avg_sq)

                    if use_compass:
                        update = c_t
                    else:
                        update = (group['alpha'] * grad_normed + exp_avg)

                    update = apply_update_strategies(update, grad_normed, update_strategy, update_strategy_scale)

                    update.div_(de_nom)

                    if not use_compass:
                        update.div_(bias_correction1)

                    update.clamp_(-adopt_clip, adopt_clip)

                    self.apply_weight_decay(
                        p=p_fp32,
                        grad=grad_normed,
                        lr=group['lr'],
                        weight_decay=group['weight_decay'],
                        weight_decouple=group['weight_decouple'],
                        fixed_decay=False,
                    )

                    p_fp32.add_(update, alpha=-group['lr'])

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
                if self.optim_state_dtype == torch.bfloat16:
                    copy_stochastic_(state["exp_avg"], exp_avg)
                    copy_stochastic_(state["exp_avg_sq"], exp_avg_sq)
                else:
                    state["exp_avg"].copy_(exp_avg, non_blocking=True)
                    state["exp_avg_sq"].copy_(exp_avg_sq, non_blocking=True)

                # ========= Check if we need to synchronize =========
                # We synchronize after processing a chunk of parameters.
                # The (i + 1) ensures we sync after the 1st, 2nd, ... chunk.
                if (i + 1) % self.chunk_size == 0:
                    torch.cuda.synchronize()

            # Final synchronization to handle the last partial chunk
            # This ensures all operations for the group are complete before exiting.
            torch.cuda.synchronize()

        return loss

@torch.no_grad()
def apply_update_strategies(update, grad, update_strategy, scale=1.0):
    """
    Applies update strategies with scaling factors.

    Args:
        update (torch.Tensor): The current update tensor to be modified.
        grad_normed (torch.Tensor): The normalized gradient.
        update_strategy (str): One of 'cautious', 'grams', 'both'.
        scale (float): Scaling factor for the Grams strategies.

    Returns:
        torch.Tensor: The modified update tensor.
    """
    if scale > 0 and update_strategy in {'cautious', 'grams', 'both'}:
        if update_strategy in {'cautious', 'both'}:
            if scale >= 1.0:
                update_before_cautious = update

                # 1. Calculate the "fully cautious" update
                mask = (update_before_cautious * grad > 0).to(grad.dtype)
                mask_mean = mask.mean().clamp_(min=1e-3) # Avoid division by zero or tiny numbers
                mask.div_(mask_mean)
                update = update.mul(mask)
            else:
                update_before_cautious = update

                # 1. Calculate the "fully cautious" update
                mask = (update_before_cautious * grad > 0).to(grad.dtype)
                mask_mean = mask.mean().clamp_(min=1e-3) # Avoid division by zero or tiny numbers
                mask.div_(mask_mean)

                update_if_fully_cautious = update_before_cautious * mask

                update = (1 - scale) * update_before_cautious + scale * update_if_fully_cautious

        if update_strategy in {'grams', 'both'}:
            if scale >= 1.0:
                update = torch.sign(grad).mul_(update.abs())
            else:
                update_before_grams = update

                update_if_fully_grams = torch.sign(grad).mul_(update_before_grams.abs())

                update = (1 - scale) * update_before_grams + scale * update_if_fully_grams

    return update