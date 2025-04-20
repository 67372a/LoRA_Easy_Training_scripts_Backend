import torch
from typing import Tuple, Union, Type, Literal, Optional, Dict, Any
from torch.nn import Parameter, ParameterList
from torch.optim import SGD, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LRScheduler
import math
import inspect
import logging

OPTIMIZER = Type[Optimizer]

NORM_TYPE = Literal['unit','global','layer']

CLIP_TYPE = Literal['unit','layer','element']

STATE_PRECISION = Literal['parameter', 'q4bit', 'q8bit', 'qfp8']

UPDATE_STRATEGY = Literal['unmodified','cautious','grams','both']

def unit_norm(x: torch.Tensor, norm: float = 2.0) -> torch.Tensor:
    r"""Get norm of unit."""
    keep_dim: bool = True
    dim: Optional[Union[int, Tuple[int, ...]]] = None

    x_len: int = len(x.shape)
    if x_len <= 1:
        keep_dim = False
    elif x_len in (2, 3):
        dim = 1
    elif x_len == 4:
        dim = (1, 2, 3)
    else:
        dim = tuple(range(1, x_len))

    return x.norm(p=norm, dim=dim, keepdim=keep_dim)

def unit_norm_logging(x: torch.Tensor, norm: float = 2.0):
    r"""Get norm of unit."""
    keep_dim: bool = True
    dim: Optional[Union[int, Tuple[int, ...]]] = None

    x_len: int = len(x.shape)
    if x_len <= 1:
        keep_dim = False
    elif x_len in (2, 3):
        dim = 1
    elif x_len == 4:
        dim = (1, 2, 3)
    else:
        dim = tuple(range(1, x_len))

    logging.info(f"unit_norm shape={str(x.shape)}")
    logging.info(f"unit_norm norms={str(torch.norm(x, p=norm, dim=dim, keepdim=keep_dim))}")


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
        target.copy_(result.view(dtype=torch.float32))
    
def agc(p: torch.Tensor, 
        grad: torch.Tensor, 
        agc_clip_val: float, 
        agc_eps: float = 1e-3, 
        eps: Optional[float] = None, 
        norm_type: NORM_TYPE = 'layer') -> torch.Tensor:
    r"""Clip gradient values in excess of the norm.
        Clip updates to be at most clipping * parameter_norm.

    References:
        [Brock, Smith, De, Simonyan 2021] High-Performance Large-Scale Image
        Recognition Without Normalization.
        
    :param p: torch.Tensor. parameter.
    :param grad: torch.Tensor, gradient.
    :param agc_eps: float. Effectively sets a floor for the p_norm, as such, any gradients smaller than this will be clipped
        as though their parameter is at least agc_eps. This helps prevent vanishing gradients and excessive clipping early in training
        for small parameters.
    :param agc_clip_val: float. The desired clipping ratio, e.x. 0.5 would mean any gradient would be clipped to be no greater than half it's
        associated parameter.
    :param eps: float. simple stop from div by zero, as such should be as small as possible to avoid skewing clipping.
    """

    if agc_eps is None or agc_eps == 0.0:
        agc_eps = torch.finfo(torch.float32).tiny    

    if eps is None or eps == 0.0:
        eps = torch.finfo(torch.float32).tiny

    if norm_type in {'global','layer'}:
        # Compute the global norm of the parameters and gradients
        p_norm = torch.norm(p).clamp_(min=agc_eps)
        g_norm = torch.norm(grad)

        # Compute the maximum allowed norm for the gradients
        max_norm = (p_norm * agc_clip_val).clamp(min=eps)

        # Compute the clipping coefficient
        clip_coef = min(1, max_norm / g_norm.clamp(min=eps))

        # Scale the gradients holistically
        grad = grad * clip_coef

        return grad
    elif norm_type == 'unit':
        p_norm = unit_norm(p).clamp_(min=agc_eps)
        g_norm = unit_norm(grad)

        max_norm = (p_norm * agc_clip_val).clamp(min=eps)

        clipped_grad = grad * (max_norm / g_norm.clamp_(min=eps))

        return torch.where(g_norm > max_norm, clipped_grad, grad)
    else:
        raise ValueError(f"'{norm_type}' is not a supported value for norm_type.")


def schedule_alpha(t_alpha: Optional[float], step: int, alpha: float) -> float:
    if t_alpha is None:
        return alpha
    return min(step * alpha / t_alpha, alpha)


def schedule_beta(t_beta: Optional[float], step: int, beta_initial: float, beta_final: float, eps: float = 1e-8) -> float:
    if t_beta is None:
        return beta_initial

    # Add eps to prevent log 0
    log_beta_intial, log_beta_final = math.log(max(beta_initial, eps)), math.log(beta_final)

    return min(
        math.exp(
            log_beta_intial * log_beta_final / ((1.0 - step / t_beta) * log_beta_final + (step / t_beta) * log_beta_intial)
        ),
        beta_final,
    )

def schedule_beta_tc(t_beta: Optional[float], step: int, beta_initial: float, beta_final: float, eps: float = 1e-8) -> float:
    if t_beta is None:
        return beta_initial

    # Add eps to prevent log 0
    log_beta_intial, log_beta_final = math.log(max(beta_initial, eps)), math.log(beta_final)

    return min(
        torch.exp(
            log_beta_intial * log_beta_final / ((1.0 - step / t_beta) * log_beta_final + (step / t_beta) * log_beta_intial)
        ),
        beta_final,
    )

@torch.no_grad()
def spam_grad_clipping(grad: torch.Tensor, 
                       second_moment: torch.Tensor, 
                       clip_threshold: float, 
                       clip_type: CLIP_TYPE = 'element', 
                       spam_clip_eps: float = 1e-37) -> torch.Tensor:
    if spam_clip_eps is None or spam_clip_eps == 0:
        spam_clip_eps = torch.finfo(torch.float32).tiny
    
    if clip_type in {'unit', 'element'}:
        # Calculate the clipping condition
        second_momentum_threshold = second_moment.mul(clip_threshold).add(spam_clip_eps)
        second_momentum_threshold_sqrt = torch.sqrt(second_momentum_threshold)
        sign_grad = grad.sign()

        # Use torch.where instead of boolean masking
        return torch.where(
            grad.square() > second_momentum_threshold,
            sign_grad * second_momentum_threshold_sqrt,
            grad
        )
    elif clip_type == 'layer':
        # Calculate the global gradient norm
        max_norm = torch.norm(torch.sqrt(second_moment * clip_threshold))
        grad_norm = torch.norm(grad)

        # Calculate scaling factor for clipping
        scale = torch.where(
            grad_norm > max_norm,
            max_norm / grad_norm,
            torch.ones_like(grad_norm)
        )

        # Apply scaling to gradient
        return grad * scale
    
def spam_grad_clipping_logging(grad: torch.Tensor, 
                               second_moment: torch.Tensor, 
                               clip_threshold: float, 
                               clip_type: str = 'element', 
                               spam_clip_eps: float = 1e-37) -> torch.Tensor:
    if spam_clip_eps is None or spam_clip_eps == 0:
        spam_clip_eps = torch.finfo(torch.float32).tiny

    if clip_type in {'unit', 'element'}:
        # Calculate the clipping condition
        second_momentum_threshold = second_moment.mul(clip_threshold).add(spam_clip_eps)
        second_momentum_threshold_sqrt = torch.sqrt(second_momentum_threshold)
        
        # Check where scaling will occur
        scaling_mask = grad.square() > second_momentum_threshold
        total_elements = grad.numel()
        
        if scaling_mask.any():
            # Calculate scaling ratios for logging
            original_values = grad[scaling_mask].abs()  # Use absolute values
            scaled_values = second_momentum_threshold_sqrt[scaling_mask]
            
            # Add small epsilon to prevent division by zero
            scaling_ratios = scaled_values / (original_values.add(spam_clip_eps))
            
            # Add more detailed logging
            logging.info(
                f"Total elements {total_elements}. "
                f"Unit-wise gradient clipping applied to {scaling_mask.sum().item()} elements. "
                f"\nOriginal values - Mean: {original_values.mean().item():.6f}, Max: {original_values.max().item():.6f}. "
                f"\nScaled values - Mean: {scaled_values.mean().item():.6f}, Max: {scaled_values.max().item():.6f}. "
                f"\nScaling ratios - Mean: {scaling_ratios.mean().item():.6f}, Max: {scaling_ratios.max().item():.6f}"
            )
        
    elif clip_type == 'layer':
        # Calculate the global gradient norm
        max_norm = torch.norm(torch.sqrt(second_moment * clip_threshold))
        grad_norm = torch.norm(grad)
        
        # Calculate scaling factor
        scale = torch.where(
            grad_norm > max_norm,
            max_norm / grad_norm,
            torch.ones_like(grad_norm)
        )
        
        # Log if scaling is applied
        if grad_norm > max_norm:
            logging.info(
                f"Layer-wise gradient clipping applied. "
                f"Gradient norm: {grad_norm.item():.4f}, "
                f"Max norm: {max_norm.item():.4f}, "
                f"Scaling factor: {scale.item():.4f}"
            )
    

# Modified Adafactor factorisation implementation by Ross Wightman 
# https://github.com/huggingface/pytorch-image-models/pull/2320
@torch.no_grad()
def create_factored_dims(
    shape,
    factored: bool,
    min_dim_size_to_factor: int):
    r"""Whether to use a factored second moment estimator.
    This function returns a tuple with the two largest axes to reduce over.
    If all dimensions have size < min_dim_size_to_factor, return None.
    Args:
    shape: an input shape
    factored: whether to use factored second-moment estimator for > 2d vars.
    min_dim_size_to_factor: only factor accumulator if all array dimensions are greater than this size.
    Returns:
    None or a tuple of ints
    """
    if not factored or len(shape) < 2:
        return None
    if all(dim < min_dim_size_to_factor for dim in shape):
        return None
    sorted_dims = sorted(((x, i) for i, x in enumerate(shape)))
    return int(sorted_dims[-2][1]), int(sorted_dims[-1][1])
    
# https://github.com/LoganBooker/prodigy-plus-schedule-free/blob/23f752a3901686d270dfdcb9b29823541ad1c3c7/prodigyplus/core_optimiser.py#L389
@torch.no_grad()
def get_denom(second_moment: torch.tensor, eps: float = 1e-16):
    # Get denom
    if isinstance(second_moment, list):
        row_var, col_var, _, _, reduce_dc = second_moment

        row_col_mean = row_var.mean(dim=reduce_dc, keepdim=True).add_(eps)
        row_factor = row_var.div(row_col_mean).sqrt_()
        col_factor = col_var.sqrt()
        denom = row_factor * col_factor
    else:
        denom = second_moment.sqrt()

    return denom
    
# https://github.com/LoganBooker/prodigy-plus-schedule-free/blob/23f752a3901686d270dfdcb9b29823541ad1c3c7/prodigyplus/core_optimiser.py#L411
@torch.no_grad()
def update_second_moment(second_moment: torch.tensor, grad: torch.tensor, beta2: float, adopt_first: bool = False) -> torch.tensor:
    # EMA updates
    if isinstance(second_moment, list):
        row_var, col_var, dr, dc, _ = second_moment
        if adopt_first:
            row_var.copy_(
                grad.norm(dim=dr, keepdim=True).square_().div_(grad.shape[dr])
            )
            col_var.copy_(
                grad.norm(dim=dc, keepdim=True).square_().div_(grad.shape[dc])
            )
        else:
            row_var.lerp_(
                grad.norm(dim=dr, keepdim=True).square_().div_(grad.shape[dr]),
                weight=1 - beta2
            )
            col_var.lerp_(
                grad.norm(dim=dc, keepdim=True).square_().div_(grad.shape[dc]),
                weight=1 - beta2
            )
    else:
        if adopt_first:
            second_moment.addcmul_(grad, grad)
        else:
            second_moment.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

    return second_moment

# Implementation from: https://github.com/LucasPrietoAl/grokking-at-the-edge-of-numerical-stability/blob/main/orthograd.py
@torch.no_grad()
def orthograd(param: torch.tensor, grad: torch.tensor, eps: float = 1e-30):
    if eps is None or eps == 0.0:
        eps = torch.finfo(torch.float32).tiny

    return torch.where(param.norm(2) <= eps,
                       grad,
                       _orthograd(param, grad, eps))

@torch.no_grad()
def _orthograd(param: torch.tensor, 
               grad: torch.tensor, 
               eps: float = 1e-30):
    if eps is None or eps == 0.0:
        eps = torch.finfo(torch.float32).tiny

    if not param.numel() > 1:
        return grad
    
    grad_shape = grad.shape
    w = param.view(-1)
    grad = grad.view(-1)

    # Perturb to prevent perfect alignment
    w_perturbed = w.clone().add_(eps)

    proj = torch.dot(w_perturbed, grad) / torch.dot(w_perturbed, w_perturbed)
    g_orth = grad.to(dtype=torch.float32, copy=True).add_(w, alpha=-proj)
    g_orth_scaled = g_orth.mul_(grad.norm(2) / (g_orth.norm(2) + eps))

    return g_orth_scaled.view(grad_shape)


# Implementation from: https://github.com/LucasPrietoAl/grokking-at-the-edge-of-numerical-stability/blob/main/orthograd.py
@torch.no_grad()
def orthograd_atan(param: torch.Tensor, grad: torch.Tensor):
    grad_shape = grad.shape
    w = param.view(-1)
    grad = grad.view(-1)

    proj = torch.dot(w, grad).atan2_(torch.dot(w, w)).mul_(1.27323954474)
    g_orth = grad.to(dtype=torch.float32, copy=True).sub_(w, alpha=proj)
    g_orth_scaled = g_orth.mul_(grad.norm(2).div_(g_orth.norm(2).clamp_(min=1e-6)))

    return g_orth_scaled.view(grad_shape)

def clean_dict_params(func, params_dict, wrapped=False):
    """
    Remove dictionary keys that don't match function parameters and warn about removals.
    
    Args:
        func: The function to check parameters against
        params_dict: Dictionary of parameters to clean
        
    Returns:
        dict: New dictionary with only valid parameters
    """
    # Get the function's signature
    sig = inspect.signature(func)
    
    # Create a new dict with only valid parameters
    valid_params = {}
    
    for key, value in params_dict.items():
        if key in sig.parameters:
            valid_params[key] = value
        else:
            print(f"Parameter '{key}' is not a valid parameter for the {'wrapped ' if wrapped else ''}optimizer and will be ignored.")
    
    return valid_params

class CosineDecay:
    """
    Applies cosine decay to a parameter (death_rate), using PyTorch's built-in
    `torch.optim.lr_scheduler.CosineAnnealingLR`.

    Args:
        death_rate (float): Initial value to be decayed.
        T_max (int): Maximum number of iterations for the decay.
        eta_min (float, optional): Minimum value of the parameter after decay.
            Defaults to 0.
        last_epoch (int, optional): The index of the last epoch. Defaults to -1.
    """

    def __init__(self, death_rate: float, T_max: int, eta_min: float = 0, last_epoch: int = -1):
        self.sgd = torch.optim.SGD(
            torch.nn.ParameterList([torch.nn.Parameter(torch.zeros(1))]),
            lr=death_rate,
        )
        self.cosine_stepper = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.sgd, T_max + 1, eta_min, last_epoch
        )
        self.T_max = T_max
        self.eta_min = eta_min

    def step(self, current_step: int) -> None:
        """
        Performs one step of the cosine decay scheduler.

        Args:
            current_step (int): Current step index.
        """
        self.cosine_stepper.step(current_step)

    def get_dr(self, current_step: int) -> float:
        """
        Returns the updated rate (death_rate) at the given step.

        Args:
            current_step (int): Current step index.

        Returns:
            float: The decayed parameter.
        """
        if current_step >= self.T_max:
            return self.eta_min
        self.step(current_step)
        return self.sgd.param_groups[0]["lr"]
    
class SSCCosineDecay:
    r"""Applies cosine decay to a parameter (death_rate), using PyTorch's built-in `CosineAnnealingLR`.

    :param death_rate: float. initial value to be decayed.
    :param t_max: int. maximum number of iterations for the decay.
    :param eta_min: Optional[float]. minimum value of the parameter after decay. defaults to 0.
    :param last_epoch: Optional[int]. the index of the last epoch. Defaults to -1.
    """

    def __init__(self, death_rate: float, t_max: int, eta_min: float = 0.0, last_epoch: int = -1):
        self.sgd: Optimizer = SGD(ParameterList([Parameter(torch.zeros(1))]), lr=death_rate)
        self.cosine_stepper: LRScheduler = CosineAnnealingLR(self.sgd, t_max + 1, eta_min, last_epoch)
        self.t_max = t_max
        self.eta_min = eta_min

    def step(self, current_step: int) -> None:
        r"""One step of the cosine decay scheduler.

        :param current_step: int. Current step index.
        """
        self.cosine_stepper.step(current_step)

    def get_death_rate(self, current_step: int) -> float:
        r"""Get the updated rate (death_rate) at the given step.

        :param current_step: int. Current step index.
        """
        if current_step >= self.t_max:
            return self.eta_min

        self.step(current_step)

        return self.sgd.param_groups[0]['lr']
    
@torch.no_grad()
def stable_spam_clipping(state: dict, 
                         grad: torch.tensor, 
                         step: int, 
                         scale: float = 1.0, 
                         eps: float = 1e-8, 
                         gamma1: float = 0.85, 
                         gamma2: float = 0.99999, 
                         gamma3: float = 0.999):    
    if eps is None or eps == 0.0:
        eps = torch.finfo(torch.float32).tiny

    if 'ssc_m_norm_t' not in state:
        state['ssc_m_norm_t'] = 0.0
        state['ssc_v_norm_t'] = 0.0
        state['ssc_m_max_t'] = 0.0

    m_max_t = state['ssc_m_max_t']

    max_grad = torch.max(grad.abs())

    m_max_t = gamma3 * m_max_t + (1 - gamma3) * max_grad

    state["ssc_m_max_t"] = m_max_t

    m_max_hat = m_max_t / (1.0 - gamma3 ** step)

    mask = grad.abs() > m_max_hat
    if mask.sum() > 0:
        grad[mask] = grad[mask] / max_grad * m_max_hat

    grad_norm = torch.norm(grad)

    m_norm_t, v_norm_t = state['ssc_m_norm_t'], state['ssc_v_norm_t']

    m_norm_t = gamma1 * scale * m_norm_t + (1 - gamma1 * scale) * grad_norm
    v_norm_t = gamma2 * v_norm_t + (1 - gamma2) * grad_norm**2

    m_norm_hat = m_norm_t / (1.0 - (gamma1 * scale) ** step)
    v_norm_hat = v_norm_t / (1.0 - gamma2 ** step)

    c_norm_t = m_norm_hat / (torch.sqrt(v_norm_hat) + eps)

    if grad_norm > 0:
        grad = grad / grad_norm * c_norm_t

    state["ssc_m_norm_t"], state["ssc_v_norm_t"] = m_norm_t, v_norm_t

    return grad

@torch.no_grad()
def stable_spam_clipping_tensors(
    ssc_m_norm_t: torch.tensor,
    ssc_v_norm_t: torch.tensor,
    ssc_m_max_t: torch.tensor, 
    grad: torch.tensor, 
    step: int, 
    scale: float = 1.0, 
    eps: float = 1e-8, 
    gamma1: float = 0.85, 
    gamma2: float = 0.99999, 
    gamma3: float = 0.999):    

        m_max_t = ssc_m_norm_t

        max_grad = torch.max(grad.abs())

        m_max_t = gamma3 * m_max_t + (1 - gamma3) * max_grad

        ssc_m_norm_t.copy_(m_max_t)

        m_max_hat = m_max_t / (1.0 - gamma3 ** step)

        grad = torch.where((grad.abs() > m_max_hat).sum() > 0,
                           grad / max_grad * m_max_hat,
                           grad)

        grad_norm = torch.norm(grad)

        m_norm_t, v_norm_t = ssc_m_norm_t, ssc_m_max_t

        m_norm_t = gamma1 * scale * m_norm_t + (1 - gamma1 * scale) * grad_norm
        v_norm_t = gamma2 * v_norm_t + (1 - gamma2) * grad_norm**2

        ssc_m_norm_t.copy_(m_norm_t)
        ssc_v_norm_t.copy_(v_norm_t)

        m_norm_hat = m_norm_t / (1.0 - (gamma1 * scale) ** step)
        v_norm_hat = v_norm_t / (1.0 - gamma2 ** step)

        c_norm_t = m_norm_hat / (torch.sqrt(v_norm_hat) + eps)

        grad = torch.where(grad_norm > 0,
                           grad / grad_norm * c_norm_t,
                           grad)

        return grad


# From: https://github.com/KellerJordan/Muon/blob/master/muon.py
@torch.no_grad()
def newton_schulz_(grad, steps=6, eps=1e-12):
    if eps is None or eps == 0.0:
        eps = torch.finfo(torch.float32).tiny

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

    X = grad.to(dtype=torch.bfloat16, copy=True)
    if grad.size(0) > grad.size(1):
        X = X.T

    X /= X.norm().add(eps) # ensure top singular value <= 1
    for a,b,c in abc_list:
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X

    if grad.size(0) > grad.size(1):
        X = X.T

    # Gradient scaling adaptation from: https://github.com/leloykun/adaptive-muon
    X = torch.einsum('ij,ij->', grad.type_as(X), X).clamp(-1.0, 1.0) * X
    grad.copy_(X)
    del X

    return grad.view(G_shape)

@torch.no_grad()
def adagc_global_clipping_calc(
        self,
        step: int,
        warmup_steps: int = 0,
        lambda_abs: float = 1.0,
        eps: float = 1e-8
) -> torch.Tensor:
        # --- Global Clipping Calculation (outside compiled step) ---
        global_norm_device = 'cpu'
        has_grad = False
        for group in self.param_groups:
             for p in group['params']:
                 if p.grad is not None:
                    has_grad = True
                    global_norm_device = p.grad.device # Use device of first grad found
                    break
             if global_norm_device != 'cpu': break # Found a device

        if step <= warmup_steps and warmup_steps > 0 and has_grad:
            # Calculate total squared global norm of gradients
            global_norm_sq_fp32 = torch.tensor(0.0, dtype=torch.float32, device=global_norm_device)
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is not None:
                        global_norm_sq_fp32.add_(p.grad.float().pow(2).sum())

            if global_norm_sq_fp32 > 0: # Avoid division by zero
                 global_norm_fp32 = torch.sqrt(global_norm_sq_fp32)
                 eps_fp32 = torch.tensor(eps, dtype=torch.float32, device=global_norm_device)
                 global_clip_factor_fp32 = torch.tensor(lambda_abs, dtype=torch.float32, device=global_norm_device) / (global_norm_fp32 + eps_fp32)
                 global_clip_factor_fp32 = torch.min(global_clip_factor_fp32, torch.tensor(1.0, device=global_norm_device, dtype=torch.float32))
            else:
                 # If global norm is 0, no clipping is needed, factor is 1.0
                 global_clip_factor_fp32 = torch.tensor(1.0, device=global_norm_device, dtype=torch.float32) # Put on device
        else:
             # If not in warm-up or no grads, global clip factor is 1.0 (ensure it's on a device if possible)
             device = global_norm_device # Use the device found earlier, or 'cpu'
             global_clip_factor_fp32 = torch.tensor(1.0, device=device, dtype=torch.float32)

        return global_clip_factor_fp32

@torch.no_grad()
def _apply_adagc_clipping_and_update_gamma(
    self,
    grad: torch.Tensor,
    state: Dict[str, Any],
    step: int,
    warmup_steps: int = 0,
    lambda_rel: float = 1.05,
    ema_beta: float = 0.98,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Applies AdaGC or global clipping to the gradient and updates the gamma state.
    Returns the clipped gradient as an FP32 tensor
    """
    with torch.no_grad():
        grad_fp32 = grad.float()


        device = grad_fp32.device
        # Get gamma state (always FP32 scalar tensor)
        if 'adagc_gamma' not in state:
            state['adagc_gamma'] = torch.tensor(lambda_rel, dtype=torch.float32, device=device)
        gamma_fp32 = state['adagc_gamma']


        # Determine the FINAL clipping factor for this parameter
        final_clip_factor_fp32: torch.Tensor # Define type hint

        if step <= warmup_steps and warmup_steps > 0:
             # Warm-up phase: Use the pre-calculated global clip factor
             # Ensure it's on the same device as the gradient we're modifying
             final_clip_factor_fp32 = self._global_clip_factor_fp32.to(device)
        else:
             # AdaGC phase: Calculate the local AdaGC scaling factor in FP32
             # Norm of the raw gradient (grad_fp32 is FP32)
             param_norm_fp32 = torch.linalg.norm(grad_fp32)

             # Get previous EMA gamma (gamma_fp32 is FP32 scalar tensor)
             prev_gamma_fp32 = gamma_fp32
             # Calculate adaptive threshold (FP32 scalar tensor)
             Arel_t = torch.tensor(lambda_rel, dtype=torch.float32, device=device)
             eps_ema_t = torch.tensor(eps, dtype=torch.float32, device=device)
             adaptive_threshold_fp32 = Arel_t * (prev_gamma_fp32 + eps_ema_t)

             # Calculate the static clipping factor: min(1.0, threshold / norm)
             eps_t = torch.tensor(eps, dtype=torch.float32, device=device)
             ratio_fp32 = adaptive_threshold_fp32 / (param_norm_fp32 + eps_t) # Add eps_t to denominator
             # Ensure ratio is not NaN/Inf in edge cases (though adding eps should help)
             ratio_fp32 = torch.nan_to_num(ratio_fp32, nan=1.0, posinf=1.0, neginf=1.0)

             # Create 1.0 tensor on the correct device for torch.min
             one_fp32 = torch.tensor(1.0, device=device, dtype=torch.float32)
             final_clip_factor_fp32 = torch.min(one_fp32, ratio_fp32)


        clipped_grad_fp32 = grad_fp32 # reference for clarity
        clipped_grad_fp32.mul_(final_clip_factor_fp32)
        clipped_param_norm_fp32 = torch.linalg.norm(clipped_grad_fp32)
        ema_beta_t = torch.tensor(ema_beta, dtype=torch.float32, device=device)
        gamma_fp32.mul_(ema_beta_t).add_(clipped_param_norm_fp32, alpha=1.0 - ema_beta_t) # gamma_fp32 is state['gamma']

        # Return the clipped FP32 gradient and the final clipping factor
        return clipped_grad_fp32


@torch.no_grad()
def _paper_orthograd(param, grad, alpha: float = 1.0, eps: float = 1e-20):
    """Applies orthogonal projection to a single parameter's gradient."""

    # Flatten parameter and gradient
    w = param.view(-1) # Use p.data to avoid graph tracking if not needed
    g = grad.view(-1)

    w_norm_sq = torch.dot(w, w)

    # Only project if the weight norm is significant
    # If w_norm_sq is near zero, the parameter contributes little,
    # and projection is ill-defined or numerically unstable.
    # Leave the gradient untouched in this case.
    if w_norm_sq > eps:
        # Calculate projection of g onto w: (w·g / w·w) * w
        proj_coeff = torch.dot(w, g) / w_norm_sq # Note: w_norm_sq already > eps
        g_parallel = proj_coeff * w

        # Subtract the parallel component to get the orthogonal one
        g_orth = g - alpha * g_parallel
        # Apply scaled orthogonalization
        g_orth_scaled = g_orth.mul_(grad.norm(2) / (g_orth.norm(2) + eps))

        # Update the gradient in-place with the orthogonal component
        grad.copy_(g_orth_scaled.view_as(grad))
    # Else: w_norm_sq is too small, leave p.grad as is.
