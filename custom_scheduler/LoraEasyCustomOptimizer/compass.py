# Copied from Lodestone and slightly modified, still should function the same, just added an extra check
# to turn off the stochastic rounding
# repo: https://github.com/lodestone-rock/compass_optimizer/blob/main/experimental/compass_experimental_sr_bf16.py
# Defaults tuned for lora training based on testing

import torch
from torch.optim import Optimizer
from .utils import copy_stochastic_, quantize, dequantize
import math
from torch.nn.functional import softplus

from bitsandbytes.functional import quantize_blockwise, dequantize_blockwise
from pytorch_optimizer.base.exception import NoSparseGradientError, ZeroParameterSizeError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.types import BETAS, CLOSURE, DEFAULTS, LOSS, PARAMETERS
from pytorch_optimizer.optimizer.agc import agc
from pytorch_optimizer.optimizer.gc import centralize_gradient
from pytorch_optimizer.optimizer.utils import normalize_gradient, unit_norm


class Compass(Optimizer):
    r"""
    Arguments:
        params (iterable):
            Iterable of parameters to optimize or dicts defining
            parameter groups.
        lr (float):
            Learning rate parameter (default 7e-5)
        betas (Tuple[float, float], optional):
            coefficients used for computing running averages of
            gradient and its square (default: (0.98, 0.999)).
        weight_decay (float):
            Weight decay, i.e. a L2 penalty (default: 0.001).
        weight_decouple (bool): 
            the optimizer uses decoupled weight decay as in AdamW. (default: true)
        lr_decouple (bool): 
            Apply fully decoupled weight decay. (default: false)
        max_lr (float): 
            Max LR used for lr_decouple (default: 0.0)
        fixed_decay (bool): 
            fix weight decay (default: false).
        clip (float):
            Clip gradient to this value (default: 0.0).
        amp_fac (float):
            amplification factor for the first moment filter (default: 2).
        eps (float):
            Term added to the denominator outside of the root operation to
            improve numerical stability. (default: 1e-8).
        centralization (float):
            center model grad (default: 0.0).
    """

    def __init__(
        self,
        params,
        lr=1e-4, #Original default 1e-3
        betas=(0.975, 0.999), #Original default 0.99, 0.999
        weight_decay=0.001, #Original default 0
        weight_decouple=True,
        lr_decouple=False,
        max_lr=0.0,
        fixed_decay=False,
        clip=0.0,
        amp_fac=2,
        eps=1e-8,
        centralization=0.0,
    ):
        defaults = dict(
            lr=lr,
            betas=betas,
            weight_decay = weight_decay,
            weight_decouple = weight_decouple,
            lr_decouple=lr_decouple,
            max_lr=max_lr,
            fixed_decay = fixed_decay,
            clip=clip,
            amp_fac=amp_fac,
            eps=eps,
            centralization=centralization,
        )



        super(Compass, self).__init__(params, defaults)

    def __str__(self) -> str:
        return 'Compass'
    
    @staticmethod
    def get_rms(x: torch.Tensor) -> float:
        r"""Get RMS."""
        return x.norm(2) / math.sqrt(x.numel())

    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            if 'step' in group:
                group['step'] += 1
            else:
                group['step'] = 1

            beta1, beta2 = group["betas"]
            amplification_factor = group["amp_fac"]
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            weight_decouple = group["weight_decouple"],
            fixed_decay = group["fixed_decay"]
            centralization = group["centralization"]
            eps = group["eps"]
            clip = group["clip"]
            lr_decouple = group["lr_decouple"]
            max_lr = group["max_lr"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("Compass does not support sparse gradients")

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    # Exponential moving average of gradient values
                    state["ema"] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state["ema_squared"] = torch.zeros_like(p.data)

                p_fp32 = p

                ema, ema_squared = state["ema"], state["ema_squared"]

                # unpack
                if p.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.to(torch.float32)
                    p_fp32 = p.clone().to(torch.float32)
                    ema = ema.to(torch.float32)
                    ema_squared = ema_squared.to(torch.float32)

                # center the gradient vector
                if centralization != 0 and grad.dim() > 1:
                    grad.sub_(
                        grad.mean(dim=tuple(range(1, grad.dim())), keepdim=True).mul_(centralization)
                    )

                # bias correction step size
                # soft warmup
                bias_correction = 1 - beta1 ** group['step']
                bias_correction_sqrt = (1 - beta2 ** group['step']) ** (1 / 2)
                debiased_lr = lr / bias_correction

                # Clip the gradient 
                if clip > 0.0:
                    grad.div_((self.get_rms(grad).add_(eps) / clip).clamp_(min=1.0))

                # Decay the first and second moment running average coefficient
                # ema = ema + (1 - beta1) * grad
                ema.mul_(beta1).add_(grad, alpha=1 - beta1)
                # grad = grad + ema * amplification_factor
                grad.add_(ema, alpha=amplification_factor)
                # ema_squared = ema + (1 - beta2) * grad ** 2
                ema_squared.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # lr scaler + eps to prevent zero division
                # denom = exp_avg_sq.sqrt() + group['eps']
                denom = (ema_squared.sqrt() / bias_correction_sqrt).add_(eps)

                if weight_decouple:
                    # Perform stepweight decay
                    p_fp32.data.mul_(1.0 - (1.0 if fixed_decay else debiased_lr if not lr_decouple else debiased_lr / max_lr) * weight_decay)
                elif weight_decay > 0.0 and grad is not None:
                    grad.add_(p_fp32, alpha=weight_decay)

                # p = p - lr * grad / denom
                p_fp32.data.addcdiv_(grad, denom, value=-debiased_lr)

                # pack
                if p.dtype in {torch.float16, torch.bfloat16}:
                    copy_stochastic_(state["ema"], ema)
                    copy_stochastic_(state["ema_squared"], ema_squared)
                    copy_stochastic_(p, p_fp32)

        return loss
    
class CompassExperimental(BaseOptimizer):
    r"""
    Arguments:
        params (iterable):
            Iterable of parameters to optimize or dicts defining
            parameter groups.
        lr (float):
            Learning rate parameter (default 7e-5)
        betas (Tuple[float, float], optional):
            coefficients used for computing running averages of
            gradient and its square (default: (0.98, 0.999)).
        weight_decay (float):
            Weight decay, i.e. a L2 penalty (default: 0.001).
        weight_decouple (bool): 
            the optimizer uses decoupled weight decay as in AdamW. (default: true)
        lr_decouple (bool): 
            Apply fully decoupled weight decay. (default: false)
        max_lr (float): 
            Max LR used for lr_decouple (default: 0.0)
        fixed_decay (bool): 
            fix weight decay (default: false).
        clip (float):
            Clip gradient to this value (default: 0.0).
        :param clip_eps: float. eps for clipping
        amp_fac (float):
            amplification factor for the first moment filter (default: 2).
        eps (float):
            Term added to the denominator outside of the root operation to
            improve numerical stability. (default: 1e-8).
        centralization (float):
            center model grad (default: 0.0).
        :param use_softplus: bool. use softplus to smooth.
        :param beta_softplus: float. beta.
        :param normalize_gradients: bool. use gradient normalization.
        :param lookahead_merge_time: int. merge time.
        :param lookahead_blending_alpha: float. blending alpha.
        :param norm_loss_factor: float. norm loss factor.
        :param adam_debias: bool. Only correct the denominator to avoid inflating step sizes early in training.
        
    """

    def __init__(
        self,
        params: PARAMETERS,
        lr: float = 1.4e-4,
        betas: BETAS = (0.99, 0.999),
        weight_decay: float = 0.1,
        weight_decouple: bool = True,
        lr_decouple: bool = False,
        max_lr: float = 0.0,
        stable_decay: bool = True,
        fixed_decay: bool = False,
        clip: float = 0.01,
        clip_eps: float = 1e-3,
        amp_fac: float = 5.0,
        eps: float = 1e-8,
        centralization: bool = True,
        normalize_gradients: bool = True,
        norm_loss_factor: float = 0.0001,
        use_softplus: bool = True,
        beta_softplus: float = 50.0,
        lookahead: bool = True,
        lookahead_merge_time: int = 5,
        lookahead_blending_alpha: float = 0.5,
        adam_debias: bool = False,
        use_pnm: bool = False,
        pnm_beta: float = 0.9,
        amsgrad: bool = False,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_betas(betas)
        self.validate_range(pnm_beta, 'pnm_beta', 0.0, 1.0, range_type='[]')
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_non_negative(eps, 'eps')
        self.validate_non_negative(max_lr, 'max_lr')
        self.validate_non_negative(clip, 'clip')
        self.validate_non_negative(clip, 'clip_eps')
        self.validate_non_negative(amp_fac, 'amp_fac')
        self.validate_non_negative(centralization, 'centralization')
        self.validate_non_negative(pnm_beta, 'pnm_beta')
        self.validate_non_negative(lookahead_blending_alpha, 'lookahead_blending_alpha')
        self.validate_non_negative(lookahead_merge_time, 'lookahead_merge_time')
        self.validate_non_negative(beta_softplus, 'beta_softplus')
        self.validate_non_negative(norm_loss_factor, 'norm_loss_factor')

        defaults: DEFAULTS = {
            'lr': lr,
            'betas': betas,
            'weight_decay' : weight_decay,
            'weight_decouple' : weight_decouple,
            'lr_decouple': lr_decouple,
            'max_lr': max_lr,
            'stable_decay': stable_decay,
            'fixed_decay': fixed_decay,
            'clip': clip,
            'clip_eps': clip_eps,
            'amp_fac': amp_fac,
            'eps': eps,
            'centralization': centralization,
            'normalize_gradients': normalize_gradients,
            'norm_loss_factor': norm_loss_factor,
            'use_softplus': use_softplus,
            'beta_softplus': beta_softplus,
            'lookahead': lookahead,
            'lookahead_merge_time': lookahead_merge_time,
            'lookahead_blending_alpha': lookahead_blending_alpha,
            'adam_debias': adam_debias,
            'use_pnm': use_pnm,
            'pnm_beta': pnm_beta,
            'amsgrad': amsgrad,
        }

        self.lookahead = lookahead
        self.lookahead_merge_time = lookahead_merge_time
        self.lookahead_blending_alpha = lookahead_blending_alpha
        self.lookahead_step: int = 0
        self.use_pnm = use_pnm
        self.adam_debias = adam_debias
        self.pnm_beta = pnm_beta
        self.amsgrad = amsgrad
        self.stable_decay = stable_decay
        self.beta_softplus = beta_softplus

        super(CompassExperimental, self).__init__(params, defaults)

    def __str__(self) -> str:
        return 'CompassExperimental'
    
    @torch.no_grad()
    def reset(self):
        for group in self.param_groups:
            group['step'] = 0
            for p in group['params']:
                state = self.state[p]

                # Exponential moving average of gradient values
                state["ema"] = torch.zeros_like(p)
                # Exponential moving average of squared gradient values
                state["ema_squared"] = torch.zeros_like(p)

                if self.use_pnm:
                    state['neg_ema'] = torch.zeros_like(p)

                if self.lookahead:
                    state['lookahead_params'] = p.clone()

                if self.amsgrad:
                    state["max_ema_squared"] = torch.zeros_like(p)
    
    @staticmethod
    def get_rms(x: torch.Tensor) -> float:
        r"""Get RMS."""
        return x.norm(2) / math.sqrt(x.numel())

    @torch.no_grad()
    def step(self, closure: CLOSURE = None) -> LOSS:
        loss: LOSS = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        param_size: int = 0
        ema_squared_sum: float = 1.0

        # Phase 1 - Condition the grads and gather aggregates 
        for group in self.param_groups:
            if 'step' in group:
                group['step'] += 1
            else:
                group['step'] = 1

            centralization = group["centralization"]
            normalize_gradients = group["normalize_gradients"]
            eps = group["eps"]
            clip = group["clip"]
            clip_eps = group["clip_eps"]

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise NoSparseGradientError(str(self))
                
                state = self.state[p]
                p_fp32 = p

                param_size += p.numel()

                # State initialization
                if len(state) == 0:
                    # Exponential moving average of gradient values
                    state["ema"] = torch.zeros_like(p)
                    # Exponential moving average of squared gradient values
                    state["ema_squared"] = torch.zeros_like(p)

                    if self.use_pnm:
                        state['neg_ema'] = torch.zeros_like(p)

                    if self.lookahead:
                        state['lookahead_params'] = p.clone()

                    if self.amsgrad:
                        state["max_ema_squared"] = torch.zeros_like(p)

                if p.dtype in {torch.float16, torch.bfloat16}:
                    p_fp32 = p.clone().to(torch.float32)
                    grad = grad.to(torch.float32)

                # Apply Adaptive Gradient Clipping (AGC)
                if clip > 0:
                    grad.copy_(agc(p_fp32, grad, clip_eps, clip))

                # Apply gradient centralization & normalization
                if centralization:
                    centralize_gradient(grad, gc_conv_only=False)

                if normalize_gradients:
                    normalize_gradient(grad)

        if param_size == 0:
            raise ZeroParameterSizeError()

        # Phase 2
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            amplification_factor = group["amp_fac"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad

                state = self.state[p]

                ema, ema_squared = state["ema"], state["ema_squared"]

                if self.use_pnm:
                    if group['step'] % 2 == 1:
                        ema = state['ema']
                    else:
                        ema = state['neg_ema']
                else:
                    ema = state["ema"]

                # unpack
                if p.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.to(torch.float32)
                    ema = ema.to(torch.float32)
                    ema_squared = ema_squared.to(torch.float32)

                # bias correction step size
                # soft warmup
                bias_correction_sqrt: float = math.sqrt(self.debias(beta2, group['step']))

                # Decay the first and second moment running average coefficient
                if self.use_pnm:
                    ema.mul_(beta1 ** 2).add_(grad, alpha=1.0 - beta1 ** 2)  # fmt: skip     
                else:
                    # ema = ema + (1 - beta1) * grad
                    ema.mul_(beta1).add_(grad, alpha=1 - beta1)

                # grad = grad + ema * amplification_factor
                grad.add_(ema, alpha=amplification_factor)
                # ema_squared = ema + (1 - beta2) * grad ** 2
                ema_squared.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                if self.stable_decay:
                    ema_squared_sum += (ema_squared / bias_correction_sqrt).sum()

                # pack
                if p.dtype in {torch.float16, torch.bfloat16}:
                    copy_stochastic_(state["ema_squared"], ema_squared)

                    if self.use_pnm:
                        if group['step'] % 2 == 1:
                            copy_stochastic_(state["ema"], ema)
                        else:
                            copy_stochastic_(state["neg_ema"], ema)
                    else:
                        copy_stochastic_(state["ema"], ema)

        if self.stable_decay:
            ema_squared_normalized = math.sqrt(ema_squared_sum / param_size)
        else:
            ema_squared_normalized = ema_squared_sum

        # Phase 3 - Weight decay and parameter update
        for group in self.param_groups:
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            weight_decouple = group["weight_decouple"],
            fixed_decay = group["fixed_decay"]
            eps = group["eps"]
            lr_decouple = group["lr_decouple"]
            max_lr = group["max_lr"]
            norm_loss_factor = group["norm_loss_factor"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad

                state = self.state[p]

                p_fp32 = p
                ema, ema_squared = state["ema"], state["ema_squared"]

                if self.use_pnm:
                    if group['step'] % 2 == 1:
                        ema, neg_ema = state['ema'], state['ema']
                    else:
                        ema, neg_ema = state['neg_ema'], state['neg_ema']

                if p.dtype in {torch.float16, torch.bfloat16}:
                    p_fp32 = p.clone().to(torch.float32)
                    grad = grad.to(torch.float32)
                    ema = ema.to(torch.float32)
                    ema_squared = ema_squared.to(torch.float32)

                    if self.use_pnm:
                        neg_ema = neg_ema.to(torch.float32)

                # bias correction step size
                # soft warmup
                bias_correction: float = self.debias(beta1, group['step'])
                bias_correction_sqrt: float = math.sqrt(self.debias(beta2, group['step']))
 
                # lr scaler + eps to prevent zero division
                # denom = exp_avg_sq.sqrt() + group['eps']
                if self.amsgrad:
                    max_ema_squared = state['max_ema_squared']

                    if p.dtype in {torch.float16, torch.bfloat16}:
                        max_ema_squared = max_ema_squared.to(torch.float32)
                        
                    torch.max(max_ema_squared, ema_squared, out=max_ema_squared)
                    denom = (max_ema_squared.sqrt() / bias_correction_sqrt).add_(eps)

                    if p.dtype in {torch.float16, torch.bfloat16}:
                        copy_stochastic_(state['max_ema_squared'], max_ema_squared)
                else:
                    denom = (ema_squared.sqrt() / bias_correction_sqrt).add_(eps)

                step_size: float = self.apply_adam_debias(group['adam_debias'], lr, bias_correction)

                if weight_decouple:
                    # Perform stepweight decay
                    p_fp32.data.mul_(1.0 - (1.0 if fixed_decay else step_size if not lr_decouple else step_size / max_lr) * weight_decay * (1.0 / ema_squared_normalized if self.stable_decay else 1.0))
                elif weight_decay > 0.0 and grad is not None:
                    grad.add_(p_fp32, alpha=weight_decay)

                if norm_loss_factor > 0:
                    # norm loss
                    correction = 2.0 * norm_loss_factor * (1.0 - 1.0 / unit_norm(p_fp32).add_(eps))
                    p_fp32.mul_(1.0 - step_size * correction)

                if self.beta_softplus:
                    denom = softplus(denom, beta=self.beta_softplus)

                if self.use_pnm:
                    noise_norm: float = math.sqrt((1.0 + self.pnm_beta) ** 2 + self.pnm_beta ** 2)       
                    update = ema.mul(1.0 + self.pnm_beta).add_(neg_ema, alpha=-self.pnm_beta).mul_(1.0 / noise_norm).div_(denom)
                else:
                    update = grad.div(denom)

                if centralization:
                    centralize_gradient(update, gc_conv_only=False)

                if normalize_gradients:
                    normalize_gradient(update) 

                # p = p - lr * grad / denom
                p_fp32.data.add_(update, alpha=-step_size)

                # pack
                if p.dtype in {torch.float16, torch.bfloat16}:
                    copy_stochastic_(p, p_fp32)

        if self.lookahead:
            self.lookahead_process_step()

        return loss
    
    def lookahead_process_step(self):
        self.lookahead_step += 1
        if self.lookahead_step >= self.lookahead_merge_time:
            self.lookahead_step: int = 0
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None:
                        continue

                    state = self.state[p]

                    p_fp32 = p

                    lookahead_params = state['lookahead_params']

                    if p.dtype in {torch.float16, torch.bfloat16}:
                        p_fp32 = p.clone().to(torch.float32)
                        lookahead_params = lookahead_params.to(torch.float32)



                    p_fp32.mul_(self.lookahead_blending_alpha).add_(
                        lookahead_params,
                        alpha=1.0 - self.lookahead_blending_alpha,
                    )

                    # pack
                    if p.dtype in {torch.float16, torch.bfloat16}:
                        copy_stochastic_(p, p_fp32)

                    state['lookahead_params'].copy_(p)

class CompassPlus(BaseOptimizer):
    r"""
    Arguments:
        :param params: PARAMETERS. iterable of parameters to optimize or dicts defining parameter groups.
        :param lr: float. learning rate.
        :param beta0: float. Manages the amplitude of the noise introduced by positive negative momentum
            While 0.9 is a recommended default value, you can use -0.5 to minimize the noise.
        :param betas: BETAS. coefficients used for computing running averages of gradient and the squared hessian trace.
        :param use_softplus: bool. use softplus to smooth.
        :param beta_softplus: float. beta.
        :param agc_clipping_value: float. Clipping threshold for adaptive gradient clipping.
        :param agc_eps: float. eps for adaptive gradient clipping.
        :param amp_fac: float. amplification factor for the first moment filter.
        :param centralize_gradients: bool. use GC both convolution & fc layers.
        :param normalize_gradients: bool. use gradient normalization.
        :param lookahead_merge_time: int. merge time.
        :param lookahead_blending_alpha: float. blending alpha.
        :param weight_decay: float. weight decay (L2 penalty).
        :param weight_decouple: bool. the optimizer uses decoupled weight decay as in AdamW.
        :param lr_decouple: bool. fully decouple weight decay from learning rate.
        :param max_lr: float. Max LR used for lr_decouple, should match your defined max LR for training.
        :param fixed_decay: bool. fix weight decay.
        :param norm_loss_factor: float. norm loss factor.
        :param adam_debias: bool. Only correct the denominator to avoid inflating step sizes early in training.
        :param eps: float. term added to the denominator to improve numerical stability.
    """

    def __init__(
        self,
        params: PARAMETERS,
        lr: float = 1.4e-4,
        betas: BETAS = (0.99, 0.999),
        weight_decay: float = 0.1,
        weight_decouple: bool = True,
        lr_decouple: bool = False,
        max_lr: float = 0.0,
        stable_decay: bool = True,
        fixed_decay: bool = False,
        clip: float = 0.01,
        clip_eps: float = 1e-3,
        amp_fac: float = 5.0,
        eps: float = 1e-8,
        centralize_gradients: bool = True,
        normalize_gradients: bool = True,
        norm_loss_factor: float = 0.0001,
        use_softplus: bool = True,
        beta_softplus: float = 50.0,
        use_lookahead: bool = True,
        lookahead_merge_time: int = 5,
        lookahead_blending_alpha: float = 0.5,
        adam_debias: bool = False,
        use_pnm: bool = False,
        pnm_beta: float = 0.9,
        amsgrad: bool = False,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_betas(betas)
        self.validate_range(pnm_beta, 'pnm_beta', 0.0, 1.0, range_type='[]')
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_non_negative(eps, 'eps')
        self.validate_non_negative(max_lr, 'max_lr')
        self.validate_non_negative(clip, 'clip')
        self.validate_non_negative(clip, 'clip_eps')
        self.validate_non_negative(amp_fac, 'amp_fac')
        self.validate_non_negative(centralize_gradients, 'centralize_gradients')
        self.validate_non_negative(pnm_beta, 'pnm_beta')
        self.validate_non_negative(lookahead_blending_alpha, 'lookahead_blending_alpha')
        self.validate_non_negative(lookahead_merge_time, 'lookahead_merge_time')
        self.validate_non_negative(beta_softplus, 'beta_softplus')
        self.validate_non_negative(norm_loss_factor, 'norm_loss_factor')

        defaults: DEFAULTS = {
            'lr': lr,
            'betas': betas,
            'weight_decay' : weight_decay,
            'weight_decouple' : weight_decouple,
            'lr_decouple': lr_decouple,
            'max_lr': max_lr,
            'stable_decay': stable_decay,
            'fixed_decay': fixed_decay,
            'clip': clip,
            'clip_eps': clip_eps,
            'amp_fac': amp_fac,
            'eps': eps,
            'centralization': centralize_gradients,
            'normalize_gradients': normalize_gradients,
            'norm_loss_factor': norm_loss_factor,
            'use_softplus': use_softplus,
            'beta_softplus': beta_softplus,
            'use_lookahead': use_lookahead,
            'lookahead_merge_time': lookahead_merge_time,
            'lookahead_blending_alpha': lookahead_blending_alpha,
            'adam_debias': adam_debias,
            'use_pnm': use_pnm,
            'pnm_beta': pnm_beta,
            'amsgrad': amsgrad,
        }

        self.use_lookahead = use_lookahead
        self.lookahead_merge_time = lookahead_merge_time
        self.lookahead_blending_alpha = lookahead_blending_alpha
        self.lookahead_step: int = 0
        self.use_pnm = use_pnm
        self.adam_debias = adam_debias
        self.pnm_beta = pnm_beta
        self.amsgrad = amsgrad
        self.stable_decay = stable_decay
        self.centralize_gradients = centralize_gradients
        self.normalize_gradients = normalize_gradients
        self.use_softplus = use_softplus
        self.beta_softplus = beta_softplus
        self.norm_loss_factor = norm_loss_factor
        self.lr_decouple = lr_decouple
        self.weight_decay = weight_decay
        self.weight_decouple = weight_decouple
        self.max_lr = max_lr
        self.fixed_decay = fixed_decay
        self.eps = eps
        self.clip = clip
        self.clip_eps = clip_eps

        super(CompassExperimental, self).__init__(params, defaults)

    def __str__(self) -> str:
        return 'CompassExperimental'
    
    @torch.no_grad()
    def reset(self):
        for group in self.param_groups:
            group['step'] = 0
            for p in group['params']:
                state = self.state[p]

                # Exponential moving average of gradient values
                state["ema"] = torch.zeros_like(p)
                # Exponential moving average of squared gradient values
                state["ema_squared"] = torch.zeros_like(p)

                if self.use_pnm:
                    state['neg_ema'] = torch.zeros_like(p)

                if self.lookahead:
                    state['lookahead_params'] = p.clone()

                if self.amsgrad:
                    state["max_ema_squared"] = torch.zeros_like(p)
    
    @staticmethod
    def get_rms(x: torch.Tensor) -> float:
        r"""Get RMS."""
        return x.norm(2) / math.sqrt(x.numel())

    @torch.no_grad()
    def step(self, closure: CLOSURE = None) -> LOSS:
        loss: LOSS = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        param_size: int = 0
        ema_squared_sum: float = 1.0

        # Phase 1 - Condition the grads and gather aggregates 
        for group in self.param_groups:
            if 'step' in group:
                group['step'] += 1
            else:
                group['step'] = 1

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise NoSparseGradientError(str(self))
                
                state = self.state[p]
                p_fp32 = p

                param_size += p.numel()

                # State initialization
                if len(state) == 0:
                    # Exponential moving average of gradient values
                    state["ema"] = torch.zeros_like(p)
                    # Exponential moving average of squared gradient values
                    state["ema_squared"] = torch.zeros_like(p)

                    if self.use_pnm:
                        state['neg_ema'] = torch.zeros_like(p)

                    if self.lookahead:
                        state['lookahead_params'] = p.clone()

                    if self.amsgrad:
                        state["max_ema_squared"] = torch.zeros_like(p)

                if p.dtype in {torch.float16, torch.bfloat16}:
                    p_fp32 = p.clone().to(torch.float32)
                    grad = grad.to(torch.float32)

                # Apply Adaptive Gradient Clipping (AGC)
                if self.clip > 0.0:
                    grad.copy_(agc(p_fp32, grad, self.clip_eps, self.clip))

                # Apply gradient centralization & normalization
                if self.centralize_gradients:
                    centralize_gradient(grad, gc_conv_only=False)

                if self.normalize_gradients:
                    normalize_gradient(grad)

        if param_size == 0:
            raise ZeroParameterSizeError()

        # Phase 2
        for group in self.param_groups:
            beta1, beta2 = group["betas"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad

                state = self.state[p]

                ema, ema_squared = state["ema"], state["ema_squared"]

                if self.use_pnm:
                    if group['step'] % 2 == 1:
                        ema = state['ema']
                    else:
                        ema = state['neg_ema']
                else:
                    ema = state["ema"]

                # unpack
                if p.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.to(torch.float32)
                    ema = ema.to(torch.float32)
                    ema_squared = ema_squared.to(torch.float32)

                # bias correction step size
                # soft warmup
                bias_correction_sqrt: float = math.sqrt(self.debias(beta2, group['step']))

                # Decay the first and second moment running average coefficient
                if self.use_pnm:
                    ema.mul_(beta1 ** 2).add_(grad, alpha=1.0 - beta1 ** 2)  # fmt: skip     
                else:
                    # ema = ema + (1 - beta1) * grad
                    ema.mul_(beta1).add_(grad, alpha=1 - beta1)

                # grad = grad + ema * amp_fac
                grad.add_(ema, alpha=self.amp_fac)
                # ema_squared = ema + (1 - beta2) * grad ** 2
                ema_squared.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                if self.stable_decay:
                    ema_squared_sum += (ema_squared / bias_correction_sqrt).sum()

                # pack
                if p.dtype in {torch.float16, torch.bfloat16}:
                    copy_stochastic_(state["ema_squared"], ema_squared)

                    if self.use_pnm:
                        if group['step'] % 2 == 1:
                            copy_stochastic_(state["ema"], ema)
                        else:
                            copy_stochastic_(state["neg_ema"], ema)
                    else:
                        copy_stochastic_(state["ema"], ema)

        if self.stable_decay:
            ema_squared_normalized = math.sqrt(ema_squared_sum / param_size)
        else:
            ema_squared_normalized = ema_squared_sum

        # Phase 3 - Weight decay and parameter update
        for group in self.param_groups:
            lr = group["lr"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad

                state = self.state[p]

                p_fp32 = p
                ema, ema_squared = state["ema"], state["ema_squared"]

                if self.use_pnm:
                    if group['step'] % 2 == 1:
                        ema, neg_ema = state['ema'], state['ema']
                    else:
                        ema, neg_ema = state['neg_ema'], state['neg_ema']

                if p.dtype in {torch.float16, torch.bfloat16}:
                    p_fp32 = p.clone().to(torch.float32)
                    grad = grad.to(torch.float32)
                    ema = ema.to(torch.float32)
                    ema_squared = ema_squared.to(torch.float32)

                    if self.use_pnm:
                        neg_ema = neg_ema.to(torch.float32)

                # bias correction step size
                # soft warmup
                bias_correction: float = self.debias(beta1, group['step'])
                bias_correction_sqrt: float = math.sqrt(self.debias(beta2, group['step']))
 
                # lr scaler + eps to prevent zero division
                # denom = exp_avg_sq.sqrt() + group['eps']
                if self.amsgrad:
                    max_ema_squared = state['max_ema_squared']

                    if p.dtype in {torch.float16, torch.bfloat16}:
                        max_ema_squared = max_ema_squared.to(torch.float32)
                        
                    torch.max(max_ema_squared, ema_squared, out=max_ema_squared)
                    denom = (max_ema_squared.sqrt() / bias_correction_sqrt).add_(self.eps)

                    if p.dtype in {torch.float16, torch.bfloat16}:
                        copy_stochastic_(state['max_ema_squared'], max_ema_squared)
                else:
                    denom = (ema_squared.sqrt() / bias_correction_sqrt).add_(self.eps)

                step_size: float = self.apply_adam_debias(group['adam_debias'], lr, bias_correction)

                if self.weight_decouple:
                    # Perform stepweight decay
                    p_fp32.data.mul_(1.0 - (1.0 if self.fixed_decay else step_size if not self.lr_decouple else step_size / self.max_lr) * self.weight_decay * (1.0 / ema_squared_normalized if self.stable_decay else 1.0))
                elif self.weight_decay > 0.0 and grad is not None:
                    grad.add_(p_fp32, alpha=self.weight_decay)

                if self.norm_loss_factor > 0.0:
                    # norm loss
                    correction = 2.0 * self.norm_loss_factor * (1.0 - 1.0 / unit_norm(p_fp32).add_(self.eps))
                    p_fp32.mul_(1.0 - step_size * correction)

                if self.use_softplus:
                    denom = softplus(denom, beta=self.beta_softplus)

                if self.use_pnm:
                    noise_norm: float = math.sqrt((1.0 + self.pnm_beta) ** 2 + self.pnm_beta ** 2)       
                    update = ema.mul(1.0 + self.pnm_beta).add_(neg_ema, alpha=-self.pnm_beta).mul_(1.0 / noise_norm).div_(denom)
                else:
                    update = grad.div(denom)

                if self.centralize_gradients:
                    centralize_gradient(update, gc_conv_only=False)

                if self.normalize_gradients:
                    normalize_gradient(update) 

                # p = p - lr * grad / denom
                p_fp32.data.add_(update, alpha=-step_size)

                # pack
                if p.dtype in {torch.float16, torch.bfloat16}:
                    copy_stochastic_(p, p_fp32)

        if self.lookahead:
            self.lookahead_process_step()

        return loss
    
    def lookahead_process_step(self):
        self.lookahead_step += 1
        if self.lookahead_step >= self.lookahead_merge_time:
            self.lookahead_step: int = 0
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None:
                        continue

                    state = self.state[p]

                    p_fp32 = p

                    lookahead_params = state['lookahead_params']

                    if p.dtype in {torch.float16, torch.bfloat16}:
                        p_fp32 = p.clone().to(torch.float32)
                        lookahead_params = lookahead_params.to(torch.float32)



                    p_fp32.mul_(self.lookahead_blending_alpha).add_(
                        lookahead_params,
                        alpha=1.0 - self.lookahead_blending_alpha,
                    )

                    # pack
                    if p.dtype in {torch.float16, torch.bfloat16}:
                        copy_stochastic_(p, p_fp32)

                    state['lookahead_params'].copy_(p)


class Compass8Bit(Optimizer):
    r"""
    Arguments:
        params (iterable):
            Iterable of parameters to optimize or dicts defining
            parameter groups.
        lr (float):
            Learning rate parameter (default 7e-5)
        betas (Tuple[float, float], optional):
            coefficients used for computing running averages of
            gradient and its square (default: (0.98, 0.999)).
        weight_decay (float):
            Weight decay, i.e. a L2 penalty (default: 0.001).
        weight_decouple (bool): 
            the optimizer uses decoupled weight decay as in AdamW. (default: true)
        fixed_decay (bool): 
            fix weight decay (default: false).
        clip (float):
            Clip gradient to this value (default: 0.0).
        amp_fac (float):
            amplification factor for the first moment filter (default: 2).
        eps (float):
            Term added to the denominator outside of the root operation to
            improve numerical stability. (default: 1e-8).
        centralization (float):
            center model grad (default: 0.0).
        quantization_group_size (int):
            number of quant group (default: 8).
        quantization_factor (float):
            non linear quantization using x^f (default: 3.2)
    """

    def __init__(
        self,
        params,
        lr=1e-4, #Original default 1e-3
        betas=(0.975, 0.999), #Original default 0.99, 0.999
        weight_decay=0.001, #Original default 0
        weight_decouple=True,
        fixed_decay=False,
        clip=0.0,
        amp_fac=2,
        eps=1e-8,
        centralization=1.0,
        quantization_group_size=8,
        quantization_factor=3.2,
    ):
        defaults = dict(
            lr=lr,
            betas=betas,
            amp_fac=amp_fac,
            eps=eps,
            weight_decay = weight_decay,
            weight_decouple = weight_decouple,
            fixed_decay = fixed_decay,
            clip=clip,
            centralization=centralization,
            group_size=quantization_group_size,
            factor=quantization_factor,
        )
        super(Compass8Bit, self).__init__(params, defaults)

    def __str__(self) -> str:
        return 'Compass8Bit'
    
    @staticmethod
    def get_rms(x: torch.Tensor) -> float:
        r"""Get RMS."""
        return x.norm(2) / math.sqrt(x.numel())

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            if 'step' in group:
                group['step'] += 1
            else:
                group['step'] = 1

            beta1, beta2 = group["betas"]
            amplification_factor = group["amp_fac"]
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            weight_decouple = group["weight_decouple"],
            fixed_decay = group["fixed_decay"]
            centralization = group["centralization"]
            eps = group["eps"]
            clip = group["clip"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("Compass8Bit does not support sparse gradients")

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    # Exponential moving average of gradient values
                    state["ema"] = quantize(
                        torch.zeros_like(p.data),
                        group_size=group["group_size"],
                        factor=group["factor"],
                    )
                    # Exponential moving average of squared gradient values
                    state["ema_squared"] = quantize(
                        torch.zeros_like(p.data),
                        group_size=group["group_size"],
                        factor=group["factor"],
                    )

                p_fp32 = p

                # unpack
                if p.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.to(torch.float32)
                    p_fp32 = p.clone().to(torch.float32)

                beta1, beta2 = group["betas"]
                amplification_factor = group["amp_fac"]
                lr = group["lr"]
                weight_decay = group["weight_decay"]
                weight_decouple = group["weight_decouple"],
                fixed_decay = group["fixed_decay"]
                centralization = group["centralization"]
                eps = group["eps"]
                clip = group["clip"]

                # center the gradient vector
                if centralization != 0 and grad.dim() > 1:
                    grad.sub_(
                        grad.mean(dim=tuple(range(1, grad.dim())), keepdim=True).mul_(centralization)
                    )

                # bias correction step size
                # soft warmup
                bias_correction = 1 - beta1 ** group['step']
                bias_correction_sqrt = (1 - beta2 ** group['step']) ** (1 / 2)
                debiased_lr = lr / bias_correction

                # Clip the gradient 
                if clip > 0.0:
                    grad.div_((self.get_rms(grad).add_(eps) / clip).clamp_(min=1.0))

                # Decay the first and second moment running average coefficient
                ema = dequantize(*state["ema"]) + (1 - beta1) * grad
                # ema.mul_(beta1).add_(grad, alpha=1 - beta1)
                # grad = grad + ema * amplification_factor
                grad.add_(ema, alpha=amplification_factor)

                ema_squared = dequantize(*state["ema_squared"]) + (1 - beta2) * grad**2
                state["ema"] = quantize(
                    ema, group_size=group["group_size"], factor=group["factor"]
                )

                # ema_squared.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # lr scaler + eps to prevent zero division
                # denom = exp_avg_sq.sqrt() + group['eps']
                denom = (ema_squared.sqrt() / bias_correction_sqrt).add_(group["eps"])
                state["ema_squared"] = quantize(
                    ema_squared, group_size=group["group_size"], factor=group["factor"]
                )

                if weight_decouple:
                    # Perform stepweight decay
                    p_fp32.data.mul_(1.0 - (1.0 if fixed_decay else debiased_lr) * weight_decay)
                elif weight_decay > 0.0 and grad is not None:
                    grad.add_(p_fp32, alpha=weight_decay)

                # p = p - lr * grad / denom
                p_fp32.data.addcdiv_(grad, denom, value=-debiased_lr)

                # pack
                if p.dtype in {torch.float16, torch.bfloat16}:
                    copy_stochastic_(p, p_fp32)

        return loss

class Compass8BitBNB(Optimizer):
    r"""
    Arguments:
        params (iterable):
            Iterable of parameters to optimize or dicts defining
            parameter groups.
        lr (float):
            Learning rate parameter (default 7e-5)
        betas (Tuple[float, float], optional):
            coefficients used for computing running averages of
            gradient and its square (default: (0.98, 0.999)).
        weight_decay (float):
            Weight decay, i.e. a L2 penalty (default: 0.001).
        weight_decouple (bool): 
            the optimizer uses decoupled weight decay as in AdamW. (default: true)
        fixed_decay (bool): 
            fix weight decay (default: false).
        clip (float):
            Clip gradient to this value (default: 0.0).
        amp_fac (float):
            amplification factor for the first moment filter (default: 2).
        eps (float):
            Term added to the denominator outside of the root operation to
            improve numerical stability. (default: 1e-8).
        centralization (float):
            center model grad (default: 0.0).
        quantization_group_size (int):
            number of quant group (default: 64).
    """

    def __init__(
        self,
        params,
        lr=1e-4, #Original default 1e-3
        betas=(0.975, 0.999), #Original default 0.99, 0.999
        weight_decay=0.001, #Original default 0
        weight_decouple=True,
        fixed_decay=False,
        clip=0.0,
        amp_fac=2,
        eps=1e-8,
        centralization=0.0,
        quantization_group_size=64,
    ):
        defaults = dict(
            lr=lr,
            betas=betas,
            amp_fac=amp_fac,
            eps=eps,
            weight_decay = weight_decay,
            weight_decouple = weight_decouple,
            fixed_decay = fixed_decay,
            clip=clip,
            centralization=centralization,
            group_size=quantization_group_size,
        )
        super(Compass8BitBNB, self).__init__(params, defaults)

    def __str__(self) -> str:
        return 'Compass8BitBNB'
    
    @staticmethod
    def get_rms(x: torch.Tensor) -> float:
        r"""Get RMS."""
        return x.norm(2) / math.sqrt(x.numel())

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            if 'step' in group:
                group['step'] += 1
            else:
                group['step'] = 1

            beta1, beta2 = group["betas"]
            amplification_factor = group["amp_fac"]
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            weight_decouple = group["weight_decouple"],
            fixed_decay = group["fixed_decay"]
            centralization = group["centralization"]
            eps = group["eps"]
            clip = group["clip"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("Compass8BitBNB does not support sparse gradients")

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    # Exponential moving average of gradient values
                    state["ema"] = quantize_blockwise(
                        torch.zeros_like(p.data),
                        blocksize=group["group_size"],
                    )
                    # Exponential moving average of squared gradient values
                    state["ema_squared"] = quantize_blockwise(
                        torch.zeros_like(p.data),
                        blocksize=group["group_size"],
                    )

                p_fp32 = p

                # unpack
                if p.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.to(torch.float32)
                    p_fp32 = p.clone().to(torch.float32)

                beta1, beta2 = group["betas"]
                amplification_factor = group["amp_fac"]
                lr = group["lr"]
                weight_decay = group["weight_decay"]
                weight_decouple = group["weight_decouple"],
                fixed_decay = group["fixed_decay"]
                centralization = group["centralization"]
                eps = group["eps"]
                clip = group["clip"]

                # center the gradient vector
                if centralization != 0 and grad.dim() > 1:
                    grad.sub_(
                        grad.mean(dim=tuple(range(1, grad.dim())), keepdim=True).mul_(centralization)
                    )

                # bias correction step size
                # soft warmup
                bias_correction = 1 - beta1 ** group['step']
                bias_correction_sqrt = (1 - beta2 ** group['step']) ** (1 / 2)
                debiased_lr = lr / bias_correction

                # Clip the gradient 
                if clip > 0.0:
                    grad.div_((self.get_rms(grad).add_(eps) / clip).clamp_(min=1.0))

                # Decay the first and second moment running average coefficient
                ema = dequantize_blockwise(*state["ema"]) + (1 - beta1) * grad
                # ema.mul_(beta1).add_(grad, alpha=1 - beta1)
                # grad = grad + ema * amplification_factor
                grad.add_(ema, alpha=amplification_factor)

                ema_squared = (
                    dequantize_blockwise(*state["ema_squared"]) + (1 - beta2) * grad**2
                )
                state["ema"] = quantize_blockwise(
                    ema,
                    blocksize=group["group_size"],
                )

                # ema_squared.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # lr scaler + eps to prevent zero division
                # denom = exp_avg_sq.sqrt() + group['eps']
                denom = (ema_squared.sqrt() / bias_correction_sqrt).add_(group["eps"])
                state["ema_squared"] = quantize_blockwise(
                    ema_squared,
                    blocksize=group["group_size"],
                )

                if weight_decouple:
                    # Perform stepweight decay
                    p_fp32.data.mul_(1.0 - (1.0 if fixed_decay else debiased_lr) * weight_decay)
                elif weight_decay > 0.0 and grad is not None:
                    grad.add_(p_fp32, alpha=weight_decay)

                # p = p - lr * grad / denom
                p_fp32.data.addcdiv_(grad, denom, value=-debiased_lr)

                # pack
                if p.dtype in {torch.float16, torch.bfloat16}:
                    copy_stochastic_(p, p_fp32)

        return loss