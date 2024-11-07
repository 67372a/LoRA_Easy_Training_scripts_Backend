# Copied from Lodestone and slightly modified, still should function the same, just added an extra check
# to turn off the stochastic rounding
# repo: https://github.com/lodestone-rock/compass_optimizer/blob/main/experimental/compass_experimental_sr_bf16.py
# Defaults tuned for lora training based on testing

import torch
from torch.optim import Optimizer
from .utils import copy_stochastic_, quantize, dequantize, agc
import math
from torch.nn.functional import softplus
from typing import Optional

from bitsandbytes.functional import quantize_blockwise, dequantize_blockwise
from pytorch_optimizer.base.exception import NoSparseGradientError, ZeroParameterSizeError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.types import BETAS, CLOSURE, DEFAULTS, LOSS, PARAMETERS
from pytorch_optimizer.optimizer.gc import centralize_gradient
from pytorch_optimizer.optimizer.utils import normalize_gradient, unit_norm

class Compass(BaseOptimizer):
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
            Gradient centralization  - https://arxiv.org/abs/2004.01461v2 (default: 0.0).
        adaptive_clipping (bool):
            enable adaptive clipping - https://arxiv.org/abs/2102.06171 (default: false).
        adaptive_clipping_eps (float):
            eps for adaptive gradient clipping (default: 1e-3).
        adam_debias: (bool)
            Only correct the denominator to avoid inflating step sizes early in training. (Default: false)
        rectify_variance: (bool)
            Rectify variance as per RAdam - https://arxiv.org/abs/1908.03265 (Default: false)
        n_sma_threshold: (int)
            Simple moving average threshold for variance rectification (recommended is 5) (Default: 5).
        degenerated_to_sgd: (bool)
            degenerated to SGD. (Default: false)
    """

    def __init__(
        self,
        params: PARAMETERS,
        lr: float = 1.4e-4, #Original default 1e-3
        betas: BETAS = (0.975, 0.999), #Original default 0.99, 0.999
        weight_decay: float = 0.001, #Original default 0
        weight_decouple: bool = True,
        lr_decouple: bool = False,
        max_lr: float = 0.0,
        fixed_decay: bool = False,
        clip: float = 0.01,
        amp_fac: float = 2.0,
        eps: float = 1e-8,
        centralization: float = 0.0,
        adaptive_clipping: bool = False,
        adaptive_clip_eps: float = 1e-3,
        adam_debias: bool = False,
        rectify_variance: bool = False,
        n_sma_threshold: int = 5,
        degenerated_to_sgd: bool = False,
        **kwargs,
    ):
        defaults: DEFAULTS = {
            'lr':lr,
            'betas':betas,
            'weight_decay' : weight_decay,
            'weight_decouple' : weight_decouple,
            'lr_decouple':lr_decouple,
            'max_lr':max_lr,
            'fixed_decay' : fixed_decay,
            'clip':clip,
            'adaptive_clip_eps':adaptive_clip_eps,
            'amp_fac':amp_fac,
            'eps':eps,
            'centralization':centralization,
            'adaptive_clipping':adaptive_clipping,
            'adam_debias': adam_debias,
            'rectify_variance': rectify_variance,
            'n_sma_threshold': n_sma_threshold,
            'degenerated_to_sgd': degenerated_to_sgd,
        }

        self.clip = clip
        self.adaptive_clip_eps = adaptive_clip_eps
        self.adaptive_clipping = adaptive_clipping
        self.adam_debias = adam_debias
        self.rectify_variance = rectify_variance
        self.n_sma_threshold = n_sma_threshold
        self.degenerated_to_sgd = degenerated_to_sgd

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'Compass'
    
    @staticmethod
    def get_rms(x: torch.Tensor) -> float:
        r"""Get RMS."""
        return x.norm(2) / math.sqrt(x.numel())
    
    @torch.no_grad()
    def reset(self):
        for group in self.param_groups:
            group['step'] = 0

            for p in group['params']:
                state = self.state[p]

                # Exponential moving average of gradient values
                state['ema'] = torch.zeros_like(p)
                # Exponential moving average of squared gradient values
                state["ema_squared"] = torch.zeros_like(p)

    @torch.no_grad()
    def step(self, closure: CLOSURE = None) -> LOSS:
        loss: LOSS = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if 'step' in group:
                group['step'] += 1
            else:
                group['step'] = 1

            beta1, beta2 = group["betas"]
            amp_fac = group["amp_fac"]
            weight_decay = group["weight_decay"]
            weight_decouple = group["weight_decouple"],
            fixed_decay = group["fixed_decay"]
            centralization = group["centralization"]
            eps = group["eps"]
            lr_decouple = group["lr_decouple"]
            max_lr = group["max_lr"]

            # bias correction step size
            # soft warmup
            bias_correction1: float = self.debias(beta1, group['step'])
            bias_correction2_sqrt: float = math.sqrt(self.debias(beta2, group['step']))

            step_size, n_sma = self.get_rectify_step_size(
                is_rectify=self.rectify_variance,
                step=group['step'],
                lr=group['lr'],
                beta2=beta2,
                n_sma_threshold=self.n_sma_threshold,
                degenerated_to_sgd=self.degenerated_to_sgd,
            )

            step_size = self.apply_adam_debias(
                adam_debias=self.adam_debias,
                step_size=step_size,
                bias_correction1=bias_correction1,
            )

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad

                if grad.is_sparse:
                    raise NoSparseGradientError(str(self))

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    # Exponential moving average of gradient values
                    state["ema"] = torch.zeros_like(p)
                    # Exponential moving average of squared gradient values
                    state["ema_squared"] = torch.zeros_like(p)

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

                if self.clip > 0.0:
                    if self.adaptive_clipping:
                        # Apply Adaptive Gradient Clipping (AGC)
                        grad.copy_(agc(p_fp32, grad, self.adaptive_clip_eps, self.clip))
                    else:
                        # Clip the gradient 
                        grad.div_((self.get_rms(grad).add_(eps) / self.clip).clamp_(min=1.0))

                # Decay the first and second moment running average coefficient
                # ema = ema + (1 - beta1) * grad
                ema.mul_(beta1).add_(grad, alpha=1 - beta1)
                # grad = grad + ema * amp_fac
                grad.add_(ema, alpha=amp_fac)
                # ema_squared = ema + (1 - beta2) * grad ** 2
                ema_squared.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                if not self.rectify_variance or step_size > 0 or n_sma >= self.n_sma_threshold:
                    if weight_decouple:
                        # Perform stepweight decay
                        p_fp32.mul_(1.0 - (1.0 if fixed_decay else step_size if not lr_decouple else step_size / max_lr) * weight_decay)
                    elif weight_decay > 0.0 and grad is not None:
                        grad.add_(p_fp32, alpha=weight_decay)

                if not self.rectify_variance or n_sma >= self.n_sma_threshold:
                    # lr scaler + eps to prevent zero division
                    # de_nom = exp_avg_sq.sqrt() + group['eps']
                    if self.rectify_variance:
                        de_nom = ema_squared.sqrt().add_(eps)
                    else:
                        de_nom = (ema_squared.sqrt() / bias_correction2_sqrt).add_(eps)

                    # p = p - lr * grad / de_nom
                    p_fp32.addcdiv_(grad, de_nom, value=-step_size)
                elif step_size > 0:
                    p_fp32.add_(grad, alpha=-step_size)

                # pack
                if p.dtype in {torch.float16, torch.bfloat16}:
                    copy_stochastic_(state["ema"], ema)
                    copy_stochastic_(state["ema_squared"], ema_squared)
                    copy_stochastic_(p, p_fp32)

        return loss
    
class CompassPlus(BaseOptimizer):
    r"""
    CompassPlus
        Components
            * Adaptive gradient clipping - https://arxiv.org/abs/2102.06171
            * Gradient centralization - https://arxiv.org/abs/2004.01461v2
            * Positive-Negative momentum - https://arxiv.org/abs/2103.17182
            * Norm loss - https://arxiv.org/abs/2103.06583v1
            * Fully decoupled weight decay - https://optimi.benjaminwarner.dev/fully_decoupled_weight_decay/ / https://arxiv.org/abs/1711.05101
            * Stable weight decay - https://arxiv.org/abs/2011.11152v3
            * Lookahead - https://arxiv.org/abs/1907.08610
            * Softplus transformation - https://arxiv.org/abs/1908.00700
            * Gradient Normalization - https://arxiv.org/pdf/1711.02257 (?)
            * Adaptive eps - https://arxiv.org/abs/2405.12807
            * Diff amp - https://github.com/Clybius/Personalized-Optimizers/blob/main/FishMonger.py
            * Slow EMA - https://arxiv.org/abs/2409.03137
            * Amsgrad - https://arxiv.org/pdf/1904.09237
            * Update Clipping - https://arxiv.org/pdf/2304.13013 (AdamWStable) / https://arxiv.org/pdf/1804.04235 (Adafactor)
            * Variance Rectification - https://arxiv.org/abs/1908.03265 (RAdam)

    Arguments:
        :param params: PARAMETERS. iterable of parameters to optimize or dicts defining parameter groups.
        :param lr: float. learning rate.
        :param betas: BETAS. coefficients used for computing running averages of gradient and the squared hessian trace.
        :param use_softplus: bool. use softplus to smooth the updaate denominator.
        :param beta_softplus: float. beta for softplus.
        :param threshold_softplus: float. threshold after which scaling returns to linear. Originally set to 20 by default, instead follows adaptive eps when set to 0.
        :param agc_clipping_value: float. Clipping threshold for adaptive gradient clipping.
        :param agc_eps: float. eps for adaptive gradient clipping.
        :param amp_fac: float. amplification factor for the first moment filter.
        :param centralize_gradients: bool. use GC both convolution & fc layers. Can be selectively applied an int: disabled(0), gradient(1), update(2), both(3)
        :param normalize_gradients: bool. use gradient normalization.  Can be selectively applied using an int: disabled(0), gradient(1), update(2), both(3)
        :param use_lookahead: bool. use lookahead. ADDS 1 STATE
        :param lookahead_merge_time: int. merge time.
        :param lookahead_blending_alpha: float. blending alpha.
        :param weight_decay: float. weight decay (L2 penalty).
        :param weight_decouple: bool. the optimizer uses decoupled weight decay as in AdamW.
        :param lr_decouple: bool. fully decouple weight decay from learning rate. This makes weight decay much stronger given the same values.
        :param max_lr: float. Max LR used for lr_decouple, should match your defined max LR for training.
        :param fixed_decay: bool. fix weight decay.
        :param norm_loss_factor: float. norm loss factor.
        :param norm_loss_eps: float. Eps is the term added to the denominator to improve numerical stability.
        :param adam_debias: bool. Only correct the denominator to avoid inflating step sizes early in training.
        :param amsgrad: bool. If true, maintains and uses the max ema squared. ADDS 1 STATE
        :param use_pnm: bool. use positive negative momentum. ADDS 1 STATE
        :param pnm_beta: float. Manages the amplitude of the noise introduced by positive negative momentum. Negative values are valid.
        :param use_slow_ema: bool. use slow ema like that from AdEMAMix. ADDS 1 STATE
        :param slow_ema_alpha: float. usually between 4 and 10 would work well. The multipler for application of the slow ema to the update.
        :param slow_ema_beta: float. coefficient used for computing running slow average of gradient.
        :param slow_ema_t_alpha_beta: Optional[float]. total number of iterations is preferred when needed. The warmup of slow_ema_alpha and slow_ema_beta over iterations. Results in more stablity.
        :param diff_amp: float. Accelerate the difference between the current and past gradient by this multiplicative value. 0 is off. ADDS 2 STATES
        :param diff_amp_beta: float. Coefficient used for computing running average of the current and past gradients
        :param eps: float. the maximum eps value for adaptive eps. Eps is the term added to the denominator outside of the root operation to improve numerical stability.
        :param eps2: float. used to multiple the grad rms for determining adaptive eps.
        :param eps_floor: float. term used to determine the floor for adaptive eps.
        :param update_clipping: bool. Apply update clipping using root mean square of the gradient, similar to Adafactor. Advise disabling gradient clipping (clip=0.0).
        :param rectify_variance: bool. Rectify variance as per RAdam - https://arxiv.org/abs/1908.03265 (Default: false)
        :param n_sma_threshold: int. Simple moving average threshold for variance rectification (recommended is 5) (Default: 5).
        :param degenerated_to_sgd: bool. degenerated to SGD. (Default: false)
    """

    def __init__(
        self,
        params: PARAMETERS,
        lr: float = 1.4e-4,
        betas: BETAS = (0.975, 0.999),
        weight_decay: float = 0.0005,
        weight_decouple: bool = True,
        lr_decouple: bool = False,
        max_lr: float = 0.0,
        stable_decay: bool = True,
        fixed_decay: bool = False,
        clip: float = 0.01,
        clip_eps: float = 1e-3,
        amp_fac: float = 5.0,
        centralize_gradients: int = 0,
        normalize_gradients: int = 0,
        norm_loss_factor: float = 0.0005,
        norm_loss_eps: float = 1e-8,
        use_softplus: bool = True,
        beta_softplus: float = 50.0,
        threshold_softplus: float = 0.0,
        use_lookahead: bool = False,
        lookahead_merge_time: int = 5,
        lookahead_blending_alpha: float = 0.5,
        adam_debias: bool = False,
        use_pnm: bool = False,
        pnm_beta: float = 0.1,
        amsgrad: bool = False,
        use_slow_ema: bool = False,
        slow_ema_beta: float = 0.9998,
        slow_ema_alpha: float = 3.0,
        slow_ema_t_alpha_beta: Optional[float] = None,
        diff_amp: float = 0.0,
        diff_amp_beta: float = 0.999,
        eps: float = 1e-8,
        eps2: float = 0.01,
        eps_floor: float = 1e-16,
        update_clipping: bool = False,
        rectify_variance: bool = False,
        n_sma_threshold: int = 5,
        degenerated_to_sgd: bool = False,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_betas(betas)
        self.validate_range(pnm_beta, 'pnm_beta', -1.0, 1.0, range_type='[]')
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_non_negative(max_lr, 'max_lr')
        self.validate_non_negative(clip, 'clip')
        self.validate_non_negative(clip, 'clip_eps')
        self.validate_non_negative(amp_fac, 'amp_fac')
        self.validate_non_negative(lookahead_blending_alpha, 'lookahead_blending_alpha')
        self.validate_non_negative(lookahead_merge_time, 'lookahead_merge_time')
        self.validate_non_negative(beta_softplus, 'beta_softplus')
        self.validate_non_negative(threshold_softplus, 'threshold_softplus')
        self.validate_non_negative(norm_loss_factor, 'norm_loss_factor')
        self.validate_non_negative(slow_ema_alpha, 'slow_ema_alpha')
        self.validate_non_negative(diff_amp, 'diff_amp')
        self.validate_non_negative(eps, 'eps')
        self.validate_non_negative(eps2, 'eps2')
        self.validate_non_negative(eps_floor, 'eps_floor')
        self.validate_range(diff_amp_beta, 'diff_amp_beta', 0.0, 1.0, range_type='[]')
        self.validate_range(slow_ema_beta, 'slow_ema_beta', 0.0, 1.0, range_type='[]')

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
            'centralize_gradients': centralize_gradients,
            'normalize_gradients': normalize_gradients,
            'norm_loss_factor': norm_loss_factor,
            'norm_loss_eps': norm_loss_eps,
            'use_softplus': use_softplus,
            'beta_softplus': beta_softplus,
            'threshold_softplus': threshold_softplus,
            'use_lookahead': use_lookahead,
            'lookahead_merge_time': lookahead_merge_time,
            'lookahead_blending_alpha': lookahead_blending_alpha,
            'adam_debias': adam_debias,
            'use_pnm': use_pnm,
            'pnm_beta': pnm_beta,
            'amsgrad': amsgrad,
            'use_slow_ema': use_slow_ema,
            'slow_ema_beta': slow_ema_beta,
            'slow_ema_alpha': slow_ema_alpha,
            'slow_ema_t_alpha_beta': slow_ema_t_alpha_beta,
            'diff_amp': diff_amp,
            'diff_amp_beta': diff_amp_beta,
            'eps': eps,
            'eps2': eps2,
            'eps_floor': eps_floor,
            'update_clipping': update_clipping,
            'rectify_variance': rectify_variance,
            'n_sma_threshold': n_sma_threshold,
            'degenerated_to_sgd': degenerated_to_sgd,
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
        self.threshold_softplus = threshold_softplus
        self.norm_loss_factor = norm_loss_factor
        self.norm_loss_eps = norm_loss_eps
        self.lr_decouple = lr_decouple
        self.weight_decay = weight_decay
        self.weight_decouple = weight_decouple
        self.max_lr = max_lr
        self.fixed_decay = fixed_decay
        self.clip = clip
        self.clip_eps = clip_eps
        self.amp_fac = amp_fac
        self.use_slow_ema = use_slow_ema
        self.slow_ema_beta = slow_ema_beta
        self.slow_ema_alpha = slow_ema_alpha
        self.slow_ema_t_alpha_beta = slow_ema_t_alpha_beta
        self.diff_amp = diff_amp
        self.diff_amp_beta = diff_amp_beta
        self.eps = eps
        self.eps2 = eps2
        self.eps_floor = eps_floor
        self.update_clipping = update_clipping
        self.rectify_variance = rectify_variance
        self.n_sma_threshold = n_sma_threshold
        self.degenerated_to_sgd = degenerated_to_sgd

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'CompassPlus'
    
    @torch.no_grad()
    def reset(self):
        for group in self.param_groups:
            group['step'] = 0

            beta1, beta2 = group["betas"]

            for p in group['params']:
                state = self.state[p]

                grad = p.grad

                # Exponential moving average of gradient values
                if beta1 > 0.0: # save memory in case beta1 is 0.0
                    state['ema'] = torch.zeros_like(p)
                else: 
                    state['ema'] = None

                # Exponential moving average of squared gradient values
                state["ema_squared"] = torch.zeros_like(p)

                if self.use_pnm:
                    state['neg_ema'] = torch.zeros_like(p)

                if self.use_lookahead:
                    state['lookahead_params'] = p.clone()

                if self.amsgrad:
                    state["max_ema_squared"] = torch.zeros_like(p)

                if self.use_slow_ema:
                    state['ema_slow'] = torch.zeros_like(p)

                # Previous grad
                if self.diff_amp:
                    state["ema_diff"] = torch.zeros_like(p)
                    state["previous_grad"] = grad.clone().mul_(-1.0)
    
    @staticmethod
    def get_rms(x: torch.Tensor) -> float:
        r"""Get RMS."""
        return x.norm(2) / math.sqrt(x.numel())
    
    @staticmethod
    def schedule_alpha(t_alpha_beta3: Optional[float], step: int, alpha: float) -> float:
        if t_alpha_beta3 is None:
            return alpha
        return min(step * alpha / t_alpha_beta3, alpha)

    @staticmethod
    def schedule_beta3(t_alpha_beta3: Optional[float], step: int, beta1: float, beta3: float) -> float:
        if t_alpha_beta3 is None:
            return beta3

        # Add eps to prevent log 0
        log_beta1, log_beta3 = math.log(beta1 + 1e-8), math.log(beta3)

        return min(
            math.exp(
                log_beta1 * log_beta3 / ((1.0 - step / t_alpha_beta3) * log_beta3 + (step / t_alpha_beta3) * log_beta1)
            ),
            beta3,
        )

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

            beta1, beta2 = group["betas"]

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
                    if beta1 > 0.0: # save memory in case beta1 is 0.0
                        state['ema'] = torch.zeros_like(p)
                    else: 
                        state['ema'] = None

                    # Exponential moving average of squared gradient values
                    state["ema_squared"] = torch.zeros_like(p)

                    if self.use_pnm:
                        state['neg_ema'] = torch.zeros_like(p)

                    if self.use_lookahead:
                        state['lookahead_params'] = p.clone()

                    if self.amsgrad:
                        state["max_ema_squared"] = torch.zeros_like(p)

                    if self.use_slow_ema:
                        state['ema_slow'] = torch.zeros_like(p)

                    # Previous grad
                    if self.diff_amp:
                        state["ema_diff"] = torch.zeros_like(p)
                        state["previous_grad"] = grad.clone().mul_(-1.0)

                if p.dtype in {torch.float16, torch.bfloat16}:
                    p_fp32 = p.clone().to(torch.float32)
                    grad = grad.to(torch.float32)

                # Apply Adaptive Gradient Clipping (AGC)
                if self.clip > 0.0:
                    grad.copy_(agc(p_fp32, grad, self.clip_eps, self.clip))

                # Apply gradient centralization & normalization
                if self.centralize_gradients in {1,3}:
                    centralize_gradient(grad, gc_conv_only=False)

                if self.normalize_gradients in {1,3}:
                    normalize_gradient(grad)

        if param_size == 0:
            raise ZeroParameterSizeError()

        # Phase 2
        for group in self.param_groups:
            beta1, beta2 = group["betas"]

            # bias correction step size
            # soft warmup
            bias_correction2_sq: float = math.sqrt(self.debias(beta2, group['step']))

            if self.use_slow_ema:
                # Scale with amp fac for consistency
                slow_ema_alpha_t: float = self.schedule_alpha(self.slow_ema_t_alpha_beta, group['step'], self.slow_ema_alpha * self.amp_fac)
                slow_ema_beta3_t: float = self.schedule_beta3(self.slow_ema_t_alpha_beta, group['step'], beta1, self.slow_ema_beta)

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad

                state = self.state[p]

                ema_squared = state["ema_squared"]

                if self.use_pnm:
                    if group['step'] % 2 == 1:
                        ema, neg_ema = state['ema'], state['neg_ema']
                    else:
                        ema, neg_ema = state['neg_ema'], state['ema']
                else:
                    ema = state["ema"]

                if self.use_slow_ema:
                    ema_slow = state['ema_slow']



                # unpack
                if p.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.to(torch.float32)
                    ema_squared = ema_squared.to(torch.float32)

                    if beta1 > 0.0: # save memory in case beta1 is 0.0
                        ema = ema.to(torch.float32)

                    if self.use_pnm:
                        neg_ema = neg_ema.to(torch.float32)

                    if self.use_slow_ema:
                        ema_slow = ema_slow.to(torch.float32)

                # Decay the first and second moment running average coefficient
                if beta1 > 0.0: # save memory in case beta1 is 0.0
                    # ema = ema + (1 - beta1) * grad
                    ema.mul_(beta1).add_(grad, alpha=1.0 - beta1)  # fmt: skip
                else:
                    ema = grad

                # Natural grad
                if self.diff_amp > 0.0 or self.use_pnm or self.use_slow_ema:
                    nat_grad = grad.clone()
                    nat_grad_amp = nat_grad.add(ema, alpha=self.amp_fac)
                else:
                    nat_grad_amp = grad
                    nat_grad = grad

                if self.use_pnm:
                    noise_norm: float = math.sqrt((1.0 + self.pnm_beta) ** 2 + self.pnm_beta ** 2)
                    adjusted_ema = ema.mul(1.0 + self.pnm_beta).add_(neg_ema, alpha=-self.pnm_beta).mul_(1.0 / noise_norm)
                else:
                    adjusted_ema = ema

                # grad = grad + ema * amplification_factor
                grad.add_(adjusted_ema, alpha=self.amp_fac)

                if self.use_slow_ema:
                    ema_slow.mul_(slow_ema_beta3_t).add_(nat_grad, alpha=1.0 - slow_ema_beta3_t)
                    grad.add_(ema_slow, alpha=slow_ema_alpha_t)

                if self.diff_amp > 0.0:
                    grad_diff = state["previous_grad"]
                    ema_diff = state['ema_diff']

                    if p.dtype in {torch.float16, torch.bfloat16}:
                        grad_diff = grad_diff.to(torch.float32)
                        ema_diff = ema_diff.to(torch.float32)

                    # grad_diff will contain the difference between prev grad and current grad
                    grad_diff.add_(nat_grad)

                    # Smooth the difference between previous grad and current grad
                    ema_diff.mul_(self.diff_amp_beta).add_(grad_diff, alpha=1 - self.diff_amp_beta)

                    # Scale with amp fac for consistency
                    grad.add_(ema_diff, alpha=self.diff_amp * self.amp_fac)

                    if p.dtype in {torch.float16, torch.bfloat16}:
                        copy_stochastic_(state["previous_grad"], -nat_grad)
                        copy_stochastic_(state["ema_diff"], ema_diff)
                    else:
                        state["previous_grad"].copy_(-nat_grad)

                # ema_squared = ema + (1 - beta2) * grad ** 2
                ema_squared.mul_(beta2).addcmul_(nat_grad_amp, nat_grad_amp, value=1.0 - beta2)
                if self.stable_decay:
                    ema_squared_sum += (ema_squared / bias_correction2_sq).sum()

                # pack
                if p.dtype in {torch.float16, torch.bfloat16}:
                    copy_stochastic_(state["ema_squared"], ema_squared)

                    if self.use_pnm:
                        if group['step'] % 2 == 1:
                            if beta1 > 0.0:
                                copy_stochastic_(state["ema"], ema)
                        else:
                            # neg_ema is previous grad if beta1 is 0.0
                            copy_stochastic_(state["neg_ema"], ema)

                    else:
                        if beta1 > 0.0:
                            copy_stochastic_(state["ema"], ema)

                    if self.use_slow_ema:
                        copy_stochastic_(state["ema_slow"], ema_slow)

        if self.stable_decay:
            ema_squared_normalized = math.sqrt(ema_squared_sum / param_size)
        else:
            ema_squared_normalized = ema_squared_sum

        # Phase 3 - Weight decay and parameter update
        for group in self.param_groups:
            # bias correction step size
            # soft warmup
            bias_correction1: float = self.debias(beta1, group['step'])
            bias_correction2_sq: float = math.sqrt(self.debias(beta2, group['step']))

            eps_p2: float = math.pow(group['eps'], 2)

            step_size, n_sma = self.get_rectify_step_size(
                is_rectify=self.rectify_variance,
                step=group['step'],
                lr=group['lr'],
                beta2=beta2,
                n_sma_threshold=self.n_sma_threshold,
                degenerated_to_sgd=self.degenerated_to_sgd,
            )

            step_size = self.apply_adam_debias(
                adam_debias=self.adam_debias,
                step_size=step_size,
                bias_correction1=bias_correction1,
            )
        
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad

                state = self.state[p]

                p_fp32 = p
                ema_squared = state["ema_squared"]

                if p.dtype in {torch.float16, torch.bfloat16}:
                    p_fp32 = p.clone().to(torch.float32)
                    grad = grad.to(torch.float32)
                    ema_squared = ema_squared.to(torch.float32)

                # Basically should allow smaller eps whenever grad is small, so eps doesn't have outsized influence
                rms_grad = grad.pow(2).mean().sqrt_()
                current_eps = max(min(rms_grad.item() * self.eps2, self.eps), self.eps_floor) # Set a floor for eps to avoid NaN
 
                # lr scaler + eps to prevent zero division
                # de_nom = exp_avg_sq.sqrt() + group['eps']
                if self.amsgrad:
                    max_ema_squared = state['max_ema_squared']

                    if p.dtype in {torch.float16, torch.bfloat16}:
                        max_ema_squared = max_ema_squared.to(torch.float32)
                        
                    torch.max(max_ema_squared, ema_squared, out=max_ema_squared)
                    if self.rectify_variance:
                        de_nom = max_ema_squared.sqrt().add_(current_eps)
                    else:
                        de_nom = (max_ema_squared.sqrt() / bias_correction2_sq).add_(current_eps)

                    if p.dtype in {torch.float16, torch.bfloat16}:
                        copy_stochastic_(state['max_ema_squared'], max_ema_squared)
                else:
                    if self.rectify_variance:
                        de_nom = ema_squared.sqrt().add_(current_eps)
                    else:
                        de_nom = (ema_squared.sqrt() / bias_correction2_sq).add_(current_eps)

                if self.use_softplus:
                    de_nom = softplus(de_nom, beta=self.beta_softplus, threshold=self.threshold_softplus if self.threshold_softplus != 0 else current_eps)

                if self.update_clipping:
                    rms = grad.pow(2).div_(ema_squared.maximum(eps_p2)).mean().sqrt_()
                    step_size = step_size / max(1, rms.item())

                if not self.rectify_variance or step_size > 0 or n_sma >= self.n_sma_threshold:
                    if self.weight_decouple:
                        # Perform stepweight decay
                        p_fp32.mul_(1.0 - (1.0 if self.fixed_decay else step_size if not self.lr_decouple else step_size / self.max_lr) * self.weight_decay * (1.0 / ema_squared_normalized if self.stable_decay else 1.0))
                    elif self.weight_decay > 0.0 and not self.use_slow_ema:
                        grad.add_(p_fp32, alpha=self.weight_decay)

                    if self.norm_loss_factor > 0.0:
                        # norm loss
                        correction = 2.0 * self.norm_loss_factor * (1.0 - 1.0 / unit_norm(p_fp32).add_(self.norm_loss_eps))
                        p_fp32.mul_(1.0 - step_size * correction)

                if not self.rectify_variance or n_sma >= self.n_sma_threshold:
                    update = grad.div(de_nom)
                else:
                    update = grad

                # Apply weight decay like AdEMAMix
                if not self.rectify_variance or step_size > 0 or n_sma >= self.n_sma_threshold:
                    if self.weight_decay > 0.0 and self.use_slow_ema and not self.weight_decouple:
                        update.add_(p_fp32, alpha=self.weight_decay)

                if self.centralize_gradients in {2,3}:
                    centralize_gradient(update, gc_conv_only=False)

                if self.normalize_gradients in {2,3}:
                    normalize_gradient(update) 

                # p = p - lr * grad / de_nom
                if not self.rectify_variance or step_size > 0 or n_sma >= self.n_sma_threshold:
                    p_fp32.add_(update, alpha=-step_size)

                # pack
                if p.dtype in {torch.float16, torch.bfloat16}:
                    copy_stochastic_(p, p_fp32)

        if self.use_lookahead:
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