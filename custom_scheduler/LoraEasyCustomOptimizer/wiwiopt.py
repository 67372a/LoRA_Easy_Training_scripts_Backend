import torch
from torch.optim import Optimizer
from typing import Optional, Tuple, Iterable, Literal
import math

def copy_stochastic_(target: torch.Tensor, source: torch.Tensor):
    """
    Fast stochastic rounding implementation for half-precision tensors.
    """
    with torch.no_grad():
        result = torch.randint_like(
            source,
            dtype=torch.int32,
            low=0,
            high=(1 << 16),
        )
        result.add_(source.view(dtype=torch.int32))
        result.bitwise_and_(-65536)
        target.copy_(result.view(dtype=torch.float32))

# Newton-Schulz iteration coefficients for orthogonalization
# From https://kexue.fm/archives/11059
NS_COEFFS = [
    (8.287212018145622, -23.59588651909882, 17.300387312530923),
    (4.107059111542197, -2.9478499167379084, 0.54484310829266),
    (3.9486908534822938, -2.908902115962947, 0.5518191394370131),
    (3.3184196573706055, -2.488488024314878, 0.5100489401237208),
    (2.3006520199548186, -1.6689039845747518, 0.4188073119525678),
    (1.8913014077874002, -1.2679958271945908, 0.37680408948524996),
    (1.875, -1.25, 0.375)
]

def reshape_to_2d(grad):
    """Reshape a tensor to 2D for matrix operations."""
    dimcount = len(grad.shape)
    if dimcount > 2:
        grad_2d = grad.reshape(len(grad), -1)
    elif dimcount < 2:
        grad_2d = grad.reshape(1, -1)
    else:
        grad_2d = grad
    return grad_2d


@torch.no_grad()
def orthogonalize(M: torch.Tensor, num_ns_steps=len(NS_COEFFS), ortho_dtype=None) -> torch.Tensor:
    """Orthogonalize a matrix via 5th order Newton-Schulz iteration."""
    if ortho_dtype is not None:
        orig_dtype = M.dtype
        M = M.to(ortho_dtype)
    
    transpose = M.shape[0] < M.shape[1]
    if transpose:
        M = M.T
    
    # Pre-calculate Identity matrix for better performance
    I = torch.eye(M.shape[1], dtype=M.dtype, device=M.device)
    
    for a, b, c in NS_COEFFS[:num_ns_steps]:
        # Faster normalization
        M = M / (torch.linalg.norm(M).clamp_min_(1e-8))
        A = M.T @ M
        # 5th order Newton-Schulz update
        M = M @ (a * I + b * A + c * A @ A)
    
    if transpose:
        M = M.T
    
    if ortho_dtype is not None:
        M = M.to(orig_dtype)
    return M


@torch.no_grad()
def sanger_update(X: torch.Tensor, V: torch.Tensor, lr: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """Single step of Sanger's Rule (Generalized Oja's rule) for online PCA."""
    X_norm = X / X.norm().clamp_min(1e-8)
    Y = X_norm @ V
    V_update = X_norm.T @ Y - V @ torch.triu(Y.T @ Y)
    V_new = V + lr * V_update
    Y_new = X @ V_new
    return V_new, Y_new


class WiwiOpt(Optimizer):
    r"""
    WiwiOpt (V1.1).

    A gradient descent optimizer that combines several stabilization & acceleration techniques to produce
    high-signal stable parameter updates.

    WiwiOpt works by:
    1. RMS-based gradient normalization: Incoming gradients are normalized
       by a polynomial-decay EMA of their per-row RMS, preventing exploding
       or vanishing gradient magnitudes.
    2. Egalitarian Gradient Descent (EGD) preconditioning: For 2D+
       parameters, a low-rank SVD approximation is used to precondition the
       gradient, equalizing contribution across singular directions.
    3. Polynomial-schedule momentum: Momentum and accumulation use
       polynomial decay schedules (``1 / step^beta``) instead of fixed
       betas, providing smoothing that naturally increases over training.
    4. Newton-Schulz orthogonalization (Muon): The effective gradient is
       orthogonalized via Newton-Schulz iteration for multi-dimensional
       parameters, producing direction-pure updates.
    5. NorMuon scaling: After orthogonalization, the update is re-scaled
       using a tracked second-moment estimate to maintain consistent update
       magnitudes, then re-projected to preserve the original norm.
    6. Projection re-scaling: The orthogonalized step is re-scaled by its
       projection onto the un-orthogonalized effective gradient, preserving
       meaningful magnitude information.
    7. Cautious masking: Updates are masked so that only components
       agreeing in sign with the raw gradient are kept, preventing
       counterproductive steps.
    8. Dynamic learning rate: Per-parameter learning rate adjustment based
       on the alignment between the EMA of parameter deltas and the EMA of
       their norms, optionally boosted by an ``atan2``-based scaling factor.

    Arguments:
        params (iterable): Iterable of parameters to optimize.
        lr (float): Learning rate (default: 1e-3).
        betas (Tuple[float, float, float] or Tuple[float, float]):
            Exponents for the de-biased beta schedules.
            ``beta1`` controls momentum and gradient accumulation decay,
            ``beta2`` controls the variance tracker and NorMuon second-moment
            decay, and ``beta3`` controls the dynamic learning rate EMAs.
            (default: (0.95, 0.995, 0.99)).
        eps (float): Numerical stability term for divisions and clamps
            (default: 1e-16).
        weight_decay (float): Decoupled weight decay coefficient
            (default: 0.0).
        normuon (bool): Apply NorMuon second-moment scaling after
            orthogonalization to stabilize update magnitudes
            (default: True).
        use_compile (bool): Use ``torch.compile`` on the orthogonalization
            and SVD functions for faster execution (default: True).
        ortho_dtype (str or None): Data type for Newton-Schulz
            orthogonalization. Accepts ``None`` (defaults to float32) or a
            string like ``"torch.bfloat16"`` (default: None).
        stochastic_fp (bool): Use stochastic rounding when parameters are
            stored in bfloat16, reducing quantization bias (default: True).
        dynamic_lr (bool): Enable per-row dynamic learning rate
            adjustment based on delta alignment (default: True).
        dynamic_lr_boost (bool): When ``dynamic_lr`` is enabled, apply an
            additional ``atan2``-based boost factor that amplifies the
            learning rate when parameter deltas are large relative to their
            directional EMA (default: True).
        egd (bool): Enable Egalitarian Gradient Descent preconditioning via
            low-rank SVD for parameters with 2+ dimensions, equalizing
            gradient contribution across singular directions
            (default: True).
        egd_oja (bool): Enables a lightweight approximation
            of EGD using Sanger's rule (Generalized Oja's rule) in place 
            of full SVD tracking (default: True).
    """

    def __init__(
        self,
        params: Iterable[torch.Tensor],
        lr: float = 1e-3,
        betas: Tuple[float, float, float] = (0.95, 0.995, 0.99),
        eps: float = 1e-16,
        weight_decay: float = 0.0,
        normuon: bool = True,
        use_compile: bool = True,
        ortho_dtype: Optional[str] = None,
        stochastic_fp: bool = True,
        dynamic_lr: bool = True,
        dynamic_lr_boost: bool = True,
        egd: bool = True, 
        egd_oja: bool = True,
        **kwargs
    ):
        if len(betas) == 2:
            betas = (betas[0], betas[0], betas[1])
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= betas[2] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 2: {betas[2]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        if ortho_dtype is None:
            ortho_dtype = torch.float32
        elif isinstance(ortho_dtype, str):
            dtype_name = ortho_dtype.split('.')[-1]
            ortho_dtype = getattr(torch, dtype_name)

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            normuon=normuon,
            use_compile=use_compile,
            ortho_dtype=ortho_dtype,
            stochastic_fp=stochastic_fp,
            dynamic_lr=dynamic_lr,
            dynamic_lr_boost=dynamic_lr_boost,
            egd=egd,
            egd_oja=egd_oja,
        )
        self.ortho_func = torch.compile(orthogonalize, mode="reduce-overhead") if use_compile else orthogonalize
        if egd:
            if egd_oja:
                self.egd_func = torch.compile(sanger_update, mode="reduce-overhead") if use_compile else sanger_update
            else:
                self.egd_func = torch.compile(torch.svd_lowrank, mode="reduce-overhead") if use_compile else torch.svd_lowrank
        super(WiwiOpt, self).__init__(params, defaults)

    @torch.no_grad()
    def reset(self):
        pass

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            if len(group['betas']) == 2:
                beta1, beta2, beta3 = group['betas'][0], group['betas'][0], group['betas'][1]
            else:
                beta1, beta2, beta3 = group['betas']
            eps = group['eps']
            weight_decay = group['weight_decay']
            stochastic_fp = group['stochastic_fp']
            egd = group['egd']
            egd_oja = group['egd_oja']
            dynamic_lr = group['dynamic_lr']
            dynamic_lr_boost = group['dynamic_lr_boost']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('WiwiOpt does not support sparse gradients')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['polyak'] = torch.ones_like(p.mean(dim=-1, keepdim=True), memory_format=torch.preserve_format)
                    state['accum'] = torch.ones_like(p.mean(dim=-1, keepdim=True), memory_format=torch.preserve_format)
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if dynamic_lr:
                        state['delta_ema'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        state['delta_norm_ema'] = torch.zeros_like(p.mean(dim=-1, keepdim=True), memory_format=torch.preserve_format)
                    if p.ndim >= 1 and group["normuon"]:
                        grad_2d = reshape_to_2d(grad)
                        state['normuon_second_momentum'] = torch.zeros(grad_2d.shape[0], 1, device=p.device, dtype=p.dtype)

                state['step'] += 1
                step = state['step']

                polyak = state['polyak']
                accum = state['accum']
                exp_avg = state['exp_avg']
                if p.ndim >= 1 and group["normuon"]:
                    normuon_second_momentum = state['normuon_second_momentum']

                # Mixed precision handling
                use_stochastic = stochastic_fp and p.dtype in {torch.bfloat16}
                if use_stochastic:
                    p_work = p.detach().to(torch.float32)
                    grad_work = grad.detach().to(torch.float32)
                    polyak_work = polyak.detach().to(torch.float32)
                    accum_work = accum.detach().to(torch.float32)
                    exp_avg_work = exp_avg.detach().to(torch.float32)
                    if dynamic_lr:
                        delta_ema_work = state['delta_ema'].detach().to(torch.float32)
                        delta_norm_ema_work = state['delta_norm_ema'].detach().to(torch.float32)
                    if p.ndim >= 1 and group["normuon"]:
                        normuon_z = normuon_second_momentum.detach().to(torch.float32)
                else:
                    p_work = p.detach()
                    grad_work = grad.detach()
                    polyak_work = polyak.detach()
                    accum_work = accum.detach()
                    exp_avg_work = exp_avg.detach()
                    if dynamic_lr:
                        delta_ema_work = state['delta_ema'].detach()
                        delta_norm_ema_work = state['delta_norm_ema'].detach()
                    if p.ndim >= 1 and group["normuon"]:
                        normuon_z = normuon_second_momentum.detach()

                poly_beta1 = ((beta1**(step) - beta1) / (beta1**(step) - 1.0))
                poly_beta2 = ((beta2**(step) - beta2) / (beta2**(step) - 1.0))
                poly_beta3 = ((beta3**(step) - beta3) / (beta3**(step) - 1.0))

                grad_rms = grad_work.pow(2).mean(dim=-1, keepdim=True)
                accum_work.lerp_(grad_rms, 1. - poly_beta1)

                grad_work.div_(accum_work.sqrt().clamp_min_(eps)).clamp_(-step, step)

                if egd and p_work.ndim >= 2:
                    grad_work_2d = reshape_to_2d(grad_work)
                    m_dim, n_dim = grad_work_2d.size(0), grad_work_2d.size(1)
                    current_rank = min(128, m_dim, n_dim)
                    
                    if current_rank > 0:
                        if egd_oja:
                            if 'oja_basis' not in state:
                                track_u = (m_dim < n_dim)
                                feature_dim = m_dim if track_u else n_dim
                                basis = torch.randn(feature_dim, current_rank, device=p_work.device, dtype=p_work.dtype)
                                basis, _ = torch.linalg.qr(basis)
                                state['oja_basis'] = basis
                                
                            track_u = (m_dim < n_dim)
                            oja_basis_work = state['oja_basis']
                            if use_stochastic:
                                oja_basis_work = oja_basis_work.detach().float()
                                
                            X_for_oja = grad_work_2d.T if track_u else grad_work_2d
                            if use_stochastic:
                                X_for_oja = X_for_oja.float()
                                
                            try:
                                oja_basis_work, Y_new = self.egd_func(X_for_oja, oja_basis_work, 1. - poly_beta1)
                                
                                if track_u:
                                    V_basis = Y_new / Y_new.norm(dim=0, keepdim=True).clamp_min_(eps)
                                    grad_precond = oja_basis_work @ V_basis.T
                                else:
                                    U_basis = Y_new / Y_new.norm(dim=0, keepdim=True).clamp_min_(eps)
                                    grad_precond = U_basis @ oja_basis_work.T
                                    
                                if use_stochastic:
                                    grad_precond = grad_precond.to(p_work.dtype)
                                    copy_stochastic_(state['oja_basis'], oja_basis_work)
                                else:
                                    state['oja_basis'].copy_(oja_basis_work)
                                    
                                grad_work = grad_precond.view_as(p_work)
                            except RuntimeError:
                                pass
                        else:
                            try:
                                # Use float32 for SVD stability if it was half precision
                                dtype_orig = grad_work_2d.dtype
                                grad_f32 = grad_work_2d.float()
                                
                                U, S, _ = self.egd_func(grad_f32, q=current_rank)
                                
                                U = U.to(dtype_orig)
                                S = S.to(dtype_orig)
                                
                                S = torch.maximum(S, torch.tensor(eps, device=S.device, dtype=S.dtype))
                                S_inv = 1.0 / S
                                
                                aux = (U * S_inv.unsqueeze(0)) @ U.mT
                                grad_precond = aux @ grad_work_2d
                                
                                grad_work = grad_precond.view_as(p_work)
                            except RuntimeError:
                                # Fallback if SVD fails to converge (rare)
                                pass

                # AdaBelief-like variance tracker for effective gradient normalization
                polyak_work.lerp_((grad_work - exp_avg_work).pow(2).mean(dim=-1, keepdim=True), weight=1. - poly_beta2)

                # Momentumize the pre-conditioned gradient
                exp_avg_work.lerp_(grad_work, weight=1. - poly_beta1)
                g_eff_mom = grad_work.lerp(exp_avg_work, weight=poly_beta1).div(polyak_work.sqrt().clamp_min_(eps))

                if p_work.ndim >= 1:
                    full_step_2d = reshape_to_2d(g_eff_mom)
                    
                    # Newton-Schulz Orthogonalization (Muon)
                    Q = self.ortho_func(full_step_2d, ortho_dtype=group["ortho_dtype"])

                    # NorMuon update & re-norm
                    if group["normuon"]:
                        vnorm = Q.norm(dim=(-2, -1), keepdim=True)

                        v_mean = torch.mean(Q * Q, dim=-1, keepdim=True)
                        normuon_z.lerp_(v_mean, 1 - poly_beta2)
                        step_size = normuon_z.sqrt().clamp_min_(eps)
                        Q.div_(step_size)

                        vnorm_new = Q.norm(dim=(-2, -1), keepdim=True)
                        Q = Q * (vnorm / vnorm_new.clamp_min(eps))

                    final_step = Q.view_as(p_work)

                    # Re-scaling: final_step functionally sums to 1.
                    # We re-scale it to the magnitude of the projection onto the un-orthogonalized effective gradient
                    scale_factor = (g_eff_mom * final_step).sum()
                    final_step.mul_(scale_factor)
                else:
                    final_step = g_eff_mom

                # Cautious masking
                scale_factor_mask = (grad_work * final_step > 0).to(final_step.dtype)
                mask_mean = scale_factor_mask.mean().clamp_min_(1e-3)
                scale_factor_mask.div_(mask_mean)
                final_step.mul_(scale_factor_mask)

                # Dynamic Learning Rate Adjustment
                if dynamic_lr:
                    if step > 1:
                        # True norm of EMA of deltas vs EMA of accumulated norms of deltas
                        alignment_ratio = delta_ema_work.norm(dim=-1, keepdim=True) / delta_norm_ema_work.clamp_min(eps)
                        # Parameter-wise update scaling
                        if dynamic_lr_boost:
                            update_ratio = delta_norm_ema_work.atan2(delta_ema_work.abs()).mul_(1.27323954474)
                            lr_adj = alignment_ratio * update_ratio
                        else:
                            lr_adj = alignment_ratio
                    else:
                        lr_adj = torch.ones_like(p.mean())
                        
                    final_step.mul_(lr_adj)
                    
                    # Update EMAs
                    current_norm = final_step.norm(dim=-1, keepdim=True)
                    delta_ema_work.lerp_(final_step, 1. - poly_beta3)
                    delta_norm_ema_work.lerp_(current_norm, 1. - poly_beta3)

                # Apply Update
                if weight_decay != 0:
                    p_work.add_(p_work * lr_adj if dynamic_lr else p_work, alpha=-lr * weight_decay)
                
                p_work.add_(final_step, alpha=-lr)

                # State Sync
                if use_stochastic:
                    copy_stochastic_(polyak, polyak_work)
                    copy_stochastic_(accum, accum_work)
                    copy_stochastic_(exp_avg, exp_avg_work)
                    if dynamic_lr:
                        copy_stochastic_(state['delta_ema'], delta_ema_work)
                        copy_stochastic_(state['delta_norm_ema'], delta_norm_ema_work)
                    copy_stochastic_(p, p_work)
                    if p.ndim >= 1 and group["normuon"]:
                        copy_stochastic_(normuon_second_momentum, normuon_z)
                else:
                    polyak.copy_(polyak_work)
                    accum.copy_(accum_work)
                    exp_avg.copy_(exp_avg_work)
                    if dynamic_lr:
                        state['delta_ema'].copy_(delta_ema_work)
                        state['delta_norm_ema'].copy_(delta_norm_ema_work)
                    p.copy_(p_work)
                    if p.ndim >= 1 and group["normuon"]:
                        normuon_second_momentum.copy_(normuon_z)

        return loss
