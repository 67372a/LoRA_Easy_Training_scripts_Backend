# Source: https://github.com/Clybius/Personalized-Optimizers

# WarpAdam: Adam with a learnable low-rank distortion matrix P
#
# WarpAdam (arXiv:2409.04244) linearly warps every gradient with a learnable
# matrix P before it enters the Adam moments:
#
#   m_t = beta1 * m_{t-1} + (1 - beta1) * (P @ grad_t)
#   v_t = beta2 * v_{t-1} + (1 - beta2) * (P @ grad_t)^2
#   w_{t+1} = w_t - lr / (sqrt(v_hat_t) + eps) * m_hat_t
#
# P is stored per parameter tensor as a low-rank, identity-anchored factor:
#
#   P = I + U @ V^T,   U = 0, V ~ N(0, 1),  U, V in R^{m x r}
#
# U is zero-initialized so P starts at exactly the identity and WarpAdam is
# exactly Adam until the factors learn; V is random-initialized to break the
# stationary point of the meta-objective at the identity anchor (with
# U = V = 0 the meta-gradients vanish and P can never learn). Two
# meta-learning strategies are provided for the factors:
#
#   mode="online" (default): closure-free training on the one-step-ahead
#       gradient prediction loss  ||P @ g_{t-1} - g_t||^2 / ||g_{t-1}||^2,
#       i.e. the transition structure ("off-diagonal information transfer")
#       of the gradient process.
#
#   mode="meta": training on the Warped-Gradient-Descent meta-objective
#       min_P L(w - lr * update(P @ g)) -- the loss at the warped trial point,
#       evaluated with a second closure call (first-order in the Adam moments).

import math
import torch
from torch.optim import Optimizer
from typing import Callable, Iterable, Optional, Tuple


def copy_stochastic_(target: torch.Tensor, source: torch.Tensor):
    """
    Fast stochastic rounding implementation for half-precision tensors.
    Thanks to Nerogar for fast stochastic pytorch implementation:
    https://github.com/pytorch/pytorch/issues/120376#issuecomment-1974828905
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


def _to_rows(x: torch.Tensor) -> torch.Tensor:
    """Reshape to (m, n): m = the dim P mixes, n = the feature dim."""
    if x.dim() <= 1:
        return x.reshape(-1, 1)
    return x.reshape(x.shape[0], -1)


def _mix_dim(p: torch.Tensor) -> int:
    """Size of the mixing dimension (rows of P)."""
    return p.numel() if p.dim() <= 1 else p.shape[0]


class WarpAdam(Optimizer):
    r"""
    WarpAdam: Adam with a learnable low-rank distortion matrix P.

    Every gradient is linearly warped by a learnable matrix P before it
    enters the Adam first- and second-moment updates (paper eq. 2):

        m_t = beta1 * m_{t-1} + (1 - beta1) * (P @ grad_t)
        v_t = beta2 * v_{t-1} + (1 - beta2) * (P @ grad_t)^2
        w_{t+1} = w_t - lr / (sqrt(v_hat_t) + eps) * m_hat_t

    P is a per-parameter low-rank, identity-anchored distortion matrix:

        P = I + U @ V^T,   U = 0, V ~ N(0, 1),  U, V in R^{m x r}

    where m is the tensor's mixing dimension (first dim for >= 2D tensors,
    the full size for 1D tensors). U is zero-initialized so P = I exactly
    and WarpAdam is exactly Adam until the factors learn; V is
    random-initialized to break the stationary point of the meta-objective
    at the identity anchor (otherwise the factors would never leave
    U = V = 0).

    Two meta-learning strategies for P are provided:

    - mode="online" (default, no closure): after each step, U and V are
      updated with one SGD step on the one-step-ahead gradient prediction
      loss  ||P @ g_{t-1} - g_t||^2 / ||g_{t-1}||^2. P learns the
      transition structure of the gradient process, i.e. how gradient
      coordinates co-move (the paper's "transfer off-diagonal information").
      The norm normalization makes the meta-dynamics scale-invariant
      (stable for meta_lr * ||V||^2 < 2, independent of the gradient
      magnitude); meta_wd damps U and V back toward 0, keeping P near the
      identity.

    - mode="meta" (requires a closure): P is trained on the
      Warped-Gradient-Descent meta-objective  min_P L(w - lr * update),
      i.e. the loss at the warped trial point, evaluated with a second
      closure call per step (first-order in the Adam moments; the moments
      are treated as constants for the meta-gradient).

    Arguments:
        params (iterable): Iterable of parameters to optimize.
        lr (float): Learning rate (default: 1e-3).
        betas (Tuple[float, float]): Coefficients for momentum (beta1) and
            variance (beta2) EMAs. (default: (0.9, 0.999))
        eps (float): Denominator epsilon (default: 1e-8).
        weight_decay (float): Decoupled weight decay (default: 0.0).
        rank (int): Rank r of the P factors U, V (capped at m).
            rank=0 disables the warp entirely (plain Adam). (default: 16)
        mode (str): "online" (closure-free prediction loss) or "meta"
            (closure-based warped-gradient-descent objective). (default:
            "online")
        meta_lr (float): Learning rate for the U, V meta-update.
            meta_lr=0 disables the warp (plain Adam). (default: 1e-3)
        meta_wd (float): Damping on U and V toward 0 (identity anchor).
            (default: 1e-2)
        cautious_update (bool): Apply cautious masking to parameter updates
            (default: True).
        stochastic_fp (bool): Use stochastic rounding for BF16 tensors
            (default: True). Note: Only applies to BF16, not FP16.

    Example:
        >>> optimizer = WarpAdam(model.parameters(), lr=1e-3)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()

        >>> optimizer = WarpAdam(model.parameters(), mode="meta")
        >>> def closure():
        ...     optimizer.zero_grad()
        ...     loss = loss_fn(model(input), target)
        ...     loss.backward()
        ...     return loss
        >>> loss = optimizer.step(closure)
    """

    requires_closure = False

    def __init__(
        self,
        params: Iterable[torch.Tensor],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        rank: int = 16,
        mode: str = "online",
        meta_lr: float = 1e-3,
        meta_wd: float = 1e-2,
        cautious_update: bool = True,
        stochastic_fp: bool = True,
        **kwargs,
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if not isinstance(rank, int) or rank < 0:
            raise ValueError(f"Invalid rank: {rank}")
        if mode not in ("online", "meta"):
            raise ValueError(f"Invalid mode: {mode}")
        if not 0.0 <= meta_lr:
            raise ValueError(f"Invalid meta_lr: {meta_lr}")
        if not 0.0 <= meta_wd:
            raise ValueError(f"Invalid meta_wd: {meta_wd}")

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            rank=rank,
            mode=mode,
            meta_lr=meta_lr,
            meta_wd=meta_wd,
            cautious_update=cautious_update,
            stochastic_fp=stochastic_fp,
        )
        super(WarpAdam, self).__init__(params, defaults)
        self.requires_closure = any(g["mode"] == "meta" for g in self.param_groups)

    def _init_state(self, p: torch.Tensor, state: dict, group: dict):
        state["step"] = 0
        state["exp_avg"] = torch.zeros(
            p.shape, dtype=torch.float32, device=p.device
        )
        state["exp_avg_sq"] = torch.zeros(
            p.shape, dtype=torch.float32, device=p.device
        )
        if group["rank"] > 0 and group["meta_lr"] > 0:
            m = _mix_dim(p)
            r = min(group["rank"], m)
            state["U"] = torch.zeros(
                m, r, dtype=torch.float32, device=p.device,
                requires_grad=group["mode"] == "meta",
            )
            state["V"] = torch.zeros(
                m, r, dtype=torch.float32, device=p.device,
                requires_grad=group["mode"] == "meta",
            ).normal_(0.0, 1.0)

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        if closure is None and self.requires_closure:
            raise ValueError(
                "WarpAdam mode='meta' requires a closure that returns the "
                "loss (and runs backward)"
            )

        if self.requires_closure:
            with torch.enable_grad():
                loss = closure()
            if loss is None:
                raise ValueError(
                    "WarpAdam: closure returned None; it must return the loss"
                )
            return self._step_meta(loss, closure)

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            meta_lr = group["meta_lr"]
            meta_wd = group["meta_wd"]
            cautious_update = group["cautious_update"]
            stochastic_fp = group["stochastic_fp"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError(
                        "WarpAdam does not support sparse gradients"
                    )

                state = self.state[p]

                # --- Lazy state initialization ---
                if len(state) == 0:
                    self._init_state(p, state, group)
                state["step"] += 1
                step = state["step"]

                # --- Mixed precision: work in FP32 ---
                use_stochastic = stochastic_fp and p.dtype == torch.bfloat16
                p_work = p.detach()
                grad_work = p.grad.detach()
                if use_stochastic:
                    p_work = p_work.to(torch.float32)
                    grad_work = grad_work.to(torch.float32)

                # --- Warp: g -> P @ g with P = I + U @ V^T ---
                U = state.get("U")
                if U is not None:
                    V = state["V"]
                    g2d = _to_rows(grad_work)
                    Pg2d = g2d + U @ (V.t() @ g2d)
                    Pg = Pg2d.reshape(grad_work.shape)
                else:
                    Pg = grad_work

                # --- Adam moments on the warped gradient ---
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                exp_avg.lerp_(Pg, weight=1.0 - beta1)
                exp_avg_sq.lerp_(Pg.pow(2), weight=1.0 - beta2)

                bias_correction1 = 1.0 - beta1 ** step
                bias_correction2 = 1.0 - beta2 ** step
                denom = exp_avg_sq.sqrt().div_(math.sqrt(bias_correction2)).add_(eps)
                update = (exp_avg / bias_correction1) / denom

                # --- Cautious Masking ---
                if cautious_update:
                    mask = (Pg * update > 0).to(update.dtype)
                    mask.div_(mask.mean().clamp_min_(1e-3))
                    update.mul_(mask)

                # --- Weight Decay (AdamW-style) ---
                if weight_decay != 0:
                    p_work.mul_(1.0 - lr * weight_decay)
                p_work.add_(update, alpha=-lr)

                # --- Meta-update: online prediction of the next gradient ---
                if U is not None:
                    prev = state.get("prev_g")
                    if prev is not None:
                        R = (prev + U @ (V.t() @ prev)) - g2d
                        scale = 1.0 / prev.norm().square().clamp_min_(1e-12)
                        U.add_(R @ (prev.t() @ V), alpha=-meta_lr * scale)
                        V.add_(prev @ (R.t() @ U), alpha=-meta_lr * scale)
                        if meta_wd > 0:
                            U.mul_(1.0 - meta_lr * meta_wd)
                            V.mul_(1.0 - meta_lr * meta_wd)
                    state["prev_g"] = g2d.clone()

                # --- State Sync ---
                if use_stochastic:
                    copy_stochastic_(p, p_work)
                else:
                    p.copy_(p_work)

        return loss

    @torch.no_grad()
    def _step_meta(self, loss1: torch.Tensor, closure: Callable):
        # --- Snapshot: p.grad may be clobbered by the trial closure ---
        work = []  # (param, x0, g, group)
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError(
                        "WarpAdam does not support sparse gradients"
                    )
                work.append((p, p.detach().clone(), p.grad.detach().clone(), group))
        if not work:
            return loss1

        if not torch.isfinite(loss1).item():
            for p, _, _, _ in work:
                if p.grad is not None:
                    p.grad.zero_()
            return loss1

        # --- Lazy state initialization ---
        for p, _, _, group in work:
            state = self.state[p]
            if len(state) == 0:
                self._init_state(p, state, group)
            state["step"] += 1

        # --- Pass 1: warped Adam updates (graph only where P is trainable) ---
        commits = {}  # p -> final position (detached)
        meta_deltas = []  # (p, w_trial with graph, state, group)
        for p, x0, g, group in work:
            state = self.state[p]
            step = state["step"]
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            cautious_update = group["cautious_update"]
            use_stochastic = group["stochastic_fp"] and p.dtype == torch.bfloat16

            x0_work = x0.to(torch.float32) if use_stochastic else x0
            grad_work = g.to(torch.float32) if use_stochastic else g

            U = state.get("U")
            trainable = U is not None and group["mode"] == "meta"
            with torch.enable_grad():
                if U is not None:
                    V = state["V"]
                    g2d = _to_rows(grad_work)
                    Pg2d = g2d + U @ (V.t() @ g2d)
                    Pg = Pg2d.reshape(grad_work.shape)
                else:
                    Pg = grad_work

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                m_new = (1.0 - beta1) * Pg + beta1 * exp_avg.detach()
                v_new = (1.0 - beta2) * Pg.pow(2) + beta2 * exp_avg_sq.detach()
                bias_correction1 = 1.0 - beta1 ** step
                bias_correction2 = 1.0 - beta2 ** step
                denom = (v_new / bias_correction2).sqrt() + eps
                update = (m_new / bias_correction1) / denom
                if cautious_update:
                    mask = (Pg * update > 0).to(update.dtype)
                    mask.div_(mask.mean().clamp_min_(1e-3))
                    update = update * mask
                delta = -lr * update
                w_trial = x0_work + delta

            state["exp_avg"].copy_(m_new.detach())
            state["exp_avg_sq"].copy_(v_new.detach())

            commits[p] = x0_work.mul(1.0 - lr * weight_decay) + delta.detach()
            if trainable:
                meta_deltas.append((p, w_trial, state, group))

        # --- Trial: evaluate the loss at the joint warped point ---
        loss2 = None
        if meta_deltas:
            for p, w_trial, _, _ in meta_deltas:
                p.data.copy_(w_trial.detach())
            for p, _, _, _ in work:
                if p.grad is not None:
                    p.grad.zero_()
            with torch.enable_grad():
                loss2 = closure()
            if loss2 is not None and torch.isfinite(loss2).item():
                for p, w_trial, state, group in meta_deltas:
                    g_trial = p.grad
                    if g_trial is None or not torch.isfinite(g_trial).all():
                        continue
                    U = state["U"]
                    V = state["V"]
                    U.grad = None
                    V.grad = None
                    w_trial.backward(
                        gradient=g_trial.detach().to(w_trial.dtype)
                    )
                    meta_lr = group["meta_lr"]
                    meta_wd = group["meta_wd"]
                    if U.grad is not None:
                        U.data.add_(U.grad, alpha=-meta_lr)
                    if V.grad is not None:
                        V.data.add_(V.grad, alpha=-meta_lr)
                    if meta_wd > 0:
                        U.data.mul_(1.0 - meta_lr * meta_wd)
                        V.data.mul_(1.0 - meta_lr * meta_wd)

        # --- Commit the pass-1 updates ---
        for p, _, _, _ in work:
            p.data.copy_(commits[p])
            if p.grad is not None:
                p.grad.zero_()

        return loss2 if loss2 is not None else loss1
