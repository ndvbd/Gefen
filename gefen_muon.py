import math
from typing import Iterable, Optional, Tuple, Union

import torch
import torch.nn as nn

from gefen.gefen import Gefen

EPS = 1e-7
DEFAULT_A = 3.4445
DEFAULT_B = -4.7750
DEFAULT_C = 2.0315
DEFAULT_NS_STEPS = 5


def _zeropower_via_newtonschulz(
    grad: torch.Tensor,
    ns_coefficients: Tuple[float, float, float],
    ns_steps: int,
    eps: float,
) -> torch.Tensor:

    if ns_steps >= 100:
        raise ValueError(
            "Number of steps must be less than 100 for computational efficiency"
        )
    if len(grad.shape) != 2:
        raise ValueError("Input tensor gradient must be a 2D matrix")
    if len(ns_coefficients) != 3:
        raise ValueError("Coefficients must be a tuple of exactly 3 values")

    a, b, c = ns_coefficients
    ortho_grad = grad.bfloat16()
    if grad.size(0) > grad.size(1):
        ortho_grad = ortho_grad.T

    ortho_grad.div_(ortho_grad.norm().clamp(min=eps))
    for _ in range(ns_steps):
        gram_matrix = ortho_grad @ ortho_grad.T
        gram_update = torch.addmm(
            gram_matrix,
            gram_matrix,
            gram_matrix,
            beta=b,
            alpha=c,
        )
        ortho_grad = torch.addmm(ortho_grad, gram_update, ortho_grad, beta=a)

    if grad.size(0) > grad.size(1):
        ortho_grad = ortho_grad.T
    return ortho_grad


def _adjust_lr(
    lr: float, adjust_lr_fn: Optional[str], param_shape: torch.Size
) -> float:

    if isinstance(lr, torch.Tensor):
        lr = lr.item()
    rows, cols = param_shape[:2]
    if adjust_lr_fn is None or adjust_lr_fn == "original":
        adjusted_ratio = math.sqrt(max(1, rows / cols))
    elif adjust_lr_fn == "match_rms_adamw":
        adjusted_ratio = 0.2 * math.sqrt(max(rows, cols))
    else:
        adjusted_ratio = 1.0
    return lr * adjusted_ratio


class GefenMuon(Gefen):

    def __init__(
        self,
        params: Iterable[Union[nn.Parameter, Tuple[str, nn.Parameter]]],
        lr: Union[float, torch.Tensor] = 1e-3,
        weight_decay: float = 0.1,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_coefficients: Tuple[float, float, float] = (DEFAULT_A, DEFAULT_B, DEFAULT_C),
        eps: float = EPS,
        ns_steps: int = DEFAULT_NS_STEPS,
        adjust_lr_fn: Optional[str] = None,
        *,
        fused: bool = True,
        verbose: bool = False,
    ) -> None:
        if isinstance(lr, torch.Tensor) and lr.numel() != 1:
            raise ValueError("Tensor lr must be 1-element")
        if not 0.0 <= lr:
            raise ValueError("Learning rate should be >= 0 but is: {}".format(lr))
        if not 0.0 <= momentum:
            raise ValueError("momentum should be >= 0 but is: {}".format(momentum))
        if not 0.0 <= weight_decay:
            raise ValueError(
                "weight decay should be >= 0 but is: {}".format(weight_decay)
            )
        if adjust_lr_fn is not None and adjust_lr_fn not in [
            "original",
            "match_rms_adamw",
        ]:
            raise ValueError(
                "Adjust learning rate function {} is not supported".format(adjust_lr_fn)
            )

        super().__init__(
            params,
            lr=lr,
            betas=(momentum, 0.0),
            eps=eps,
            weight_decay=weight_decay,
            fused=fused,
            verbose=verbose,
        )

        for group in self.param_groups:
            group["momentum"] = momentum
            group["nesterov"] = nesterov
            group["ns_coefficients"] = ns_coefficients
            group["ns_steps"] = ns_steps
            group["adjust_lr_fn"] = adjust_lr_fn
            for p in group["params"]:
                if p.ndim != 2:
                    raise ValueError(
                        "GefenMuon only supports 2D parameters whereas we found a parameter with size: {}".format(
                            p.size(),
                        )
                    )

    def _init_gefen_muon_state(self, state, grad_view: torch.Tensor) -> None:
        self._init_gefen_state(state, grad_view)

    def _quantize_momentum_(self, state, momentum_view: torch.Tensor) -> None:
        period = state["automatic_period"]

        if period == 1:
            magnitude_period = self._period_one_divisor_from_state(
                state, momentum_view.numel()
            )
        elif period > 1:
            magnitude_period = period
        else:
            raise ValueError("automatic_period must be positive, got {}".format(period))
        magnitude_fp32 = self._automatic_reduce(
            momentum_view.float().abs().reshape(-1), magnitude_period, reduce_op="max"
        )
        state["m_magnitude"].copy_(magnitude_fp32)

        magnitude_for_m = self._m_magnitude_view_for_m(state, momentum_view)
        momentum_view.div_(
            magnitude_for_m.clamp(min=torch.finfo(momentum_view.dtype).tiny)
        )
        momentum_view.masked_fill_(magnitude_for_m <= 0, 0.0)

        indices = self._gefen_nearest_indices(momentum_view)
        self._gefen_set_indices(state, indices)

        codebook = self._gefen_codebook
        if codebook is None:
            raise ValueError(
                "Expected Gefen codebook to be initialized before reconstructing quantized momentum."
            )
        if codebook.device != indices.device:
            raise ValueError(
                "Gefen codebook device {} does not match index device {}.".format(
                    codebook.device,
                    indices.device,
                )
            )

        momentum_view.copy_(codebook[indices.long()].to(dtype=momentum_view.dtype))
        momentum_view.mul_(self._m_magnitude_view_for_m(state, momentum_view))

    def _fused_quantized_momentum_update(
        self,
        p: torch.Tensor,
        state,
        grad_view: torch.Tensor,
        momentum: float,
    ) -> torch.Tensor:
        codebook = self._gefen_codebook
        if codebook is None:
            raise ValueError(
                "Expected Gefen codebook to be initialized before fused GefenMuon update."
            )

        if state["automatic_period"] == 1:
            period_one_divisor = self._period_one_divisor_from_state(
                state, grad_view.numel()
            )
            fused_grad_view = grad_view.reshape(-1, period_one_divisor)
        elif state["automatic_period"] > 1:
            fused_grad_view = grad_view
        else:
            raise ValueError(
                "automatic_period must be positive, got {}".format(
                    state["automatic_period"]
                )
            )

        momentum_update = torch.empty_like(fused_grad_view)
        from gefen.kernels.automatic_gefen_fused import (
            automatic_gefen_muon_momentum_cuda,
        )

        automatic_gefen_muon_momentum_cuda(
            grad_view=fused_grad_view,
            m_sign=state["m_codebook"],
            m_magnitude=state["m_magnitude"],
            codebook=codebook,
            momentum_update=momentum_update,
            packed_indices=False,
            beta1=momentum,
        )
        return momentum_update.reshape_as(grad_view)

    def _step_automatic(
        self, group, param_name: str, p: torch.Tensor, grad: torch.Tensor
    ) -> None:
        local_p = self._resolve_local_tensor(p, "parameter {}".format(param_name))
        grad = self._resolve_local_tensor(grad, "gradient {}".format(param_name))

        if torch.is_complex(local_p):
            raise RuntimeError("GefenMuon does not support complex parameters")
        if grad.is_sparse:
            raise RuntimeError("GefenMuon does not support sparse gradients")
        if grad.ndim != 2:
            raise ValueError(
                "GefenMuon gradient must be a 2D matrix for {}".format(param_name)
            )

        state = self.state[p]
        lr = group["lr"]
        momentum = group["momentum"]

        flat_grad = grad.reshape(-1)
        local_numel = flat_grad.numel()
        if local_p.numel() != local_numel:
            raise ValueError(
                "Local parameter shard numel {} does not match local gradient numel {} for {}.".format(
                    local_p.numel(),
                    local_numel,
                    param_name,
                )
            )
        if local_numel == 0:
            return

        if "step" not in state:
            if "automatic_period" in state:
                automatic_period = state["automatic_period"]
            elif local_numel == 1:
                automatic_period = 1
            elif local_numel > 1:
                automatic_period = self._predict_period_from_grad_sq(
                    param_name, p, grad
                )
            else:
                raise ValueError(
                    "Automatic partition received an empty local shard for parameter {}".format(
                        param_name
                    )
                )

            if local_numel % automatic_period != 0:
                raise ValueError(
                    "Automatic partition period {} does not divide local shard of parameter {} with numel {}".format(
                        automatic_period,
                        param_name,
                        local_numel,
                    )
                )

            state["automatic_period"] = automatic_period
            state["step"] = 0
            self._init_gefen_muon_state(
                state, self._automatic_view(flat_grad, automatic_period)
            )

            self._print_period(param_name, local_p, automatic_period)

        automatic_period = state["automatic_period"]
        grad_view = self._automatic_view(flat_grad, automatic_period)

        if self._use_fused_gefen_automatic_step():
            if local_p.is_contiguous():
                fused_p = local_p
            elif hasattr(p, "to_local") and hasattr(p, "placements"):

                fused_p = local_p.contiguous()
            else:
                raise ValueError(
                    "GefenMuon fused update requires a contiguous parameter tensor for {}.".format(
                        param_name
                    )
                )
            momentum_update = self._fused_quantized_momentum_update(
                fused_p,
                state,
                grad_view,
                momentum,
            )
        else:
            momentum_update = self._gefen_dequantize_m_coefficients_notfused(
                state, grad_view
            )
            momentum_update.mul_(self._m_magnitude_view_for_m(state, grad_view))
            momentum_update.lerp_(grad_view, 1 - momentum)
            self._quantize_momentum_(state, momentum_update)

        state["step"] += 1

        if group["nesterov"]:
            momentum_update.mul_(momentum).add_(grad_view, alpha=1 - momentum)
            update = momentum_update.view_as(grad)
        else:
            update = momentum_update.view_as(grad)

        update = _zeropower_via_newtonschulz(
            update,
            group["ns_coefficients"],
            group["ns_steps"],
            group["eps"],
        )

        adjusted_lr = _adjust_lr(lr, group["adjust_lr_fn"], p.shape)
        if group["weight_decay"] > 0.0:
            local_p.mul_(1 - lr * group["weight_decay"])

        local_p.add_(update, alpha=-adjusted_lr)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self._maybe_refresh_gefen_codebook()
        self._maybe_save_gefen_grad_histogram()

        for group in self.param_groups:
            name = group["name"]
            for p in group["params"]:
                grad = p.grad
                if grad is None:
                    continue
                self._step_automatic(group, name, p, grad)

        self._gefen_global_step += 1
        return loss
