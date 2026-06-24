import math
from typing import Iterable, Optional, Tuple, Union

import torch
import torch.nn as nn

from gefen.partitioning import find_period_by_block_variance
import gefen.quantization as quantization_module

FIND_PERIOD_BACKEND = None
FUSE_AUTOMATIC_VMEAN_UPDATE = True
FUSE_GEFEN_AUTOMATIC_STEP = True
FUSE_HISTOGRAM_FOR_EXACT = True


def automatic_partition_view(flat_tensor: torch.Tensor, period: int) -> torch.Tensor:
    if flat_tensor.dim() != 1:
        raise ValueError(
            "Automatic partition view expects a 1D tensor, got dim={}".format(
                flat_tensor.dim()
            )
        )
    if period <= 0:
        raise ValueError(
            "Automatic partition period must be positive, got {}".format(period)
        )
    if flat_tensor.numel() % period != 0:
        raise ValueError(
            "Automatic partition expects numel {} to be divisible by period {}".format(
                flat_tensor.numel(),
                period,
            )
        )
    return flat_tensor.view(-1, period)


def automatic_partition_reduce(
    flat_tensor: torch.Tensor, period: int, reduce_op: str
) -> torch.Tensor:
    flat_view = automatic_partition_view(flat_tensor, period)
    if reduce_op == "mean":
        return flat_view.mean(dim=1, keepdim=True)
    if reduce_op == "max":
        return flat_view.max(dim=1, keepdim=True).values
    raise ValueError("Unexpected reduce_op: {}".format(reduce_op))


def automatic_vmean_update(
    vmean: torch.Tensor,
    grad_view: torch.Tensor,
    beta2: float,
    *,
    use_fused: bool,
) -> None:
    if use_fused:
        if grad_view.device.type != "cuda":
            raise ValueError(
                "FUSE_AUTOMATIC_VMEAN_UPDATE requires CUDA grad_view, got device {}".format(
                    grad_view.device
                )
            )
        if vmean.device.type != "cuda":
            raise ValueError(
                "FUSE_AUTOMATIC_VMEAN_UPDATE requires CUDA vmean, got device {}".format(
                    vmean.device
                )
            )
        from gefen.kernels.automatic_vmean import automatic_vmean_update_cuda

        automatic_vmean_update_cuda(vmean, grad_view, beta2)
        return

    block_mean_grad_sq = torch.mean(grad_view * grad_view, dim=1, keepdim=True)
    vmean.mul_(beta2).add_(block_mean_grad_sq, alpha=1 - beta2)


def gefen_nearest_codebook_indices(
    codebook: torch.Tensor, normalized_vals: torch.Tensor
) -> torch.Tensor:

    if codebook.device != normalized_vals.device:
        raise ValueError(
            "Gefen codebook device {} does not match normalized_vals device {}.".format(
                codebook.device,
                normalized_vals.device,
            )
        )
    k = codebook.numel()
    flat = normalized_vals.reshape(-1).float()
    idx = torch.searchsorted(codebook, flat)
    left = (idx - 1).clamp(0, k - 1)
    right = idx.clamp(0, k - 1)
    assignments = torch.where(
        (flat - codebook[left]).abs() <= (flat - codebook[right]).abs(),
        left,
        right,
    )
    return assignments.to(torch.uint8).view(normalized_vals.shape)


def gefen_dequantize_unpacked_indices(
    codebook: torch.Tensor,
    stored_indices: torch.Tensor,
    like_tensor: torch.Tensor,
) -> torch.Tensor:
    if codebook.device != stored_indices.device:
        raise ValueError(
            "Gefen codebook device {} does not match stored index device {}.".format(
                codebook.device,
                stored_indices.device,
            )
        )

    coeff = codebook[stored_indices.long()].to(
        device=like_tensor.device, dtype=like_tensor.dtype
    )
    if hasattr(like_tensor, "device_mesh") and hasattr(like_tensor, "placements"):
        coeff_cls = type(like_tensor)
        if hasattr(coeff_cls, "from_local"):
            return coeff_cls.from_local(
                coeff, like_tensor.device_mesh, like_tensor.placements
            )
    return coeff


def gefen_set_unpacked_indices(
    stored_indices: torch.Tensor, indices: torch.Tensor
) -> None:
    stored_indices.copy_(indices.to(torch.uint8))


def gefen_automatic_fused_update(
    *,
    p: torch.Tensor,
    grad_view: torch.Tensor,
    m_codebook: torch.Tensor,
    m_magnitude: torch.Tensor,
    stepsize: torch.Tensor,
    codebook: torch.Tensor,
    packed_indices: bool,
    beta1: float,
    lr: float,
) -> None:
    if grad_view.device.type != "cuda":
        raise ValueError("Fused automatic Gefen update requires CUDA tensors.")
    if codebook.device != grad_view.device:
        raise ValueError(
            "Gefen codebook device {} does not match grad_view device {}.".format(
                codebook.device,
                grad_view.device,
            )
        )

    from gefen.kernels.automatic_gefen_fused import automatic_gefen_fused_update_cuda

    automatic_gefen_fused_update_cuda(
        p,
        grad_view,
        m_codebook,
        m_magnitude,
        stepsize,
        codebook,
        packed_indices,
        beta1,
        lr,
    )


def learn_gefen_exact_codebook_from_grad_periods(
    *,
    grad_periods,
    codebook_device: torch.device,
    num_codebooks: int,
    force_endpoints: bool,
    verbose: bool,
    compute_mse_logging: bool,
    use_fused_histogram: bool = FUSE_HISTOGRAM_FOR_EXACT,
) -> Optional[torch.Tensor]:
    if hasattr(quantization_module, "SAVE_CODEBOOK_PREFIX"):
        save_codebook_prefix = quantization_module.SAVE_CODEBOOK_PREFIX
        if save_codebook_prefix is not None:
            if not hasattr(quantization_module, "LIST_STEPS_SAVE_HIST_GRAD"):
                raise ValueError(
                    "Gefen exact codebook learning does not support SAVE_CODEBOOK_PREFIX exports. "
                    "Disable exports or define LIST_STEPS_SAVE_HIST_GRAD for the explicit histogram export path."
                )
            elif quantization_module.LIST_STEPS_SAVE_HIST_GRAD is None:
                raise ValueError(
                    "Gefen exact codebook learning does not support SAVE_CODEBOOK_PREFIX exports. "
                    "Disable exports or use a non-exact research path outside the publishable Gefen optimizer."
                )

    histogram_bins = num_codebooks * 16
    bin_width = 2.0 / float(histogram_bins)
    bin_counts_cpu = torch.zeros(histogram_bins, dtype=torch.float32, device="cpu")
    total_numel = 0

    prev_mode = torch.get_deterministic_debug_mode()
    torch.set_deterministic_debug_mode(0)
    try:
        for _, flat, period, _ in grad_periods:
            flat_float = flat.float()
            if use_fused_histogram and flat_float.device.type == "cuda":
                from gefen.kernels.exact_histogram_fused import (
                    gefen_exact_histogram_cuda,
                )

                local_counts_cuda = torch.zeros(
                    histogram_bins, dtype=torch.int64, device=flat_float.device
                )
                gefen_exact_histogram_cuda(flat_float, period, local_counts_cuda)
                bin_counts_cpu.add_(local_counts_cuda.cpu().to(torch.float32))
                total_numel += flat_float.numel()
            else:
                blocks = automatic_partition_view(flat_float, period)
                absmax = blocks.abs().amax(dim=1, keepdim=True)
                normalized = torch.where(
                    absmax > 0, blocks / absmax, torch.zeros_like(blocks)
                )
                normalized_flat = normalized.reshape(-1)
                bin_indices = torch.floor((normalized_flat + 1.0) / bin_width).to(
                    torch.long
                )
                bin_indices = bin_indices.clamp(0, histogram_bins - 1)
                local_counts = torch.bincount(
                    bin_indices, minlength=histogram_bins
                ).float()
                bin_counts_cpu.add_(local_counts.cpu())
                total_numel += normalized_flat.numel()
    finally:
        torch.set_deterministic_debug_mode(prev_mode)

    if total_numel == 0:
        return None

    bin_edges_cpu = torch.linspace(
        -1.0, 1.0, steps=histogram_bins + 1, dtype=torch.float32
    )
    bin_centers_cpu = 0.5 * (bin_edges_cpu[:-1] + bin_edges_cpu[1:])
    active_bins_cpu = bin_counts_cpu > 0
    histogram_stats = {
        "bin_edges": bin_edges_cpu,
        "bin_centers": bin_centers_cpu,
        "bin_counts": bin_counts_cpu,
        "active_bins": active_bins_cpu,
        "active_centers": bin_centers_cpu[active_bins_cpu],
        "active_counts": bin_counts_cpu[active_bins_cpu],
        "active_sums": bin_centers_cpu[active_bins_cpu]
        * bin_counts_cpu[active_bins_cpu],
    }

    if verbose:
        variant = "ForceOne (-1 and 1 fixed)" if force_endpoints else "standard"
        histogram_backend = (
            "fused cuda kernel" if use_fused_histogram else "reference pytorch"
        )

    codebook = quantization_module.exact_dp(
        histogram_stats=histogram_stats,
        num_codebooks=num_codebooks,
        force_endpoints=force_endpoints,
    ).to(codebook_device)

    if verbose and compute_mse_logging:
        codebook_cpu = codebook.cpu()
        active_centers = histogram_stats["active_centers"]
        idx = torch.searchsorted(codebook_cpu, active_centers)
        left_idx = (idx - 1).clamp(0, codebook_cpu.numel() - 1)
        right_idx = idx.clamp(0, codebook_cpu.numel() - 1)
        assignments = torch.where(
            (active_centers - codebook_cpu[left_idx]).abs()
            <= (active_centers - codebook_cpu[right_idx]).abs(),
            left_idx,
            right_idx,
        )
        errors = active_centers - codebook_cpu[assignments]
        mse_normalized = (
            errors.square() * histogram_stats["active_counts"]
        ).sum().item() / total_numel
        print("  exact DP: MSE_normalized={:.6e}".format(mse_normalized))

    return codebook


def _resolve_find_period_backend(grad: torch.Tensor) -> str:
    global FIND_PERIOD_BACKEND
    if FIND_PERIOD_BACKEND is not None:
        return FIND_PERIOD_BACKEND

    grad_work = grad.to_local() if hasattr(grad, "to_local") else grad
    if grad_work.device.type == "cuda":
        FIND_PERIOD_BACKEND = "cuda_kernel"
    else:
        FIND_PERIOD_BACKEND = "cpu"
    return FIND_PERIOD_BACKEND


class Gefen(torch.optim.Optimizer):

    def __init__(
        self,
        params: Iterable[Union[nn.Parameter, Tuple[str, nn.Parameter]]],
        lr: Union[float, torch.Tensor] = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        *,
        fused: bool = True,
        verbose: bool = False,
    ):
        if fused and not torch.cuda.is_available():
            print(
                "Gefen Optimizer:  got fused=True, but CUDA is not available. Changing fused to False."
            )
            fused = False
        print("Initializing Gefen optimizer (fused={}).".format(fused))

        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        def validate_group_options(
            group_lr, group_betas, group_eps, group_weight_decay
        ):
            if not 0.0 <= group_lr:
                raise ValueError("Invalid learning rate: {}".format(group_lr))
            elif not 0.0 <= group_betas[0] < 1.0:
                raise ValueError(
                    "Invalid beta parameter at index 0: {}".format(group_betas[0])
                )
            elif not 0.0 <= group_betas[1] < 1.0:
                raise ValueError(
                    "Invalid beta parameter at index 1: {}".format(group_betas[1])
                )
            elif not 0.0 <= group_weight_decay:
                raise ValueError(
                    "Invalid weight_decay value: {}".format(group_weight_decay)
                )
            elif not 0.0 <= group_eps:
                raise ValueError("Invalid epsilon value: {}".format(group_eps))

        def iter_params_with_names(group_params):
            if isinstance(group_params, torch.Tensor):
                yield None, group_params
                return
            for item_index, param_or_named_param in enumerate(group_params):
                if isinstance(param_or_named_param, tuple):
                    param_name, param = param_or_named_param
                else:
                    param_name = None
                    param = param_or_named_param
                if not isinstance(param, torch.Tensor):
                    raise TypeError(
                        "Gefen can only optimize Tensors, but one of the params is {}".format(
                            type(param).__name__,
                        )
                    )
                yield param_name, param

        def append_param_group(
            param_name, param, group_lr, group_betas, group_eps, group_weight_decay
        ):
            param_name = str(param_name).lower()
            if not param.requires_grad:
                return
            optim_groups.append(
                {
                    "name": param_name,
                    "params": param,
                    "lr": group_lr,
                    "beta1": group_betas[0],
                    "beta2": group_betas[1],
                    "eps": group_eps,
                    "weight_decay": group_weight_decay,
                }
            )

        self.fused = fused
        self.verbose = verbose

        self._printed_optimizer_memory = False
        self._gefen_codebook = None
        self._gefen_global_step = 0
        self._printed_gefen_materialization = False

        optim_groups = []
        for group_index, param_or_group in enumerate(params):
            if isinstance(param_or_group, dict):
                if "params" not in param_or_group:
                    raise ValueError(
                        "Gefen parameter group {} is missing the 'params' key.".format(
                            group_index
                        )
                    )
                group_lr = param_or_group.get("lr", lr)
                group_betas = param_or_group.get("betas", betas)
                group_eps = param_or_group.get("eps", eps)
                group_weight_decay = param_or_group.get("weight_decay", weight_decay)
                validate_group_options(
                    group_lr, group_betas, group_eps, group_weight_decay
                )
                for param_index, (param_name, param) in enumerate(
                    iter_params_with_names(param_or_group["params"])
                ):
                    if param_name is None:
                        param_name = "group_{}_param_{}".format(
                            group_index, param_index
                        )
                    append_param_group(
                        param_name,
                        param,
                        group_lr,
                        group_betas,
                        group_eps,
                        group_weight_decay,
                    )
            else:
                for param_name, param in iter_params_with_names((param_or_group,)):
                    if param_name is None:
                        param_name = "param_{}".format(group_index)
                    append_param_group(param_name, param, lr, betas, eps, weight_decay)

        defaults = dict(lr=lr, beta1=betas[0], beta2=betas[1], eps=eps)
        super().__init__(optim_groups, defaults)

    def _use_fused_automatic_vmean(self) -> bool:
        return self.fused and FUSE_AUTOMATIC_VMEAN_UPDATE

    def _use_fused_gefen_automatic_step(self) -> bool:
        return self.fused and FUSE_GEFEN_AUTOMATIC_STEP

    def _automatic_vmean_update(
        self,
        vmean: torch.Tensor,
        grad_view: torch.Tensor,
        beta2: float,
    ) -> None:
        automatic_vmean_update(
            vmean, grad_view, beta2, use_fused=self._use_fused_automatic_vmean()
        )

    def _gefen_codebook_device(self) -> torch.device:
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    return p.grad.device
                return p.device
        raise ValueError(
            "Expected at least one parameter when choosing the Gefen codebook device."
        )

    def _gefen_nearest_indices(self, normalized_vals: torch.Tensor) -> torch.Tensor:

        codebook = self._gefen_codebook
        if codebook is None:
            raise ValueError(
                "Expected Gefen codebook to be initialized before nearest-index lookup."
            )
        return gefen_nearest_codebook_indices(codebook, normalized_vals)

    def _gefen_dequantize_m_coefficients(
        self, state, like_tensor: torch.Tensor
    ) -> torch.Tensor:

        stored = state["m_codebook"]
        codebook = self._gefen_codebook
        if codebook is None:
            raise ValueError(
                "Expected Gefen codebook to be initialized before dequantizing m coefficients."
            )
        return gefen_dequantize_unpacked_indices(codebook, stored, like_tensor)

    def _gefen_set_indices(self, state, indices: torch.Tensor) -> None:
        gefen_set_unpacked_indices(state["m_codebook"], indices)

    def _print_gefen_codebook(self, label: str, codebook: torch.Tensor) -> None:
        if self.verbose:
            print("{} ({} entries):".format(label, codebook.numel()))

    def _automatic_view(self, flat_tensor: torch.Tensor, period: int) -> torch.Tensor:
        return automatic_partition_view(flat_tensor, period)

    def _automatic_reduce(
        self, flat_tensor: torch.Tensor, period: int, reduce_op: str
    ) -> torch.Tensor:
        return automatic_partition_reduce(flat_tensor, period, reduce_op)

    def _predict_period_from_grad_sq(
        self, param_name: str, param: torch.Tensor, grad: torch.Tensor
    ) -> int:
        backend = _resolve_find_period_backend(grad)
        if backend == "cuda_kernel":
            grad_work = grad.to_local() if hasattr(grad, "to_local") else grad
            grad_flat = grad_work.detach().reshape(-1)
            try:
                return find_period_by_block_variance(
                    grad_flat,
                    print_results=False,
                    parameter_name=param_name,
                    parameter_shape=tuple(param.shape),
                    backend=backend,
                    input_is_squared=False,
                )
            except RuntimeError as exc:
                msg = str(exc)
                needs_materialization = (
                    "data is not allocated yet" in msg
                    or "invalid python storage" in msg
                )
                if not needs_materialization:
                    raise
                if not self._printed_gefen_materialization:
                    pass

                grad_work.add_(0)
                grad_flat = grad_work.detach().reshape(-1)
                return find_period_by_block_variance(
                    grad_flat,
                    print_results=False,
                    parameter_name=param_name,
                    parameter_shape=tuple(param.shape),
                    backend=backend,
                    input_is_squared=False,
                )

        if backend == "cpu":
            period_input = grad.detach().float().square().reshape(-1).cpu().numpy()
        elif backend == "gpu":
            if grad.device.type != "cuda":
                raise ValueError("FIND_PERIOD_BACKEND='gpu' requires a CUDA tensor")
            period_input = grad.detach().float().square().reshape(-1)
        else:
            raise ValueError("Unexpected FIND_PERIOD_BACKEND: {}".format(backend))
        return find_period_by_block_variance(
            period_input,
            print_results=False,
            parameter_name=param_name,
            parameter_shape=tuple(param.shape),
            backend=backend,
            input_is_squared=True,
        )

    def _print_v_period(
        self, param_name: str, param: torch.Tensor, shared_v: torch.Tensor
    ) -> None:
        if not self.verbose:
            return
        shared_v_numel = shared_v.numel()
        param_numel = param.numel()
        grad = param.grad
        if shared_v_numel == 0:
            raise ValueError(
                "shared_v must have at least one element for {}".format(param_name)
            )
        if grad is None:
            raise ValueError("Expected .grad to exist for {}".format(param_name))
        if grad.shape != param.shape:
            raise ValueError(
                ".grad shape {} does not match parameter shape {} for {}".format(
                    tuple(grad.shape),
                    tuple(param.shape),
                    param_name,
                )
            )
        if param_numel % shared_v_numel != 0:
            raise ValueError(
                "Cannot compute an integer shared-v period for {}: param_numel={} shared_v_numel={}".format(
                    param_name,
                    param_numel,
                    shared_v_numel,
                )
            )

    def _iter_gefen_grad_periods(self, reuse_existing_periods: bool = False):

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if hasattr(grad, "to_local"):
                    grad = grad.to_local()
                if hasattr(grad, "wait"):
                    grad = grad.wait()
                grad = grad.detach()
                flat = grad.reshape(-1)
                if flat.numel() == 0:
                    continue

                if reuse_existing_periods:
                    state = self.state[p]
                    if "automatic_period" not in state:
                        raise ValueError(
                            "Expected automatic_period to exist for {} before refreshing Gefen codebook at optimizer step {}".format(
                                group["name"],
                                self._gefen_global_step,
                            )
                        )
                    period = state["automatic_period"]
                elif flat.numel() == 1:
                    period = 1
                else:
                    period = self._predict_period_from_grad_sq(group["name"], p, grad)

                self.state[p]["automatic_period"] = period

                if flat.numel() % period != 0:
                    raise ValueError(
                        "Automatic partition period {} does not divide parameter {} with numel {} while learning Gefen codebook".format(
                            period,
                            group["name"],
                            flat.numel(),
                        )
                    )

                yield group["name"], flat, period, grad

    def _learn_gefen_exact_codebook(
        self,
        reuse_existing_periods: bool = False,
        compute_mse_logging: bool = True,
    ) -> Optional[torch.Tensor]:
        codebook = learn_gefen_exact_codebook_from_grad_periods(
            grad_periods=self._iter_gefen_grad_periods(
                reuse_existing_periods=reuse_existing_periods
            ),
            codebook_device=self._gefen_codebook_device(),
            num_codebooks=256,
            force_endpoints=True,
            verbose=self.verbose,
            compute_mse_logging=compute_mse_logging,
            use_fused_histogram=FUSE_HISTOGRAM_FOR_EXACT,
        )
        if codebook is None:
            return None

        return codebook

    def _ensure_gefen_codebook(self, reuse_existing_periods: bool = False) -> None:
        compute_mse_logging = False
        if "COMPUTE_GEFEN_EXACT_MSE_LOGGING" in globals():
            compute_mse_logging = COMPUTE_GEFEN_EXACT_MSE_LOGGING
        codebook = self._learn_gefen_exact_codebook(
            reuse_existing_periods=reuse_existing_periods,
            compute_mse_logging=compute_mse_logging,
        )
        if codebook is not None:
            self._gefen_codebook = codebook

    def _maybe_refresh_gefen_codebook(self) -> None:
        if self._gefen_codebook is None:
            self._ensure_gefen_codebook()

    def _maybe_save_gefen_grad_histogram(self) -> None:
        if not hasattr(quantization_module, "LIST_STEPS_SAVE_HIST_GRAD"):
            return
        requested_steps = quantization_module.LIST_STEPS_SAVE_HIST_GRAD
        if requested_steps is None:
            return
        if self._gefen_global_step not in requested_steps:
            return
        if not hasattr(quantization_module, "SAVE_CODEBOOK_PREFIX"):
            raise ValueError(
                "LIST_STEPS_SAVE_HIST_GRAD requires SAVE_CODEBOOK_PREFIX to be defined and set."
            )
        save_codebook_prefix = quantization_module.SAVE_CODEBOOK_PREFIX
        if save_codebook_prefix is None:
            raise ValueError(
                "LIST_STEPS_SAVE_HIST_GRAD requires SAVE_CODEBOOK_PREFIX to be set."
            )
        if self._gefen_codebook is None:
            raise ValueError(
                "Expected Gefen codebook to be initialized before saving gradient histogram."
            )

        histogram_bins = self._gefen_codebook.numel() * 16
        bin_width = 2.0 / float(histogram_bins)
        bin_counts_cpu = torch.zeros(histogram_bins, dtype=torch.float32, device="cpu")
        total_numel = 0

        for _, flat, period, _ in self._iter_gefen_grad_periods(
            reuse_existing_periods=True
        ):
            flat_float = flat.float()
            blocks = self._automatic_view(flat_float, period)
            absmax = blocks.abs().amax(dim=1, keepdim=True)
            normalized = torch.where(
                absmax > 0, blocks / absmax, torch.zeros_like(blocks)
            )
            normalized_flat = normalized.reshape(-1)
            bin_indices = torch.floor((normalized_flat + 1.0) / bin_width).to(
                torch.long
            )
            bin_indices = bin_indices.clamp(0, histogram_bins - 1)
            local_counts = torch.bincount(bin_indices, minlength=histogram_bins).float()
            bin_counts_cpu.add_(local_counts.cpu())
            total_numel += normalized_flat.numel()

        if total_numel == 0:
            raise ValueError(
                "Cannot save Gefen gradient histogram because no gradients were available."
            )

        quantization_module.save_distribution_histogram_pt_from_counts(
            hist=bin_counts_cpu,
            prefix=save_codebook_prefix,
            suffix="grad_hist_step_{}".format(self._gefen_global_step),
            num_samples=total_numel,
        )

    def _init_gefen_state(self, state, grad_view: torch.Tensor) -> None:
        state["m_codebook_shape"] = tuple(grad_view.shape)
        state["m_codebook"] = torch.zeros_like(
            grad_view, dtype=torch.uint8, memory_format=torch.preserve_format
        )

        state["m_magnitude"] = torch.zeros_like(
            grad_view[:, 0:1],
            dtype=torch.float32,
            memory_format=torch.preserve_format,
        )

    def _automatic_gefen_fused_update(
        self, p, state, grad_view, beta1, stepsize, lr
    ) -> None:

        codebook = self._gefen_codebook
        if codebook is None:
            raise ValueError(
                "Expected Gefen codebook to be initialized before fused update."
            )
        gefen_automatic_fused_update(
            p=p,
            grad_view=grad_view,
            m_codebook=state["m_codebook"],
            m_magnitude=state["m_magnitude"],
            stepsize=stepsize,
            codebook=codebook,
            packed_indices=False,
            beta1=beta1,
            lr=lr,
        )

    def _automatic_momentum_update(
        self, state, grad_view: torch.Tensor, beta1: float
    ) -> torch.Tensor:
        if grad_view.dim() != 2:
            raise ValueError(
                "Automatic shared momentum expects a 2D tensor, got dim={}".format(
                    grad_view.dim()
                )
            )

        period = state["automatic_period"]
        current_m = (
            self._gefen_dequantize_m_coefficients(state, grad_view)
            * state["m_magnitude"]
        )
        updated_m = current_m.lerp(grad_view, 1 - beta1)

        state["m_magnitude"].copy_(
            self._automatic_reduce(updated_m.abs().reshape(-1), period, reduce_op="max")
        )

        nonzero_mask = state["m_magnitude"] > 0
        updated_m.div_(state["m_magnitude"])
        updated_m.masked_fill_(~nonzero_mask, 0.0)

        indices = self._gefen_nearest_indices(updated_m)
        self._gefen_set_indices(state, indices)

        codebook = self._gefen_codebook
        if codebook is None:
            raise ValueError(
                "Expected Gefen codebook to be initialized before reconstructing quantized m."
            )
        if codebook.device != indices.device:
            raise ValueError(
                "Gefen codebook device {} does not match index device {}.".format(
                    codebook.device,
                    indices.device,
                )
            )

        quantized_m = codebook[indices.long()].to(dtype=grad_view.dtype)
        return quantized_m * state["m_magnitude"]

    def _step_automatic(
        self, group, param_name: str, p: torch.Tensor, grad: torch.Tensor
    ) -> None:
        if not self.fused:
            if hasattr(grad, "to_local"):
                grad = grad.to_local()
            if hasattr(grad, "wait"):
                grad = grad.wait()

        state = self.state[p]
        beta1 = group["beta1"]
        beta2 = group["beta2"]
        lr = group["lr"]
        eps = group["eps"]

        flat_grad = grad.reshape(-1)

        if "step" not in state:
            if "automatic_period" in state:
                automatic_period = state["automatic_period"]
            elif p.numel() == 1:
                automatic_period = 1
            elif p.numel() > 1:
                automatic_period = self._predict_period_from_grad_sq(
                    param_name, p, grad
                )
            else:
                raise ValueError(
                    "Automatic partition received an empty parameter {}".format(
                        param_name
                    )
                )

            if p.numel() % automatic_period != 0:
                raise ValueError(
                    "Automatic partition period {} does not divide parameter {} with numel {}".format(
                        automatic_period,
                        param_name,
                        p.numel(),
                    )
                )

            state["automatic_period"] = automatic_period
            state["step"] = 0
            grad_view = self._automatic_view(flat_grad, automatic_period)
            self._init_gefen_state(state, grad_view)

            state["vmean"] = torch.zeros_like(
                grad_view[:, 0:1],
                dtype=torch.float32,
                memory_format=torch.preserve_format,
            )
            self._print_v_period(param_name, p, state["vmean"])

        automatic_period = state["automatic_period"]
        grad_view = self._automatic_view(flat_grad, automatic_period)

        self._automatic_vmean_update(state["vmean"], grad_view, beta2)

        state["step"] += 1

        if group["weight_decay"] > 0.0:
            p.mul_(1 - lr * group["weight_decay"])

        bias_correction_1 = 1 - beta1 ** state["step"]
        bias_correction_2 = 1 - beta2 ** state["step"]
        h = (state["vmean"].sqrt() / math.sqrt(bias_correction_2)).add_(eps)
        stepsize = (1 / bias_correction_1) / h

        if self._use_fused_gefen_automatic_step():
            self._automatic_gefen_fused_update(
                p,
                state,
                grad_view,
                beta1,
                stepsize,
                lr,
            )

            return

        shared_m = self._automatic_momentum_update(state, grad_view, beta1)
        update = (shared_m * stepsize).view(grad.size())
        update.mul_(lr)
        if not self.fused and hasattr(p, "to_local") and hasattr(p, "placements"):
            p.to_local().add_(-update)
        else:
            p.add_(-update)

    def _optimizer_state_memory_bytes(self) -> int:
        total = 0
        for state in self.state.values():
            for value in state.values():
                if torch.is_tensor(value):
                    total += value.numel() * value.element_size()
        return total

    def state_dict(self):
        state_dict = super().state_dict()
        state_dict["gefen_global_step"] = self._gefen_global_step
        return state_dict

    def load_state_dict(self, state_dict):
        gefen_global_step = state_dict.pop("gefen_global_step", 0)
        super().load_state_dict(state_dict)
        self._gefen_global_step = gefen_global_step

    @torch.no_grad()
    def step(self, closure=None):
        self._maybe_refresh_gefen_codebook()
        self._maybe_save_gefen_grad_histogram()

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            name = group["name"]
            for p in group["params"]:
                grad = p.grad
                if grad is None:
                    continue
                self._step_automatic(group, name, p, grad)

        self._gefen_global_step += 1
        return loss
