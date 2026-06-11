import math
from itertools import chain
from pathlib import Path
from typing import Iterable, Optional, Tuple, Union

import torch
import torch.nn as nn

from gefen.partitioning import divisors, find_period_by_block_variance
import gefen.quantization as quantization_module

FIND_PERIOD_BACKEND = None


CPBS = True

FUSE_AUTOMATIC_VMEAN_UPDATE = True
FUSE_GEFEN_AUTOMATIC_STEP = True
FUSE_HISTOGRAM_FOR_EXACT = True
F_FP32 = False
NETWORK_STRUCTURE_FILENAME = "gefen_network_structure.pt"
NETWORK_STRUCTURE_VERSION = 1


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


def gefen_state_dtype_from_grad(grad_view: torch.Tensor) -> torch.dtype:
    force_fp32 = globals().get("F_FP32", False)
    if not isinstance(force_fp32, bool):
        raise TypeError(
            "F_FP32 must be a bool, got {}".format(type(force_fp32).__name__)
        )
    if force_fp32:
        return torch.float32
    if grad_view.dtype == torch.float32:
        return torch.float32
    elif grad_view.dtype == torch.float16:
        return torch.float16
    elif grad_view.dtype == torch.bfloat16:
        return torch.bfloat16
    raise ValueError(
        "Gefen state dtype is undefined for gradient dtype {}".format(grad_view.dtype)
    )


def largest_divisor_at_most(value: int, limit: int) -> int:
    if value <= 0:
        raise ValueError("Expected value to be positive, got {}".format(value))
    elif limit <= 0:
        raise ValueError("Expected limit to be positive, got {}".format(limit))
    if value <= limit:
        return value
    value_divisors = divisors(value, max_divisor=limit)
    if len(value_divisors) == 0:
        raise ValueError("Could not find a positive divisor for value {}".format(value))
    return value_divisors[-1]


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

    grad_view_f32 = grad_view.float()
    block_mean_grad_sq = torch.mean(grad_view_f32 * grad_view_f32, dim=1, keepdim=True)
    updated_vmean = vmean.float().mul(beta2).add(block_mean_grad_sq, alpha=1 - beta2)
    vmean.copy_(updated_vmean)


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


def gefen_automatic_fused_update_from_vmean(
    *,
    p: torch.Tensor,
    grad_view: torch.Tensor,
    m_codebook: torch.Tensor,
    m_magnitude: torch.Tensor,
    vmean: torch.Tensor,
    codebook: torch.Tensor,
    packed_indices: bool,
    beta1: float,
    lr: float,
    bias_correction_1: float,
    bias_correction_2: float,
    eps: float,
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

    from gefen.kernels.automatic_gefen_fused import (
        automatic_gefen_fused_update_from_vmean_cuda,
    )

    automatic_gefen_fused_update_from_vmean_cuda(
        p,
        grad_view,
        m_codebook,
        m_magnitude,
        vmean,
        codebook,
        packed_indices,
        beta1,
        lr,
        bias_correction_1,
        bias_correction_2,
        eps,
    )


def gefen_automatic_period1_update(
    *,
    p: torch.Tensor,
    grad_flat: torch.Tensor,
    m_codebook: torch.Tensor,
    m_magnitude: torch.Tensor,
    vmean: torch.Tensor,
    codebook: torch.Tensor,
    period_one_divisor: int,
    beta1: float,
    beta2: float,
    lr: float,
    bias_correction_1: float,
    bias_correction_2: float,
    eps: float,
) -> None:
    if grad_flat.device.type != "cuda":
        raise ValueError("Fused period-1 Gefen update requires CUDA tensors.")
    if codebook.device != grad_flat.device:
        raise ValueError(
            "Gefen codebook device {} does not match grad_flat device {}.".format(
                codebook.device,
                grad_flat.device,
            )
        )

    from gefen.kernels.automatic_gefen_fused import automatic_gefen_period1_update_cuda

    automatic_gefen_period1_update_cuda(
        p,
        grad_flat,
        m_codebook,
        m_magnitude,
        vmean,
        codebook,
        period_one_divisor,
        beta1,
        beta2,
        lr,
        bias_correction_1,
        bias_correction_2,
        eps,
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
            if use_fused_histogram and flat.device.type == "cuda":
                from gefen.kernels.exact_histogram_fused import (
                    gefen_exact_histogram_cuda,
                )

                local_counts_cuda = torch.zeros(
                    histogram_bins, dtype=torch.int64, device=flat.device
                )
                gefen_exact_histogram_cuda(flat, period, local_counts_cuda)
                bin_counts_cpu.add_(local_counts_cuda.cpu().to(torch.float32))
                total_numel += flat.numel()
            else:

                flat_float = flat.float()
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
        save_structure: bool = False,
        load_structure: bool = False,
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
        if not isinstance(save_structure, bool):
            raise TypeError(
                "save_structure must be a bool, got {}".format(
                    type(save_structure).__name__
                )
            )
        if not isinstance(load_structure, bool):
            raise TypeError(
                "load_structure must be a bool, got {}".format(
                    type(load_structure).__name__
                )
            )
        self.save_structure = save_structure
        self.load_structure = load_structure
        self._network_structure_path = Path.cwd() / NETWORK_STRUCTURE_FILENAME
        self._loaded_network_structure = False
        self._saved_network_structure = False

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
        if self._cap_period_by_shape_enabled():
            self._record_constructor_parameter_shapes()

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

    def _gefen_dequantize_m_coefficients_notfused(
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

    def _period_one_divisor_for_numel(self, numel: int) -> int:
        return largest_divisor_at_most(numel, 256)

    def _period_one_divisor_from_state(self, state, total_numel: int) -> int:
        period_one_divisor = state.get("period_one_divisor")
        if period_one_divisor is None:
            period_one_divisor = self._period_one_divisor_for_numel(total_numel)
            state["period_one_divisor"] = period_one_divisor
        elif not isinstance(period_one_divisor, int):
            raise TypeError(
                "period_one_divisor must be an int, got {}".format(
                    type(period_one_divisor).__name__
                )
            )
        elif period_one_divisor <= 0:
            raise ValueError(
                "period_one_divisor must be positive, got {}".format(period_one_divisor)
            )
        elif total_numel % period_one_divisor != 0:
            raise ValueError(
                "period_one_divisor {} must divide flattened tensor numel {}".format(
                    period_one_divisor,
                    total_numel,
                )
            )
        m_magnitude = state.get("m_magnitude")
        if torch.is_tensor(m_magnitude):
            expected_rows = total_numel // period_one_divisor
            if m_magnitude.shape != (expected_rows, 1):
                raise ValueError(
                    "Expected m_magnitude shape {} for period_one_divisor {}, got {}".format(
                        (expected_rows, 1),
                        period_one_divisor,
                        tuple(m_magnitude.shape),
                    )
                )
        return period_one_divisor

    def _m_magnitude_view_for_m(self, state, like_tensor: torch.Tensor) -> torch.Tensor:
        automatic_period = state["automatic_period"]
        if automatic_period == 1:
            period_one_divisor = self._period_one_divisor_from_state(
                state, like_tensor.numel()
            )
            return state["m_magnitude"].repeat_interleave(period_one_divisor, dim=0)
        elif automatic_period > 1:
            return state["m_magnitude"]
        raise ValueError(
            "automatic_period must be positive, got {}".format(automatic_period)
        )

    def _print_gefen_codebook(self, label: str, codebook: torch.Tensor) -> None:
        if not self.verbose:
            return
        print("{} ({} entries):".format(label, codebook.numel()))

    def _automatic_view(self, flat_tensor: torch.Tensor, period: int) -> torch.Tensor:
        return automatic_partition_view(flat_tensor, period)

    def _automatic_reduce(
        self, flat_tensor: torch.Tensor, period: int, reduce_op: str
    ) -> torch.Tensor:
        return automatic_partition_reduce(flat_tensor, period, reduce_op)

    def _resolve_local_tensor(
        self, tensor: torch.Tensor, tensor_name: str
    ) -> torch.Tensor:
        if hasattr(tensor, "to_local"):
            tensor = tensor.to_local()
        if hasattr(tensor, "wait"):
            tensor = tensor.wait()
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(
                "Expected {} to resolve to a torch.Tensor, got {}.".format(
                    tensor_name,
                    type(tensor),
                )
            )
        return tensor

    def _cap_period_by_shape_enabled(self) -> bool:
        return globals().get("CPBS", False)

    def _record_constructor_parameter_shapes(self) -> None:

        for group in self.param_groups:
            for p in group["params"]:
                self.state[p]["parameter_shape"] = tuple(p.shape)

    def _parameter_shape_for_period_cap(self, param: torch.Tensor) -> Tuple[int, ...]:
        state = self.state.get(param, {})
        return tuple(state.get("parameter_shape", tuple(param.shape)))

    def _max_period_for_param(self, param: torch.Tensor) -> Optional[int]:
        if not self._cap_period_by_shape_enabled():
            return None
        parameter_shape = self._parameter_shape_for_period_cap(param)
        if len(parameter_shape) == 0:
            return 1
        return max(int(dim) for dim in parameter_shape)

    def _predict_period_from_grad_sq(
        self, param_name: str, param: torch.Tensor, grad: torch.Tensor
    ) -> int:
        backend = _resolve_find_period_backend(grad)
        if self._cap_period_by_shape_enabled():
            parameter_shape = self._parameter_shape_for_period_cap(param)
            max_period = self._max_period_for_param(param)
        else:
            parameter_shape = tuple(param.shape)
            max_period = None
        if backend == "cuda_kernel":
            grad_work = grad.to_local() if hasattr(grad, "to_local") else grad
            grad_flat = grad_work.detach().reshape(-1)
            try:
                return find_period_by_block_variance(
                    grad_flat,
                    print_results=False,
                    parameter_name=param_name,
                    parameter_shape=parameter_shape,
                    backend=backend,
                    input_is_squared=False,
                    max_period=max_period,
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
                    parameter_shape=parameter_shape,
                    backend=backend,
                    input_is_squared=False,
                    max_period=max_period,
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
            parameter_shape=parameter_shape,
            backend=backend,
            input_is_squared=True,
            max_period=max_period,
        )

    def _print_period(self, param_name: str, param: torch.Tensor, period: int) -> None:
        if not self.verbose:
            return
        print(
            "Gefen shared-v:",
            param_name,
            "| shape=",
            tuple(param.shape),
            "| numel=",
            param.numel(),
            "| automatic period=",
            period,
        )

    def _print_v_period(
        self,
        param_name: str,
        param: torch.Tensor,
        shared_v: torch.Tensor,
        grad: Optional[torch.Tensor] = None,
    ) -> None:
        if not self.verbose:
            return
        shared_v_numel = shared_v.numel()
        param_numel = param.numel()
        if grad is None:
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
        period = param_numel // shared_v_numel
        self._print_period(param_name, param, period)

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
        self._print_gefen_codebook("Gefen codebook", codebook)
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

    def _has_restored_automatic_periods(self) -> bool:

        return any(
            "automatic_period" in self.state.get(p, {})
            for group in self.param_groups
            for p in group["params"]
        )

    def _iter_network_structure_records(self):
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state.get(p, {})
                if "automatic_period" not in state:
                    raise ValueError(
                        "Cannot save Gefen network structure because automatic_period is missing for {}".format(
                            group["name"],
                        )
                    )
                yield {
                    "name": group["name"],
                    "shape": tuple(p.shape),
                    "parameter_shape": tuple(
                        state.get("parameter_shape", tuple(p.shape))
                    ),
                    "numel": p.numel(),
                    "automatic_period": int(state["automatic_period"]),
                }

    def _save_network_structure(self) -> None:

        records = list(self._iter_network_structure_records())
        if len(records) == 0:
            raise ValueError(
                "Cannot save Gefen network structure because no parameters were available."
            )
        metadata = {
            "version": NETWORK_STRUCTURE_VERSION,
            "parameters": records,
        }
        save_structure_and_codebook = globals().get(
            "SAVE_STRUCTURE_AND_CODEBOOK", False
        )
        if save_structure_and_codebook:
            if self._gefen_codebook is None:
                raise ValueError("Cannot save Gefen codebook before it is initialized.")
            metadata["codebook"] = self._gefen_codebook.detach().cpu()
        torch.save(
            metadata,
            self._network_structure_path,
        )
        if save_structure_and_codebook:
            print(
                "Saved Gefen network structure and codebook to {}".format(
                    self._network_structure_path.resolve()
                )
            )
        elif not save_structure_and_codebook:
            print(
                "Saved Gefen network structure to {}".format(
                    self._network_structure_path.resolve()
                )
            )
        self._saved_network_structure = True

    def _load_network_structure(self) -> None:
        if not self._network_structure_path.exists():
            raise FileNotFoundError(
                "Expected Gefen network structure file at {}".format(
                    self._network_structure_path,
                )
            )

        metadata = torch.load(self._network_structure_path, map_location="cpu")
        if not isinstance(metadata, dict):
            raise TypeError("Gefen network structure file must contain a dict.")
        elif metadata.get("version") != NETWORK_STRUCTURE_VERSION:
            raise ValueError(
                "Unsupported Gefen network structure version: {}".format(
                    metadata.get("version"),
                )
            )

        saved_records = metadata.get("parameters")
        if not isinstance(saved_records, list):
            raise TypeError(
                "Gefen network structure file must contain a parameters list."
            )

        current_entries = [
            (group["name"], p) for group in self.param_groups for p in group["params"]
        ]
        if len(saved_records) != len(current_entries):
            raise ValueError(
                "Gefen network structure parameter count mismatch: file has {}, optimizer has {}".format(
                    len(saved_records),
                    len(current_entries),
                )
            )

        for index, (record, (param_name, param)) in enumerate(
            zip(saved_records, current_entries)
        ):
            if not isinstance(record, dict):
                raise TypeError(
                    "Gefen network structure parameter record {} must be a dict.".format(
                        index
                    )
                )

            expected_shape = tuple(param.shape)
            saved_shape = tuple(record.get("shape", ()))
            saved_name = record.get("name")
            saved_numel = record.get("numel")
            saved_period = record.get("automatic_period")

            if saved_name != param_name:
                raise ValueError(
                    "Gefen network structure parameter name mismatch at index {}: file has {}, optimizer has {}".format(
                        index,
                        saved_name,
                        param_name,
                    )
                )
            elif saved_shape != expected_shape:
                raise ValueError(
                    "Gefen network structure shape mismatch for {}: file has {}, optimizer has {}".format(
                        param_name,
                        saved_shape,
                        expected_shape,
                    )
                )
            elif saved_numel != param.numel():
                raise ValueError(
                    "Gefen network structure numel mismatch for {}: file has {}, optimizer has {}".format(
                        param_name,
                        saved_numel,
                        param.numel(),
                    )
                )
            elif not isinstance(saved_period, int):
                raise TypeError(
                    "Gefen automatic_period for {} must be an int.".format(param_name)
                )
            elif saved_period <= 0:
                raise ValueError(
                    "Gefen automatic_period for {} must be positive.".format(param_name)
                )
            elif param.numel() % saved_period != 0:
                raise ValueError(
                    "Gefen automatic_period {} from file does not divide parameter {} with numel {}".format(
                        saved_period,
                        param_name,
                        param.numel(),
                    )
                )

            self.state[param]["automatic_period"] = saved_period

        save_structure_and_codebook = globals().get(
            "SAVE_STRUCTURE_AND_CODEBOOK", False
        )
        if save_structure_and_codebook:
            codebook = metadata.get("codebook")
            if not torch.is_tensor(codebook):
                raise TypeError(
                    "Gefen network structure file must contain a tensor codebook."
                )
            self._gefen_codebook = codebook
            print(
                "Loaded Gefen network structure and codebook from {}".format(
                    self._network_structure_path.resolve()
                )
            )
        elif not save_structure_and_codebook:
            print(
                "Loaded Gefen network structure from {}".format(
                    self._network_structure_path.resolve()
                )
            )
        self._loaded_network_structure = True

    def _maybe_refresh_gefen_codebook(self) -> None:

        if self._gefen_codebook is not None:

            device = self._gefen_codebook_device()
            if self._gefen_codebook.device != device:
                self._gefen_codebook = self._gefen_codebook.to(device)
            return

        restored_periods = self._has_restored_automatic_periods()
        if (
            self.load_structure
            and not restored_periods
            and not self._loaded_network_structure
        ):

            self._load_network_structure()
            restored_periods = True
            if self._gefen_codebook is not None:
                device = self._gefen_codebook_device()
                if self._gefen_codebook.device != device:
                    self._gefen_codebook = self._gefen_codebook.to(device)
                return

        self._ensure_gefen_codebook(reuse_existing_periods=restored_periods)
        if (
            self.save_structure
            and self._gefen_codebook is not None
            and not self._saved_network_structure
        ):
            self._save_network_structure()

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
        state_dtype = gefen_state_dtype_from_grad(grad_view)
        if grad_view.shape[1] == 1:
            period_one_divisor = self._period_one_divisor_for_numel(grad_view.numel())
            state["period_one_divisor"] = period_one_divisor
            magnitude_rows = grad_view.numel() // period_one_divisor
        elif grad_view.shape[1] > 1:
            magnitude_rows = grad_view.shape[0]
        else:
            raise ValueError(
                "Expected grad_view second dimension to be positive, got {}".format(
                    grad_view.shape[1]
                )
            )

        state["m_magnitude"] = torch.zeros_like(
            grad_view[:magnitude_rows, 0:1],
            dtype=state_dtype,
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

    def _automatic_gefen_fused_update_from_vmean(
        self,
        p,
        state,
        grad_view,
        beta1,
        lr,
        bias_correction_1,
        bias_correction_2,
        eps,
    ) -> None:

        codebook = self._gefen_codebook
        if codebook is None:
            raise ValueError(
                "Expected Gefen codebook to be initialized before fused update."
            )
        gefen_automatic_fused_update_from_vmean(
            p=p,
            grad_view=grad_view,
            m_codebook=state["m_codebook"],
            m_magnitude=state["m_magnitude"],
            vmean=state["vmean"],
            codebook=codebook,
            packed_indices=False,
            beta1=beta1,
            lr=lr,
            bias_correction_1=bias_correction_1,
            bias_correction_2=bias_correction_2,
            eps=eps,
        )

    def _automatic_gefen_period1_update(
        self,
        p,
        state,
        grad_flat,
        beta1,
        beta2,
        lr,
        bias_correction_1,
        bias_correction_2,
        eps,
    ) -> None:

        codebook = self._gefen_codebook
        if codebook is None:
            raise ValueError(
                "Expected Gefen codebook to be initialized before fused period-1 update."
            )
        gefen_automatic_period1_update(
            p=p,
            grad_flat=grad_flat,
            m_codebook=state["m_codebook"],
            m_magnitude=state["m_magnitude"],
            vmean=state["vmean"],
            codebook=codebook,
            period_one_divisor=self._period_one_divisor_from_state(
                state, grad_flat.numel()
            ),
            beta1=beta1,
            beta2=beta2,
            lr=lr,
            bias_correction_1=bias_correction_1,
            bias_correction_2=bias_correction_2,
            eps=eps,
        )

    def _automatic_momentum_update_nonfused(
        self, state, grad_view: torch.Tensor, beta1: float
    ) -> torch.Tensor:

        if grad_view.dim() != 2:
            raise ValueError(
                "Automatic shared momentum expects a 2D tensor, got dim={}".format(
                    grad_view.dim()
                )
            )

        period = state["automatic_period"]
        current_m_fp32 = (
            self._gefen_dequantize_m_coefficients_notfused(state, grad_view).float()
            * self._m_magnitude_view_for_m(state, grad_view).float()
        )

        updated_m_fp32 = current_m_fp32.lerp(grad_view.float(), 1 - beta1)

        if period == 1:
            magnitude_period = self._period_one_divisor_from_state(
                state, updated_m_fp32.numel()
            )
        elif period > 1:
            magnitude_period = period
        else:
            raise ValueError("automatic_period must be positive, got {}".format(period))
        magnitude_fp32 = self._automatic_reduce(
            updated_m_fp32.abs().reshape(-1), magnitude_period, reduce_op="max"
        )
        state["m_magnitude"].copy_(magnitude_fp32)
        magnitude_for_m = self._m_magnitude_view_for_m(state, updated_m_fp32).float()
        nonzero_mask = magnitude_for_m > 0
        updated_m_fp32.div_(magnitude_for_m)
        updated_m_fp32.masked_fill_(~nonzero_mask, 0.0)

        indices = self._gefen_nearest_indices(updated_m_fp32)
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
        return quantized_m * self._m_magnitude_view_for_m(state, grad_view)

    def _step_automatic(
        self, group, param_name: str, p: torch.Tensor, grad: torch.Tensor
    ) -> None:
        local_p = self._resolve_local_tensor(p, "parameter {}".format(param_name))
        grad = self._resolve_local_tensor(grad, "gradient {}".format(param_name))
        state = self.state[p]
        beta1 = group["beta1"]
        beta2 = group["beta2"]
        lr = group["lr"]
        eps = group["eps"]

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
            grad_view = self._automatic_view(flat_grad, automatic_period)
            self._init_gefen_state(state, grad_view)
            state_dtype = gefen_state_dtype_from_grad(grad_view)
            state["vmean"] = torch.zeros_like(
                grad_view[:, 0:1],
                dtype=state_dtype,
                memory_format=torch.preserve_format,
            )
            self._print_v_period(param_name, local_p, state["vmean"], grad)

        automatic_period = state["automatic_period"]
        grad_view = self._automatic_view(flat_grad, automatic_period)
        use_fused_gefen_step = self._use_fused_gefen_automatic_step()
        use_period1_fused_step = use_fused_gefen_step and automatic_period == 1

        if not use_period1_fused_step:
            self._automatic_vmean_update(state["vmean"], grad_view, beta2)

        state["step"] += 1

        if group["weight_decay"] > 0.0:
            local_p.mul_(1 - lr * group["weight_decay"])

        bias_correction_1 = 1 - beta1 ** state["step"]
        bias_correction_2 = 1 - beta2 ** state["step"]

        if use_fused_gefen_step:
            if local_p.is_contiguous():
                fused_p = local_p
            elif hasattr(p, "to_local") and hasattr(p, "placements"):

                fused_p = local_p.contiguous()
            else:
                raise ValueError(
                    "Gefen fused update requires a contiguous parameter tensor for {}.".format(
                        param_name
                    )
                )
            if use_period1_fused_step:
                self._automatic_gefen_period1_update(
                    fused_p,
                    state,
                    flat_grad,
                    beta1,
                    beta2,
                    lr,
                    bias_correction_1,
                    bias_correction_2,
                    eps,
                )
            elif not use_period1_fused_step:
                self._automatic_gefen_fused_update_from_vmean(
                    fused_p,
                    state,
                    grad_view,
                    beta1,
                    lr,
                    bias_correction_1,
                    bias_correction_2,
                    eps,
                )
            if fused_p is not local_p:
                local_p.copy_(fused_p)

        elif not use_fused_gefen_step:

            h = (state["vmean"].float().sqrt() / math.sqrt(bias_correction_2)).add_(eps)
            stepsize = (1 / bias_correction_1) / h
            shared_m = self._automatic_momentum_update_nonfused(state, grad_view, beta1)
            update = (shared_m * stepsize).view(grad.size())
            update.mul_(lr)
            local_p.add_(-update)

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

        state_dict["gefen_codebook"] = self._gefen_codebook
        return state_dict

    def load_state_dict(self, state_dict):

        state_dict = dict(state_dict)

        gefen_global_step = state_dict.pop("gefen_global_step", 0)

        saved_state = state_dict.get("state", {}) or {}

        gefen_codebook = state_dict.pop("gefen_codebook", None)

        super().load_state_dict(state_dict)
        self._gefen_global_step = gefen_global_step
        if self._cap_period_by_shape_enabled():
            for group in self.param_groups:
                for param in group["params"]:
                    self.state[param].setdefault("parameter_shape", tuple(param.shape))

        self._gefen_codebook = gefen_codebook

        id_map = dict(
            zip(
                chain.from_iterable(
                    group["params"] for group in state_dict["param_groups"]
                ),
                chain.from_iterable(group["params"] for group in self.param_groups),
            )
        )
        for param_id, saved_param_state in saved_state.items():
            param = id_map.get(param_id)
            if param is None:
                continue
            live_state = self.state.get(param)
            if live_state is None:
                continue
            for key, saved_value in saved_param_state.items():
                if not torch.is_tensor(saved_value):
                    continue
                live_value = live_state.get(key)
                if (
                    torch.is_tensor(live_value)
                    and live_value.dtype == saved_value.dtype
                ):
                    continue
                device = (
                    live_value.device if torch.is_tensor(live_value) else param.device
                )
                live_state[key] = saved_value.to(device=device, dtype=saved_value.dtype)

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
